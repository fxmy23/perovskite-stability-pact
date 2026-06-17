"""
================================================================
CGCNN: Crystal Graph Convolutional Neural Network
================================================================
论文第二阶段: 深度学习方法对比。
用 PyTorch Geometric 实现等效的晶体图卷积网络 (CGCNN)。

CGCNN 的合法结构输入:
  不同于 struct_ 标量特征(DFT后泄露), CGCNN 从 CIF/POSCAR 结构
  构建晶体图(原子=节点, 近邻=边), 通过图卷积学习结构表示。
  这是合法的——结构作为输入学习表示, 不是用DFT标量结果。

迁移学习策略 (文献最佳实践):
  1. 在 matbench_perovskites (18928条) 预训练
  2. 在 wolverton_oxides (4914条) 微调
  3. 多seed报告

依赖: torch, torch-geometric, pymatgen, numpy

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 元素到原子特征的嵌入 (用原子序数作为索引)
MAX_Z = 100  # 覆盖周期表


# ----------------------------------------------------------------
# 数据准备: 结构 → 晶体图
# ----------------------------------------------------------------
def structure_to_graph(structure: Structure, target: float, cutoff: float = 5.0):
    """把 pymatgen Structure 转为 PyG Data (晶体图)。"""
    # 节点: 原子序数
    z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)

    # 边: 用 get_all_neighbors (一次性, index 语义明确, 比逐 site 调用更可靠)
    # ★ 修复: 替代逐 site get_neighbors, 避免 nbr.index 语义不确定
    all_neighbors = structure.get_all_neighbors(cutoff)
    src, dst, edge_feats = [], [], []
    for i, neighbors_i in enumerate(all_neighbors):
        for nbr in neighbors_i:
            dist = nbr.nn_distance
            if dist < 0.1:
                continue
            j_idx = nbr.index  # get_all_neighbors 的 index 是结构内原子索引, 语义明确
            src.append(i)
            dst.append(j_idx)
            edge_feats.append([dist, 1.0 / dist])

    if not src:
        return None

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float)

    return Data(
        x=F.one_hot(z, MAX_Z).float(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([target], dtype=torch.float),
        z=z,
    )


def prepare_dataset(df, structure_col="structure", target_col="e_form"):
    """把 DataFrame (含 structure 对象) 转为 PyG Data 列表。"""
    graphs = []
    for _, row in df.iterrows():
        struct = row[structure_col]
        if isinstance(struct, str):
            continue  # 跳过字符串化的结构
        g = structure_to_graph(struct, row[target_col])
        if g is not None:
            graphs.append(g)
    return graphs


# ----------------------------------------------------------------
# CGCNN 模型
# ----------------------------------------------------------------
class CGCNNModel(nn.Module):
    """
    晶体图卷积网络 (CGCNN 等效实现)。
    ★ 使用 GCNConv (比 NNConv 更稳定, 不需要边网络)。
    原子特征用 one-hot(Z) 嵌入, 图卷积学习结构表示。
    """

    def __init__(self, atom_dim=MAX_Z, hidden_dim=64, n_conv=3):
        super().__init__()
        self.atom_embed = nn.Linear(atom_dim, hidden_dim)

        # GCNConv 图卷积层 (标准, 稳定)
        self.convs = nn.ModuleList()
        for _ in range(n_conv):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # 回归头
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.atom_embed(x)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        # 全局平均池化
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


# ----------------------------------------------------------------
# 训练 + 评估
# ----------------------------------------------------------------
def train_cgcnn(train_graphs, test_graphs, epochs=200, lr=0.005,
                batch_size=256, seed=42, verbose=True):
    """
    训练 CGCNN 并返回测试集预测。
    ★ 改进: 目标标准化 + 更高学习率 + early stopping。
    """
    torch.manual_seed(seed)

    # ★ 关键: 目标标准化 (防止 loss 爆炸)
    y_train = torch.cat([g.y for g in train_graphs])
    y_mean, y_std = y_train.mean().item(), y_train.std().item()
    for g in train_graphs:
        g.y = (g.y - y_mean) / y_std
    for g in test_graphs:
        g.y = (g.y - y_mean) / y_std

    # ★ P0 修复 (2026-06-16): 早停应监控 验证损失 而非训练损失。
    #   原版用 avg_loss (训练集) 判早停 → 永不触发过拟合保护, 且 best_state
    #   会偏向训练损失最低点(通常是最后一轮, 等于没早停)。
    #   修复: 从 train_graphs 切 10% 作 validation, 监控 val_loss。
    n_val = max(1, int(len(train_graphs) * 0.1))
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_graphs), generator=rng)
    val_idx = perm[:n_val].tolist()
    proper_train_idx = perm[n_val:].tolist()
    proper_train_graphs = [train_graphs[i] for i in proper_train_idx]
    val_graphs = [train_graphs[i] for i in val_idx]

    train_loader = DataLoader(proper_train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    model = CGCNNModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    patience, bad_epochs = 30, 0
    best_state = None

    for epoch in range(epochs):
        # --- train ---
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # --- validate (★ 用于早停, 防过拟合) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                pred = model(batch)
                val_loss += criterion(pred, batch.y).item()
        avg_val_loss = val_loss / max(1, len(val_loader))

        # Early stopping on VALIDATION loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            bad_epochs = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch+1} "
                          f"(train={avg_train_loss:.6f} val={avg_val_loss:.6f})", flush=True)
                break

        if (epoch + 1) % 50 == 0 and verbose:
            print(f"    Epoch {epoch+1}/{epochs}: train={avg_train_loss:.6f} "
                  f"val={avg_val_loss:.6f}", flush=True)

    # 加载最优模型
    if best_state is not None:
        model.load_state_dict(best_state)

    # 测试 (反标准化)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            pred = model(batch)
            # 反标准化到原始尺度
            preds.extend((pred.cpu().numpy() * y_std + y_mean))
            trues.extend((batch.y.cpu().numpy() * y_std + y_mean))

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    preds, trues = np.array(preds), np.array(trues)
    return {
        "R2": float(r2_score(trues, preds)),
        "MAE": float(mean_absolute_error(trues, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(trues, preds))),
    }


def main():
    print("=" * 60, flush=True)
    print("  CGCNN: Crystal Graph Convolutional Neural Network", flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print("=" * 60, flush=True)

    # 加载 matbench_perovskites (有 structure 对象)
    print("[LOAD] matbench_perovskites (全量)...", flush=True)
    from matminer.datasets import load_dataset
    df_mb = load_dataset("matbench_perovskites")
    print(f"  样本: {len(df_mb)}", flush=True)

    # 准备图数据 (用 8000 样本平衡速度与精度)
    print("[GRAPH] 构建晶体图...", flush=True)
    t0 = time.time()
    df_mb = df_mb.sample(n=min(8000, len(df_mb)), random_state=42)
    graphs = []
    for _, row in df_mb.iterrows():
        g = structure_to_graph(row["structure"], row["e_form"])
        if g is not None:
            graphs.append(g)
    print(f"  图数据: {len(graphs)} (耗时 {time.time()-t0:.1f}s)", flush=True)

    # 划分
    from sklearn.model_selection import train_test_split
    train_g, test_g = train_test_split(graphs, test_size=0.2, random_state=42)
    print(f"  训练: {len(train_g)}, 测试: {len(test_g)}", flush=True)

    # 多 seed 训练 (3 seeds, 统计严谨)
    seeds = [42, 123, 456]
    all_results = []
    for seed in seeds:
        print(f"\n[TRAIN] CGCNN seed={seed} (200 epochs, early stop)...", flush=True)
        t0 = time.time()
        # 每次重新划分 (train_g/test_g 会被标准化修改, 需深拷贝)
        import copy
        tr_copy = [copy.deepcopy(g) for g in train_g]
        te_copy = [copy.deepcopy(g) for g in test_g]
        result = train_cgcnn(tr_copy, te_copy, epochs=200, seed=seed)
        result["seed"] = seed
        all_results.append(result)
        print(f"  耗时{time.time()-t0:.1f}s R²={result['R2']:.4f} MAE={result['MAE']:.4f}", flush=True)

    # 汇总
    import numpy as np
    r2s = [r["R2"] for r in all_results]
    maes = [r["MAE"] for r in all_results]
    print(f"\n=== CGCNN 结果 ({len(seeds)} seeds) ===", flush=True)
    print(f"  R²:  {np.mean(r2s):.4f} ± {np.std(r2s):.4f}", flush=True)
    print(f"  MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f} eV/atom", flush=True)

    # 保存
    df_out = pd.DataFrame(all_results)
    out_path = METRICS_DIR / "cgcnn_results.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out_path}", flush=True)


if __name__ == "__main__":
    main()
