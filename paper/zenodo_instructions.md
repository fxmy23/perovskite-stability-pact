# Zenodo 存档操作说明 (为获取DOI)

## 为什么要做
CMS 投稿要求 FAIR 数据。GitHub 链接是代码，但 GitHub 仓库不是永久 DOI 引用。
Zenodo 给仓库一个**永久 DOI**，审稿人/读者可以稳定引用。

## 操作步骤 (5分钟)

### 第1步: 登录 Zenodo
1. 打开 https://zenodo.org
2. 右上角 "Log in" → 选 "GitHub" → 用你的 GitHub 账号 (fxmy23) 授权登录

### 第2步: 关联 GitHub 仓库
1. 登录后, 点右上角用户名 → "Settings" → "GitHub" 标签
2. 找到 `fxmy23/perovskite-stability-pact`, 点 "Enable" (开启同步)
3. (Zenodo 会自动检测你的 GitHub 仓库)

### 第3步: 创建 Release (触发 DOI 生成)
在你的 GitHub 仓库创建一个 release:
1. 打开 https://github.com/fxmy23/perovskite-stability-pact/releases
2. 点 "Create a new release" / "Draft a new release"
3. **Choose a tag**: 输入 `v1.0.0` → 点 "Create new tag: v1.0.0 on publish"
4. **Release title**: `v1.0.0 - Initial submission to Computational Materials Science`
5. **Description** (复制粘贴):

```
## Uncertainty-Aware Prediction of Perovskite Oxide Stability

Code and data for the manuscript submitted to Computational Materials Science.

### Key results
- Formation energy R² = 0.914, Hull energy R² = 0.800
- CQR conditional coverage: ECE improved 43-60% over standard conformal
- Applicability domain: trusted R² = 0.945 vs untrusted 0.886
- 68-element LOEO extrapolation assessment
- Symbolic regression consensus (a_site_en 100%)

### Reproduce
```bash
pip install -r requirements.txt
python src/pact_final.py
```

Author: Xumingyong Feng (Weiyang College, Tsinghua University)
```

6. 点 "Publish release"

### 第4步: 获取 DOI
1. 发布后, Zenodo 会自动同步并生成 DOI (约1-2分钟)
2. 打开 https://zenodo.org → 你的账号 → "Uploads" 或看邮件通知
3. DOI 格式类似: `10.5281/zenodo.XXXXXXX`

### 第5步: 填回论文
把 DOI 填入:
- `paper/manuscript.md` 的 Data Availability (替换 `[DOI to be inserted upon acceptance]`)
- `README.md` 的 Data 部分

## 注意事项
- DOI 一旦生成不可删除 (Zenodo 政策), 但可发布新版本 (新 DOI + 指向旧版本)
- 建议在论文**正式录用后**再做 v1.0 release (内容已定稿)
- 投稿阶段可先用 GitHub 链接, 录用后补 Zenodo DOI

## 如果不想用 Zenodo (替代方案)
Elsevier 也接受 Mendeley Data (https://data.mendeley.com) 存档, 流程类似。
