"""
================================================================
论文图表统一生成 (CMS投稿规范)
================================================================
规范 (paper/cms_submission_specs.md):
- 字体: Arial 7pt正文/6pt下标
- 配色: 色盲友好 (viridis/colorblind)
- 分辨率: 组合图500dpi, 矢量PDF优先
- 坐标轴: 必须有标签+单位
- 尺寸: 单栏90mm, 双栏190mm

生成: F1-F6 + Graphical Abstract
作者: 清华大学材料学院本科生
================================================================
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# === CMS 统一样式 ===
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
rcParams['font.size'] = 7
rcParams['axes.labelsize'] = 7
rcParams['axes.titlesize'] = 8
rcParams['xtick.labelsize'] = 6
rcParams['ytick.labelsize'] = 6
rcParams['legend.fontsize'] = 6
rcParams['axes.linewidth'] = 0.6
rcParams['xtick.major.width'] = 0.6
rcParams['ytick.major.width'] = 0.6
rcParams['lines.linewidth'] = 1.0
rcParams['savefig.dpi'] = 500
rcParams['savefig.bbox'] = 'tight'
rcParams['pdf.fonttype'] = 42  # TrueType embed (Elsevier要求)
rcParams['ps.fonttype'] = 42

# 色盲友好配色
CB = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00', '#F0E442']
SIGMA_CMAP = 'viridis'

FIG_DIR = Path('paper/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS = Path('results/metrics')

def save(fig, name):
    """保存PDF(矢量)+PNG(预览)。"""
    fig.savefig(FIG_DIR / f'{name}.pdf', format='pdf')
    fig.savefig(FIG_DIR / f'{name}.png', format='png')
    plt.close(fig)
    print(f'  [SAVE] {name}.pdf + .png', flush=True)


# ================================================================
# F2: Parity plot (形成能 + 凸包能, 2面板)
# ================================================================
def fig2_parity():
    print('[F2] Parity plot...', flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    targets = [('formation_energy_per_atom', 'Formation energy (eV/atom)', '(a)'),
               ('energy_above_hull', 'Energy above hull (eV/atom)', '(b)')]
    for ax, (t, label, panel) in zip(axes, targets):
        df = pd.read_csv(METRICS / f'pact_final_oof_{t}.csv')
        x, y = df['y_true'].values, df['oof_mu'].values
        sig = df['oof_sigma'].values
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y, x); mae = mean_absolute_error(y, x)
        # 散点, 颜色=σ
        sc = ax.scatter(x, y, c=sig, cmap=SIGMA_CMAP, s=4, alpha=0.6, edgecolors='none')
        # y=x 参考线
        lim = [min(x.min(), y.min())-0.1, max(x.max(), y.max())+0.1]
        ax.plot(lim, lim, 'k--', lw=0.5, alpha=0.7, label='$y = x$')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f'DFT {label}')
        ax.set_ylabel(f'Predicted {label}')
        ax.set_aspect('equal')
        ax.text(0.05, 0.92, f'{panel}\n$R^2$ = {r2:.3f}\nMAE = {mae:.3f}',
                transform=ax.transAxes, va='top', fontsize=6,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='none'))
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('Ensemble $\\sigma$', fontsize=6)
        cb.ax.tick_params(labelsize=5)
    fig.tight_layout()
    save(fig, 'Figure_2_parity')


# ================================================================
# F3: Reliability diagram (CQR vs 标准conformal)
# ================================================================
def fig3_reliability():
    print('[F3] Reliability diagram...', flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    targets = [('formation_energy_per_atom', '(a) Formation energy'),
               ('energy_above_hull', '(b) Energy above hull')]
    for ax, (t, title) in zip(axes, targets):
        df = pd.read_csv(METRICS / f'pact_final_oof_{t}.csv')
        y = df['y_true'].values
        sig = df['oof_sigma'].values
        covered_cqr = ((y >= df['cqr_lower']) & (y <= df['cqr_upper'])).astype(float)
        covered_std = ((y >= df['std_lower']) & (y <= df['std_upper'])).astype(float)
        order = np.argsort(sig)
        bins = np.array_split(order, 10)
        emp_cqr = [covered_cqr[b].mean() for b in bins]
        emp_std = [covered_std[b].mean() for b in bins]
        bin_centers = np.arange(1, 11)
        ax.axhline(0.80, color='gray', ls=':', lw=0.5, label='Nominal (0.80)')
        ax.plot(bin_centers, emp_std, 'o-', color=CB[1], ms=3, lw=1.0, label='Standard conformal')
        ax.plot(bin_centers, emp_cqr, 's-', color=CB[0], ms=3, lw=1.0, label='CQR')
        ax.set_xlabel('Uncertainty decile (low $\\rightarrow$ high $\\sigma$)')
        ax.set_ylabel('Empirical coverage')
        ax.set_title(title, fontsize=7)
        ax.set_ylim(0.5, 1.0)
        ax.legend(loc='lower left', framealpha=0.9)
        ax.set_xticks([1, 5, 10])
    fig.tight_layout()
    save(fig, 'Figure_3_reliability')


# ================================================================
# F4: 应用域可视化 (可信/不可信分区)
# ================================================================
def fig4_ad():
    print('[F4] Applicability domain...', flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    targets = [('formation_energy_per_atom', '(a) Formation energy'),
               ('energy_above_hull', '(b) Energy above hull')]
    for ax, (t, title) in zip(axes, targets):
        df = pd.read_csv(METRICS / f'pact_final_oof_{t}.csv')
        trusted = df['trust_sigma'] == 1
        ax.scatter(df.loc[~trusted, 'oof_mu'], df.loc[~trusted, 'abs_err'],
                   c=CB[1], s=3, alpha=0.4, label=f'Untrusted (n={(~trusted).sum()})')
        ax.scatter(df.loc[trusted, 'oof_mu'], df.loc[trusted, 'abs_err'],
                   c=CB[0], s=3, alpha=0.4, label=f'Trusted (n={trusted.sum()})')
        ax.set_xlabel('Predicted value (eV/atom)')
        ax.set_ylabel('Absolute error (eV/atom)')
        ax.set_title(title, fontsize=7)
        ax.legend(loc='upper left', framealpha=0.9, markerscale=3)
    fig.tight_layout()
    save(fig, 'Figure_4_AD')


# ================================================================
# F5: LOEO 外推 (代表性元素条形图)
# ================================================================
def fig5_loeo():
    print('[F5] LOEO extrapolation...', flush=True)
    df = pd.read_csv(METRICS / 'loeo_sr_full_results.csv')
    # 选代表性元素 (好/中/差), 按R²排序取两端+中间
    df = df.dropna(subset=['r2_pure_ml']).sort_values('r2_pure_ml')
    n = len(df)
    idx = list(range(0, 5)) + list(range(n//2-2, n//2+3)) + list(range(n-5, n))
    idx = sorted(set(idx))
    sub = df.iloc[idx]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(sub))
    ax.bar(x - 0.15, sub['r2_pure_ml'], 0.3, color=CB[0], label='Pure ML')
    ax.bar(x + 0.15, sub['r2_sr_ml'], 0.3, color=CB[2], label='SR + ML')
    ax.set_xticks(x)
    ax.set_xticklabels(sub['element'].values, rotation=45, ha='right', fontsize=5)
    ax.set_ylabel('LOEO $R^2$ (leave-one-element-out)')
    ax.set_xlabel('Held-out A-site element')
    ax.axhline(0, color='gray', lw=0.4)
    ax.legend(loc='upper left')
    fig.tight_layout()
    save(fig, 'Figure_5_LOEO')


# ================================================================
# F6: SHAP 重要性 vs SR 共识
# ================================================================
def fig6_shap_sr():
    print('[F6] SHAP vs SR consensus...', flush=True)
    shap_df = pd.read_csv(METRICS / 'shap_importance.csv')
    # 形成能 top-15
    sf = shap_df[shap_df['target'] == 'formation_energy_per_atom'].sort_values(
        'shap_importance', ascending=False).head(15).copy()
    sf['short'] = sf['feature'].str.replace('magpie_', 'M:').str.replace('phys_', 'P:')
    # SR共识
    sr = pd.read_csv(METRICS / 'sr_ensemble_consensus.csv')
    sr['short'] = sr['feature_full'].str.replace('phys_', 'P:')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5), gridspec_kw={'width_ratios': [2, 1]})
    # 左: SHAP top-15
    colors = [CB[2] if 'P:' in s else CB[0] for s in sf['short']]
    ax1.barh(range(len(sf)), sf['shap_importance'], color=colors)
    ax1.set_yticks(range(len(sf)))
    ax1.set_yticklabels(sf['short'], fontsize=5)
    ax1.invert_yaxis()
    ax1.set_xlabel('SHAP importance')
    ax1.set_title('(a) LightGBM feature importance (top-15)', fontsize=7)
    # 右: SR共识
    sr_sorted = sr.sort_values('consensus_pct', ascending=True)
    ax2.barh(range(len(sr_sorted)), sr_sorted['consensus_pct'],
             color=[CB[1] if p >= 50 else CB[5] for p in sr_sorted['consensus_pct']])
    ax2.set_yticks(range(len(sr_sorted)))
    ax2.set_yticklabels(sr_sorted['short'], fontsize=5)
    ax2.set_xlabel('SR consensus (%)')
    ax2.set_title('(b) Symbolic regression consensus', fontsize=7)
    ax2.axvline(50, color='gray', ls=':', lw=0.5)
    fig.tight_layout()
    save(fig, 'Figure_6_SHAP_SR')


if __name__ == '__main__':
    fig2_parity()
    fig3_reliability()
    fig4_ad()
    fig5_loeo()
    fig6_shap_sr()
    print('\nAll data-driven figures generated.', flush=True)
