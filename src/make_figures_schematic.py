"""
F1 工作流图 + Graphical Abstract (schematic, 无数据)
CMS规范: 矢量PDF, Arial 7pt, 色盲友好
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['font.size'] = 7
rcParams['pdf.fonttype'] = 42
rcParams['savefig.dpi'] = 500

FIG_DIR = Path('paper/figures')
CB = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00']
LIGHT = {'phys': '#D6EAF8', 'ml': '#D5F5E3', 'uq': '#FDEBD0', 'ad': '#FADBD8', 'out': '#E8DAEF'}

def box(ax, x, y, w, h, text, color, fs=7):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
                       fc=color, ec='#333333', lw=0.8)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, wrap=True)

def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle='->', mutation_scale=8, lw=1.0, color='#333333'))


# === F1: 工作流图 (双栏宽) ===
fig, ax = plt.subplots(figsize=(7.4, 3.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
ax.axis('off')

# 输入
box(ax, 0.2, 2.0, 1.6, 1.5, 'Input\n110 features\n(96 Magpie +\n14 physics)', LIGHT['phys'])
# 物理层
box(ax, 2.3, 3.3, 1.8, 1.0, 'Physics baseline\nKernelRidge\n$\\mu_p$ (R²=0.77)', LIGHT['phys'])
# ML残差
box(ax, 2.3, 1.2, 1.8, 1.5, 'ML residual\nGBDT stacking\n(LGB+XGB+HGB)\n$\\mu_r$', LIGHT['ml'])
# 点预测
box(ax, 4.7, 2.2, 1.5, 1.0, 'Point prediction\n$\\mu = \\mu_p + \\mu_r$\n(R²=0.914)', LIGHT['out'])
# CQR区间
box(ax, 4.7, 3.8, 1.5, 1.0, 'CQR interval\n[q$_{10}$−d, q$_{90}$+d]\n(PICP≥0.80)', LIGHT['uq'])
# 应用域
box(ax, 4.7, 0.5, 1.5, 1.0, 'Applicability\ndomain\n(σ/kNN/leverage)', LIGHT['ad'])
# 输出
box(ax, 7.2, 1.8, 2.5, 1.8,
    'Output\n• Point estimate\n• Adaptive 80% interval\n• Trust label\n(reliable/unreliable)', LIGHT['out'])

# 箭头
arrow(ax, 1.8, 2.75, 2.3, 3.7)   # input→phys
arrow(ax, 1.8, 2.75, 2.3, 1.9)   # input→ml
arrow(ax, 4.1, 3.7, 4.7, 3.3)    # phys→mu (part of point)
arrow(ax, 4.1, 1.9, 4.7, 2.5)    # ml→mu
arrow(ax, 6.2, 2.7, 7.2, 2.7)    # point→out
arrow(ax, 6.2, 4.0, 6.7, 3.5); arrow(ax, 6.7, 3.5, 7.2, 3.2)  # cqr→out
arrow(ax, 6.2, 0.9, 6.7, 1.8); arrow(ax, 6.7, 1.8, 7.2, 2.1)  # ad→out

# 物理层→ML残差 (残差连接)
ax.annotate('', xy=(3.2, 2.7), xytext=(3.2, 3.3),
            arrowprops=dict(arrowstyle='->', lw=0.7, color='gray', ls='--'))
ax.text(3.35, 3.0, 'residual', fontsize=5, color='gray', style='italic')

# 分组标注
ax.text(3.2, 4.55, 'Interpretability anchor', fontsize=6, ha='center', color=CB[0], style='italic')
ax.text(3.2, 0.85, 'Accuracy', fontsize=6, ha='center', color=CB[2], style='italic')
ax.text(5.45, 5.05, 'Uncertainty (decoupled)', fontsize=6, ha='center', color=CB[1], style='italic')

fig.tight_layout()
fig.savefig(FIG_DIR / 'Figure_1_workflow.pdf', format='pdf')
fig.savefig(FIG_DIR / 'Figure_1_workflow.png', format='png')
plt.close(fig)
print('[SAVE] Figure_1_workflow')


# === Graphical Abstract (531x1328 px, 纵向) ===
fig, ax = plt.subplots(figsize=(2.6, 6.5))  # ~531x1328px at 200dpi ratio
ax.set_xlim(0, 10); ax.set_ylim(0, 20)
ax.axis('off')

box(ax, 1, 17.5, 8, 1.8, 'ABO₃ perovskite\n(4914 compounds)', LIGHT['phys'], fs=8)
arrow(ax, 5, 17.5, 5, 16.8)

box(ax, 1, 14.5, 3.5, 2.0, 'Physics baseline\n(KernelRidge)\n$\\mu_p$', LIGHT['phys'], fs=7)
box(ax, 5.5, 14.5, 3.5, 2.0, 'GBDT stacking\n(LGB+XGB+HGB)\nresidual $\\mu_r$', LIGHT['ml'], fs=7)
arrow(ax, 3.0, 16.8, 3.0, 16.5)
arrow(ax, 7.0, 16.8, 7.0, 16.5)
arrow(ax, 4.5, 15.5, 5.0, 14.0)

box(ax, 2, 12.5, 6, 1.5, 'Point prediction $\\mu$\nR² = 0.914 (formation)', LIGHT['out'], fs=8)
arrow(ax, 5, 12.5, 5, 11.8)

box(ax, 1, 9.5, 8, 2.0, 'Conditional conformal (CQR)\nSample-adaptive intervals\nECE ↓ 43–60% vs standard', LIGHT['uq'], fs=7)
arrow(ax, 5, 9.5, 5, 8.8)

box(ax, 1, 6.5, 8, 2.0, 'Applicability domain\n(σ / k-NN / leverage)\nTrusted R² = 0.945', LIGHT['ad'], fs=7)
arrow(ax, 5, 6.5, 5, 5.8)

box(ax, 1, 3.5, 8, 2.0, 'Extrapolation (LOEO, 68 elements)\n+ Candidate screening\n(9 candidates, 3-tier validation)', LIGHT['phys'], fs=7)
arrow(ax, 5, 3.5, 5, 2.8)

box(ax, 1.5, 0.8, 7, 1.8, 'Reliable perovskite\nstability prediction\nwith uncertainty', '#F9E79F', fs=8)

fig.savefig(FIG_DIR / 'Graphical_Abstract.pdf', format='pdf')
fig.savefig(FIG_DIR / 'Graphical_Abstract.png', format='png', dpi=200)
plt.close(fig)
print('[SAVE] Graphical_Abstract')
