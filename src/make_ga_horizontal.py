"""
Regenerate Graphical Abstract - HORIZONTAL (CMS requires 1328x531+ px, ratio ~2.5:1)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['font.size'] = 10
rcParams['pdf.fonttype'] = 42
rcParams['savefig.dpi'] = 300

FIG_DIR = Path('paper/figures')
CB = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00']
LIGHT = {'phys': '#D6EAF8', 'ml': '#D5F5E3', 'uq': '#FDEBD0', 'ad': '#FADBD8', 'out': '#E8DAEF'}

def box(ax, x, y, w, h, text, color, fs=9):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
                       fc=color, ec='#333333', lw=1.0)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, wrap=True)

def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle='->', mutation_scale=10, lw=1.2, color='#333333'))

# 横向布局 (宽13, 高5.2, 比例~2.5:1)
fig, ax = plt.subplots(figsize=(13.28, 5.31))  # 1328x531 at 100dpi base
ax.set_xlim(0, 26)
ax.set_ylim(0, 10)
ax.axis('off')

# 左: 输入
box(ax, 0.5, 3.5, 3.5, 3, 'ABO3 perovskite\n4,914 compounds\n110 features', LIGHT['phys'], fs=10)
arrow(ax, 4, 5, 5, 5)

# 中左: 两个并行层
box(ax, 5, 6, 4.5, 2.5, 'Physics baseline\nKernelRidge\n(interpretability anchor)', LIGHT['phys'], fs=9)
box(ax, 5, 1, 4.5, 2.5, 'GBDT stacking\nLGB+XGB+HGB\n(point prediction)', LIGHT['ml'], fs=9)
arrow(ax, 3.5, 5.5, 5, 6.8)
arrow(ax, 3.5, 4.5, 5, 2.2)

# 中: 融合
box(ax, 10, 3.5, 3, 3, 'Point prediction\nR2=0.914\n(formation E)', LIGHT['out'], fs=10)
arrow(ax, 9.5, 7, 10, 5.5)  # phys->point
arrow(ax, 9.5, 2.2, 10, 4.5)  # ml->point

# 中右: 不确定性(并行)
box(ax, 14, 6.5, 4, 2.5, 'CQR intervals\nECE reduced 43-60%\n(PICP >= 0.80)', LIGHT['uq'], fs=9)
box(ax, 14, 0.8, 4, 2.5, 'Applicability domain\nsigma/kNN/leverage\n(trusted vs untrusted)', LIGHT['ad'], fs=9)
arrow(ax, 13, 5.5, 14, 7)
arrow(ax, 13, 4.5, 14, 2)

# 右: 输出
box(ax, 19, 3, 6, 4,
    'Reliable perovskite\nstability prediction\nwith calibrated\nuncertainty', '#F9E79F', fs=11)
arrow(ax, 18, 7, 19, 5.5)
arrow(ax, 18, 2, 19, 4.5)

# 底部标注
ax.text(7.25, 9.2, 'Method', fontsize=11, ha='center', fontweight='bold', color='#555555')
ax.text(16, 9.2, 'Uncertainty & Trust', fontsize=11, ha='center', fontweight='bold', color='#555555')

fig.tight_layout()
fig.savefig(FIG_DIR / 'Graphical_Abstract.pdf', format='pdf')
fig.savefig(FIG_DIR / 'Graphical_Abstract.png', format='png', dpi=300)
plt.close(fig)
print('[SAVE] Graphical_Abstract regenerated (horizontal, 300dpi)')

# 验证
from PIL import Image
img = Image.open(FIG_DIR / 'Graphical_Abstract.png')
w, h = img.size
dpi = img.info.get('dpi', (0, 0))
print('New size: %dx%d (w x h)' % (w, h))
print('Ratio: %.2f (target ~2.5)' % (w/h))
print('DPI: %s' % str(dpi))
print('Compliant: %s' % (w >= 1328 and h >= 531 and dpi[0] >= 300))
