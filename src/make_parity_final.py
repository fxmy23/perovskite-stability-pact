"""
Parity Plot - publication quality (CMS规范)
X: DFT formation energy (eV/atom)
Y: ML predicted formation energy (eV/atom)
每个点 = 一个化合物, 颜色 = ensemble sigma
对角虚线 y = x
标注 R², MAE
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path

# === CMS 规范样式 ===
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['font.size'] = 8
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 10
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 8
rcParams['axes.linewidth'] = 0.8
rcParams['savefig.dpi'] = 500
rcParams['pdf.fonttype'] = 42

# === 读取真实数据 ===
df = pd.read_csv('results/metrics/pact_final_oof_formation_energy_per_atom.csv')
x_dft = df['y_true'].values
y_pred = df['oof_mu'].values
sigma = df['oof_sigma'].values

r2 = r2_score(x_dft, y_pred)
mae = mean_absolute_error(x_dft, y_pred)
rmse = np.sqrt(np.mean((x_dft - y_pred)**2))
n = len(x_dft)

print('Data: %d compounds' % n)
print('R2 = %.4f' % r2)
print('MAE = %.4f eV/atom' % mae)
print('RMSE = %.4f eV/atom' % rmse)
print('DFT range: [%.3f, %.3f]' % (x_dft.min(), x_dft.max()))
print('Pred range: [%.3f, %.3f]' % (y_pred.min(), y_pred.max()))

# === 绘图 ===
fig, ax = plt.subplots(figsize=(4.5, 4.0))

# 数据范围
pad = 0.15
lim = [min(x_dft.min(), y_pred.min()) - pad,
       max(x_dft.max(), y_pred.max()) + pad]

# 散点: 颜色 = ensemble sigma (viridis, 色盲友好)
sc = ax.scatter(x_dft, y_pred, c=sigma, cmap='viridis', s=12,
                alpha=0.5, edgecolors='none', zorder=2)

# y = x 参考线 (完美预测)
ax.plot(lim, lim, color='#D55E00', ls='--', lw=1.2, alpha=0.8,
        label='$y = x$ (perfect prediction)', zorder=3)

# 线性拟合线 (显示拟合优度)
coeffs = np.polyfit(x_dft, y_pred, 1)
x_fit = np.array(lim)
y_fit = np.polyval(coeffs, x_fit)
ax.plot(x_fit, y_fit, color='#0072B2', ls='-', lw=1.0, alpha=0.6,
        label='Linear fit ($R^2 = %.3f$)' % r2, zorder=3)

# 坐标轴
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.set_xlabel('DFT formation energy (eV/atom)', fontsize=10)
ax.set_ylabel('Predicted formation energy (eV/atom)', fontsize=10)

# R² / MAE / N 标注 (左上角)
text_x = 0.05
text_y = 0.95
ax.text(text_x, text_y, '$R^2$ = %.3f\nMAE = %.3f eV/atom\nRMSE = %.3f eV/atom\n$N$ = %d' % (r2, mae, rmse, n),
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray', lw=0.5))

# 色标
cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Ensemble $\\sigma$ (eV/atom)', fontsize=8)
cbar.ax.tick_params(labelsize=7)

# 图例
ax.legend(loc='lower right', framealpha=0.9, fontsize=7)

# 网格 (淡)
ax.grid(True, alpha=0.15, ls='-', lw=0.3)

fig.tight_layout()

# 保存 PDF (矢量) + PNG (预览)
out_dir = Path('paper/figures')
fig.savefig(out_dir / 'Figure_2_parity_publication.pdf', format='pdf')
fig.savefig(out_dir / 'Figure_2_parity_publication.png', format='png')
plt.close(fig)

print('\n[SAVE] paper/figures/Figure_2_parity_publication.pdf + .png')
print('Figure size: 4.5x4.0 inch (single-column CMS)')
print('Resolution: 500 dpi (PNG), vector (PDF)')
