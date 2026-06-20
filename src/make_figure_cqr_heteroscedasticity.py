"""
Figure 2: CQR heteroscedastic interval illustration
真实数据驱动, 展示区间宽度随样本不确定性自适应变化
横轴: 样本按 sigma 排序的索引 (低不确定 -> 高不确定)
纵轴: 区间宽度 / 绝对误差
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def smooth(x, w=100):
    """滑动平均平滑 (w=窗口大小)"""
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='valid')


fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={'wspace': 0.28})

for ax_idx, (target, title_short, ylim) in enumerate([
    ('formation_energy_per_atom', 'Formation energy', 2.2),
    ('energy_above_hull', 'Energy above hull', 2.8),
]):
    ax = axes[ax_idx]
    df = pd.read_csv(f'results/metrics/pact_final_oof_{target}.csv')

    # 按 sigma 排序 (低不确定 -> 高不确定)
    df = df.sort_values('oof_sigma').reset_index(drop=True)

    cqr_width = (df['cqr_upper'] - df['cqr_lower']).values
    std_width = (df['std_upper'] - df['std_lower']).values
    abs_err = df['abs_err'].values
    sigma = df['oof_sigma'].values

    n = len(df)
    x_idx = np.arange(n)
    w = 150  # 平滑窗口

    # 平滑后的曲线
    cqr_smooth = smooth(cqr_width, w)
    std_smooth = smooth(std_width, w)
    err_smooth = smooth(abs_err, w)
    x_smooth = np.arange(len(cqr_smooth))

    # 画 CQR 区间宽度 (橙色填充)
    ax.fill_between(x_smooth, 0, cqr_smooth, color='#E69F00', alpha=0.35,
                    label='CQR interval width')
    ax.plot(x_smooth, cqr_smooth, color='#D55E00', linewidth=2.0, zorder=3)

    # 画 standard conformal 区间宽度 (灰色虚线, 等宽)
    ax.plot(x_smooth, std_smooth, color='#999999', linewidth=1.8,
            linestyle='--', label='Standard conformal (uniform)')

    # 画绝对误差 (蓝色实线)
    ax.plot(x_smooth, err_smooth, color='#0072B2', linewidth=1.8,
            label='Absolute error', zorder=4)

    # 分区背景: 左 1/3 绿 (interpolation), 右 1/3 红 (extrapolation)
    ax.axvspan(0, n/3, color='#2ca02c', alpha=0.06)
    ax.axvspan(2*n/3, n, color='#d62728', alpha=0.06)

    # 分区标注
    ax.text(n/6, ylim*0.93, 'interpolation\n(low σ)', ha='center',
            fontsize=8, style='italic', color='#2ca02c')
    ax.text(5*n/6, ylim*0.93, 'extrapolation\n(high σ)', ha='center',
            fontsize=8, style='italic', color='#d62728')

    # 关键数值标注
    q20 = cqr_width[:int(n*0.2)].mean()
    q80 = cqr_width[int(n*0.8):].mean()
    err_q20 = abs_err[:int(n*0.2)].mean()
    err_q80 = abs_err[int(n*0.8):].mean()
    ax.annotate(f'width {q20:.2f}', xy=(int(n*0.1), q20),
                xytext=(int(n*0.1), q20+0.25),
                fontsize=7.5, color='#D55E00', ha='center',
                arrowprops=dict(arrowstyle='->', color='#D55E00', lw=0.8))
    ax.annotate(f'width {q80:.2f}', xy=(int(n*0.9), q80),
                xytext=(int(n*0.9), q80+0.25),
                fontsize=7.5, color='#D55E00', ha='center',
                arrowprops=dict(arrowstyle='->', color='#D55E00', lw=0.8))

    ax.set_xlim(0, n)
    ax.set_ylim(0, ylim)
    ax.set_xlabel('Samples sorted by ensemble σ  (low → high uncertainty)',
                  fontsize=9.5)
    ax.set_ylabel('Interval width / |error|  (eV/atom)', fontsize=9.5)
    ax.set_title(f'({chr(97+ax_idx)}) {title_short}', fontsize=10, pad=8)

    ax.tick_params(labelsize=8.5)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    # 只在右图显示图例
    if ax_idx == 1:
        legend_elements = [
            Patch(facecolor='#E69F00', alpha=0.35, edgecolor='#D55E00',
                  linewidth=1.5, label='CQR interval width'),
            Line2D([0], [0], color='#999999', linewidth=1.8, linestyle='--',
                   label='Standard conformal (uniform)'),
            Line2D([0], [0], color='#0072B2', linewidth=1.8,
                   label='Absolute error'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
                  framealpha=0.9)

plt.suptitle('Sample-adaptive (heteroscedastic) CQR intervals versus '
             'uniform standard-conformal intervals',
             fontsize=11, y=1.02)

plt.tight_layout()
plt.savefig('paper/figures/Figure_2_cqr_heteroscedasticity.png',
            dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/Figure_2_cqr_heteroscedasticity.pdf',
            bbox_inches='tight')
print('[OK] saved: paper/figures/Figure_2_cqr_heteroscedasticity.png/pdf')

# 打印关键统计, 确认图的真实性
print()
print('=== 图的真实数据依据 ===')
for target, name in [('formation_energy_per_atom', 'formation'),
                      ('energy_above_hull', 'hull')]:
    df = pd.read_csv(f'results/metrics/pact_final_oof_{target}.csv')
    df = df.sort_values('oof_sigma')
    w = df['cqr_upper'] - df['cqr_lower']
    n = len(df)
    q20 = w.iloc[:int(n*0.2)].mean()
    q80 = w.iloc[int(n*0.8):].mean()
    e20 = df['abs_err'].iloc[:int(n*0.2)].mean()
    e80 = df['abs_err'].iloc[int(n*0.8):].mean()
    print(f'{name}: 窄区间组 width={q20:.3f}/err={e20:.3f}, '
          f'宽区间组 width={q80:.3f}/err={e80:.3f}, '
          f'宽度比={q80/q20:.1f}x, 误差比={e80/e20:.1f}x')
