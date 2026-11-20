"""An overview and image index for the rolling conditional-curve validation plots."""
import json
from pathlib import Path

from matplotlib import font_manager, pyplot as plt, rcParams
import numpy as np


def use_plot_font():
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    choices = ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'Microsoft YaHei', 'PingFang SC', 'SimHei')
    rcParams['font.family'] = [name for name in choices if name in fonts] + ['DejaVu Sans']


def write_gallery(report, out, focus_ranks):
    out = Path(out)
    use_plot_font()
    events = sorted(report['events'], key=lambda e: (e['rule'] != 'new', -e['event']))
    values = np.full((len(events), len(focus_ranks)+1), np.nan)
    for i, event in enumerate(events):
        scores = {r['rank']: r['mape'] for r in event['ranks']}
        values[i] = [scores.get(rank) for rank in focus_ranks] + [event['focus_mape']]
    fig, ax = plt.subplots(figsize=(8.1, max(3.5, .30*len(events)+1.8)), layout='constrained')
    cmap = plt.get_cmap('YlOrRd').copy()
    cmap.set_bad('#e8ecee')
    high = max(1., float(values[np.isfinite(values)].max(initial=0.)))
    heatmap = ax.imshow(values, aspect='auto', vmin=0, vmax=high, cmap=cmap)
    ax.set_xticks(range(len(focus_ranks)+1), [f'{r} 档' for r in focus_ranks]+['活动均值'])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(events)), [f"{e['event']}  ·  {'新规则' if e['rule']=='new' else '旧规则'}"
                                      for e in events])
    for (i,j), value in np.ndenumerate(values):
        label = f'{value:.2f}' if np.isfinite(value) else '—'
        ax.text(j, i, label, ha='center', va='center', fontsize=9,
                color='white' if value > high*.55 else '#283138')
    new_count = sum(e['rule'] == 'new' for e in events)
    if 0 < new_count < len(events):
        ax.axhline(new_count-.5, color='white', lw=2.5)
    ax.axvline(len(focus_ranks)-.5, color='white', lw=2)
    ax.set_title('条件曲线 · 验证集滚动结算误差\n每半小时起报，数值为平均绝对百分比误差（%）',
                 pad=16, fontsize=13)
    fig.colorbar(heatmap, ax=ax, shrink=.55, label='误差（%）')
    fig.savefig(out / 'overview.png', dpi=150)
    plt.close(fig)
    template = Path(__file__).with_name('validation_gallery.html').read_text()
    data = json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    tmp = out / 'index.tmp'
    tmp.write_text(template.replace('__VALIDATION_REPORT__', data))
    tmp.replace(out / 'index.html')
