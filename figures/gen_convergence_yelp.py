"""
Generate Yelp convergence figure for the paper.
Run from ~/FedLLM-Re/rework/ on the GPU server.
Output: figures/yelp_convergence_v2.pdf
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = os.path.expanduser('~/FedLLM-Re/rework')

SCAN_DIRS = [
    os.path.join(ROOT, 'results_prof',        'yelp'),
    os.path.join(ROOT, 'results_v2',          'yelp'),
    os.path.join(ROOT, 'results_hetloram_b05','yelp'),
]

METHODS = ['homo_r8', 'hetero_pad', 'flexlora', 'hetlora', 'hetlora_m']
LABELS  = {
    'homo_r8':    'Homo r=8',
    'hetero_pad': 'Hetero-Pad',
    'flexlora':   'FlexLoRA',
    'hetlora':    'HetLoRA',
    'hetlora_m':  'FedMoLoRA (ours)',
}
COLORS  = {
    'homo_r8':    '#888888',
    'hetero_pad': '#4e79a7',
    'flexlora':   '#f28e2b',
    'hetlora':    '#e15759',
    'hetlora_m':  '#9467bd',
}
LINESTYLES = {
    'homo_r8':    '--',
    'hetero_pad': '-.',
    'flexlora':   ':',
    'hetlora':    '-',
    'hetlora_m':  '-',
}
LINEWIDTHS = {
    'homo_r8':    1.4,
    'hetero_pad': 1.4,
    'flexlora':   1.4,
    'hetlora':    1.6,
    'hetlora_m':  2.4,
}
EXCLUDE = {'spa_m', 'hetero_spa', 'homo_r4'}

# ── load ──────────────────────────────────────────────────────────────────────
rows, seen = [], set()
for d in SCAN_DIRS:
    for fp in sorted(glob.glob(os.path.join(d, '*.json'))):
        try:
            data = json.load(open(fp))
        except Exception:
            continue
        method = data.get('method', '')
        if method in EXCLUDE:
            continue
        seed  = data.get('seed', -1)
        alpha = data.get('alpha', -1)
        key   = (method, alpha, seed)
        if key in seen:
            continue
        seen.add(key)
        for r in data.get('rounds', []):
            val = r.get('accuracy')
            if val is None:
                continue
            rows.append({'method': method, 'alpha': float(alpha),
                         'seed': int(seed), 'round': int(r['round']),
                         'acc': float(val)})

import pandas as pd
df = pd.DataFrame(rows)
print(f'Loaded {len(seen)} runs, {len(df)} round-rows')
print(f'Methods: {sorted(df["method"].unique())}')
print(f'Alphas:  {sorted(df["alpha"].unique())}')

# ── plot ──────────────────────────────────────────────────────────────────────
ALPHAS  = [0.5, 0.1, 0.01]
TITLES  = [r'$\alpha=0.5$', r'$\alpha=0.1$', r'$\alpha=0.01$']

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
})

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=False)

for ax, alpha, title in zip(axes, ALPHAS, TITLES):
    sub = df[df['alpha'] == alpha]
    for m in METHODS:
        g = sub[sub['method'] == m]
        if g.empty:
            continue
        pivot = g.pivot_table(index='round', columns='seed', values='acc')
        rounds = pivot.index.values
        mu  = pivot.mean(axis=1).values * 100
        std = pivot.std(axis=1).fillna(0).values * 100
        ax.plot(rounds, mu,
                label=LABELS[m], color=COLORS[m],
                ls=LINESTYLES[m], lw=LINEWIDTHS[m])
        ax.fill_between(rounds, mu - std, mu + std,
                        alpha=0.10, color=COLORS[m])
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Round', fontsize=8)
    if ax is axes[0]:
        ax.set_ylabel('Accuracy (%)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(left=1)

# single legend under the three panels
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center', ncol=5,
           fontsize=7.5, frameon=False,
           bbox_to_anchor=(0.5, -0.18))

plt.suptitle('Yelp Sentiment — Per-Round Accuracy', fontsize=10, y=1.02)
plt.tight_layout()

out_dir = os.path.join(ROOT, 'figures')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'yelp_convergence_v2.pdf')
plt.savefig(out, bbox_inches='tight', format='pdf')
print(f'Saved → {out}')
