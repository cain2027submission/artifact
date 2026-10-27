#!/usr/bin/env python3
# Interpreter: python3 (requires numpy and matplotlib in the environment)
"""Generate three publication-quality PDF figures for IEEE two-column paper."""

import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Configuration
FIGURE_WIDTH = 3.4  # inches, IEEE single column
FONT_SIZE = 8
DPI = 300

# Colourblind-safe palette (Okabe-Ito)
COLORS = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'bluish_green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'reddish_purple': '#CC79A7',
    'black': '#000000',
    'gray': '#999999'
}

# Set global matplotlib parameters
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
    'figure.dpi': DPI,
})

def load_compression_data():
    """Load RQ2 compression data from CSV."""
    data_file = REPO_ROOT / 'data/paper-rollups/rq2_compression_corrected.csv'
    sizes = []
    ratios = []

    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sizes.append(float(row['serialized_bytes']))
            ratios.append(float(row['comp_tokens']))

    return np.array(sizes), np.array(ratios)

def load_composition_data():
    """Load RQ1 composition data from JSON."""
    data_file = REPO_ROOT / 'data/paper-rollups/rq1_corrected_summary.json'
    with open(data_file, 'r') as f:
        data = json.load(f)

    parent = data['shares_substantive']['_pooled']
    subagent = data['shares_subagent_transcripts']['_pooled']

    return parent, subagent

def load_thinking_data():
    """Load thinking persistence data from JSON."""
    data_file = REPO_ROOT / 'data/paper-rollups/rq1_corrected_summary.json'
    with open(data_file, 'r') as f:
        data = json.load(f)

    return data['thinking_persistence']

def make_figure1(sizes, ratios):
    """Figure 1: Compression vs size scatter plot."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_WIDTH * 0.75))

    # Count points below y=1.0
    below_one = ratios < 1.0
    count_below = np.sum(below_one)
    pct_below = 100 * count_below / len(ratios)

    # Scatter plot
    ax.scatter(sizes[~below_one], ratios[~below_one],
              alpha=0.5, s=10, color=COLORS['blue'], label='Compressed')
    ax.scatter(sizes[below_one], ratios[below_one],
              alpha=0.5, s=10, color=COLORS['vermillion'],
              marker='s', label='Rollup larger')

    # Horizontal reference line at y=1.0
    ax.axhline(y=1.0, color=COLORS['black'], linestyle='--', linewidth=1)
    ax.text(sizes.max() * 0.95, 1.05, 'rollup larger than session',
           ha='right', va='bottom', fontsize=FONT_SIZE-1)

    # Shade region below y=1.0 with hatching (not color alone)
    ax.axhspan(ratios.min() * 0.9, 1.0, alpha=0.1, color=COLORS['vermillion'],
              hatch='///', edgecolor=COLORS['vermillion'], linewidth=0)

    # Vertical guides at size boundaries
    for size, label in [(10e3, '10KB'), (100e3, '100KB'), (1e6, '1MB')]:
        ax.axvline(x=size, color=COLORS['gray'], linestyle=':', linewidth=0.8, alpha=0.5)
        ax.text(size, ratios.max() * 0.95, label, ha='center', va='top',
               fontsize=FONT_SIZE-1, rotation=0)

    # Annotation for points below y=1.0
    ax.text(0.02, 0.98, f'n={count_below} ({pct_below:.1f}%)\nbelow 1.0',
           transform=ax.transAxes, va='top', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=COLORS['gray']))

    # Log scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Labels
    ax.set_xlabel('Transcript size (bytes)')
    ax.set_ylabel('Token compression ratio')
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(REPO_ROOT / 'paper/fig/compression_vs_size.pdf', bbox_inches='tight')
    plt.close()

    return count_below, pct_below

def make_figure2(parent, subagent):
    """Figure 2: Composition stacked bar chart."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_WIDTH * 0.5))

    # Category order and labels
    categories = ['tool_result', 'tool_use', 'user_text', 'assistant_text', 'thinking', 'system_injected']
    colors_map = {
        'tool_result': COLORS['blue'],
        'tool_use': COLORS['sky_blue'],
        'user_text': COLORS['orange'],
        'assistant_text': COLORS['bluish_green'],
        'thinking': COLORS['vermillion'],  # Distinct color for visibility
        'system_injected': COLORS['yellow']
    }

    # Convert fractions to percentages
    parent_pct = {k: parent[k] * 100 for k in categories}
    subagent_pct = {k: subagent[k] * 100 for k in categories}

    # Bar labels
    bar_labels = ['Parent sessions (n=236)', 'Subagent files (n=1075)']
    y_pos = np.arange(len(bar_labels))

    # Stacked bars
    left_parent = 0
    left_subagent = 0

    for cat in categories:
        # Parent bar
        ax.barh(y_pos[0], parent_pct[cat], left=left_parent,
               color=colors_map[cat], label=cat.replace('_', ' '))
        # Label if >5%
        if parent_pct[cat] > 5:
            ax.text(left_parent + parent_pct[cat]/2, y_pos[0],
                   f"{parent_pct[cat]:.1f}%",
                   ha='center', va='center', fontsize=FONT_SIZE-1, color='white', weight='bold')
        left_parent += parent_pct[cat]

        # Subagent bar
        ax.barh(y_pos[1], subagent_pct[cat], left=left_subagent,
               color=colors_map[cat])
        # Label if >5%
        if subagent_pct[cat] > 5:
            ax.text(left_subagent + subagent_pct[cat]/2, y_pos[1],
                   f"{subagent_pct[cat]:.1f}%",
                   ha='center', va='center', fontsize=FONT_SIZE-1, color='white', weight='bold')
        left_subagent += subagent_pct[cat]

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels)
    ax.set_xlabel('Percentage (%)')
    ax.set_xlim(0, 100)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

    plt.tight_layout()
    plt.savefig(REPO_ROOT / 'paper/fig/composition.pdf', bbox_inches='tight')
    plt.close()

def make_figure3(thinking_data):
    """Figure 3: Thinking persistence grouped bar chart."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_WIDTH * 0.6))

    # Data extraction
    parent_total = thinking_data['parent']['blocks']
    parent_retained = parent_total - thinking_data['parent']['empty']
    parent_empty_pct = thinking_data['parent']['empty_pct']

    subagent_total = thinking_data['subagent']['blocks']
    subagent_retained = subagent_total - thinking_data['subagent']['empty']
    subagent_empty_pct = thinking_data['subagent']['empty_pct']

    # Bar positions
    x = np.arange(2)
    width = 0.35

    # Grouped bars
    bars1 = ax.bar(x - width/2, [parent_total, subagent_total], width,
                   label='Total blocks', color=COLORS['blue'])
    bars2 = ax.bar(x + width/2, [parent_retained, subagent_retained], width,
                   label='Blocks retaining text', color=COLORS['bluish_green'])

    # Annotate empty percentages above each group
    ax.text(x[0], parent_total * 1.05, f'{parent_empty_pct:.1f}% empty',
           ha='center', va='bottom', fontsize=FONT_SIZE)
    ax.text(x[1], subagent_total * 1.05, f'{subagent_empty_pct:.1f}% empty',
           ha='center', va='bottom', fontsize=FONT_SIZE)

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(['Parent', 'Subagent'])
    ax.set_ylabel('Number of thinking blocks')
    ax.legend(loc='upper right', frameon=True)
    ax.set_ylim(0, max(parent_total, subagent_total) * 1.15)

    plt.tight_layout()
    plt.savefig(REPO_ROOT / 'paper/fig/thinking_persistence.pdf', bbox_inches='tight')
    plt.close()

def main():
    """Generate all three figures."""
    # Ensure output directory exists
    (REPO_ROOT / 'paper/fig').mkdir(parents=True, exist_ok=True)

    # Load data and generate figures
    print("Loading compression data...")
    sizes, ratios = load_compression_data()
    print(f"Generating Figure 1 ({len(sizes)} sessions)...")
    count_below, pct_below = make_figure1(sizes, ratios)

    print("Loading composition data...")
    parent, subagent = load_composition_data()
    print("Generating Figure 2...")
    make_figure2(parent, subagent)

    print("Loading thinking persistence data...")
    thinking_data = load_thinking_data()
    print("Generating Figure 3...")
    make_figure3(thinking_data)

    print("\nFigures generated successfully:")
    print(f"  - Figure 1: {count_below} sessions ({pct_below:.1f}%) below compression 1.0")

    # Verify outputs
    for fig_name in ['compression_vs_size.pdf', 'composition.pdf', 'thinking_persistence.pdf']:
        fig_path = REPO_ROOT / 'paper/fig' / fig_name
        if fig_path.exists():
            print(f"  - {fig_name}: {fig_path.stat().st_size} bytes")
        else:
            print(f"  - ERROR: {fig_name} not created!")
            return 1

    return 0

if __name__ == '__main__':
    exit(main())
