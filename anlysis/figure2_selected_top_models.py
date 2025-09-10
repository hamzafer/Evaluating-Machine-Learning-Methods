import os
import pandas as pd
import matplotlib.pyplot as plt


BASE_RESULTS_DIR = os.path.join('main', 'cmy2xyz', 'results')
# Order datasets PC10 first for consistency
DATASETS = ['PC10', 'PC11', 'FOGRA']
OUT_DIR = 'figures'


# Selected top models; we will aggregate from append files using run-mean
TOP_MODELS = [
    'Gradient Boosting',
    'Bayesian (Gaussian Process)',
    'k-NN',
]

# Map display names to append-file algorithm keys
DISPLAY_TO_APPEND_KEY = {
    'Gradient Boosting': 'GradientBoost',
    'Bayesian (Gaussian Process)': 'Bayesian',
    'k-NN': 'k_Nearest',
}


def run_mean_from_append(path: str, algo_key: str) -> float:
    """Average of 'Mean Error' across runs for the given algorithm in append file."""
    if not os.path.exists(path):
        return float('nan')
    df = pd.read_csv(path)
    sub = df[df['Algorithm'] == algo_key]
    if sub.empty:
        return float('nan')
    return float(sub['Mean Error'].mean())


def mean_for_poly_degree(path: str, degree: int) -> float:
    df = pd.read_csv(path)
    deg_val = float(degree)
    row = df[df['Configuration'] == deg_val]
    if row.empty:
        return float('nan')
    return float(row.iloc[0]['Mean Error'])


def build_summary(poly_degree: int = 3) -> pd.DataFrame:
    rows = []

    # Three selected ML models: run-mean across append results per dataset
    for display_name in TOP_MODELS:
        row = {'Algorithm': display_name}
        for ds in DATASETS:
            append_path = os.path.join(BASE_RESULTS_DIR, ds, f"{ds}_append_results.csv")
            algo_key = DISPLAY_TO_APPEND_KEY.get(display_name, display_name)
            row[ds] = run_mean_from_append(append_path, algo_key)
        rows.append(row)

    # Polynomial Regression (3rd order)
    poly_label = 'Polynomial Regression (3rd)'
    row = {'Algorithm': poly_label}
    for ds in DATASETS:
        csv_path = os.path.join(BASE_RESULTS_DIR, ds, 'polynomial_regression_results.csv')
        if not os.path.exists(csv_path):
            row[ds] = float('nan')
        else:
            row[ds] = mean_for_poly_degree(csv_path, poly_degree)
    rows.append(row)

    return pd.DataFrame(rows)


def save_grouped_bar(df: pd.DataFrame, out_path: str, *, ylim_min: float | None = None, ylim_max: float | None = None, show_labels: bool = False):
    labels = df['Algorithm'].tolist()
    x = list(range(len(labels)))
    width = 0.22

    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {
        'PC10': '#ff7f0e',   # orange
        'PC11': '#1f77b4',   # blue
        'FOGRA': '#2ca02c',  # green
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    # Bars: PC10 left, PC11 center, FOGRA right
    bars_pc10 = ax.bar([i - width for i in x], df['PC10'], width=width, label='PC10', color=colors['PC10'], edgecolor='black', linewidth=0.3)
    bars_pc11 = ax.bar(x, df['PC11'], width=width, label='PC11', color=colors['PC11'], edgecolor='black', linewidth=0.3)
    bars_fogra = ax.bar([i + width for i in x], df['FOGRA'], width=width, label='FOGRA51', color=colors['FOGRA'], edgecolor='black', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Mean ΔE')
    ax.set_title('Mean ΔE Across PC10, PC11, and FOGRA51 (selected top models vs Polynomial Regression 3rd)')

    ymax = float(pd.concat([df['PC10'], df['PC11'], df['FOGRA']]).max())
    if ylim_min is not None and ylim_max is not None:
        ax.set_ylim(ylim_min, ylim_max)
    else:
        ax.set_ylim(0, ymax * 1.18)
    ax.margins(x=0.06)
    ax.legend(ncols=1, frameon=True, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=10, fancybox=True, framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.grid(axis='x', visible=False)
    ax.margins(x=0.01)
    if show_labels:
        def _label(bars):
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, h, f"{h:.3f}", ha='center', va='bottom', fontsize=8)
        _label(bars_pc10)
        _label(bars_pc11)
        _label(bars_fogra)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    import argparse
    parser = argparse.ArgumentParser(description='Generate Figure 2: selected top models vs polynomial regression (3rd).')
    parser.add_argument('--ylim-min', type=float, default=1.2, help='Y-axis lower bound (default 1.2).')
    parser.add_argument('--ylim-max', type=float, default=1.5, help='Y-axis upper bound (default 1.5).')
    parser.add_argument('--no-labels', action='store_true', help='Disable bar value labels.')
    parser.add_argument('--poly-degree', type=int, default=3, help='Polynomial degree to compare (default 3).')
    args = parser.parse_args()

    df = build_summary(poly_degree=args.poly_degree)

    # Save CSV
    csv_path = os.path.join(OUT_DIR, 'Figure2_Selected_Top_Models_Mean_DeltaE.csv')
    df.to_csv(csv_path, index=False)

    # Save PNG
    png_path = os.path.join(OUT_DIR, 'Figure2_Selected_Top_Models.png')
    save_grouped_bar(df, png_path, ylim_min=args.ylim_min, ylim_max=args.ylim_max, show_labels=not args.no_labels)
    print(f"Wrote: {csv_path}\nWrote: {png_path}")


if __name__ == '__main__':
    main()
