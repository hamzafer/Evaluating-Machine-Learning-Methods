import os
import math
import pandas as pd
import matplotlib.pyplot as plt


BASE_RESULTS_DIR = os.path.join('main', 'cmy2xyz', 'results')
# Order for display and CSV: PC10, then PC11, then FOGRA
DATASETS = ['PC10', 'PC11', 'FOGRA']
OUT_DIR = os.path.join('figures', 'figure1')


# File names per algorithm and display names used in the plot
ALGO_FILES = {
    'Bayesian (Gaussian Process)': 'nn_Bayesian_results.csv',
    'Decision Tree': 'nn_DecisionTree_results.csv',
    'Deep Learning (Neural Network)': 'nn_DeepLearning_results.csv',
    'Elastic Net': 'nn_Elastic_results.csv',
    'Gradient Boosting': 'nn_GradientBoost_results.csv',
    'k-NN': 'nn_k_Nearest_results.csv',
    'Lasso Regression': 'nn_Lasso_results.csv',
    'PCR': 'nn_PCR_results.csv',
    'PLSR': 'nn_PLSRegression_results.csv',
    'Random Forest': 'nn_RandomForestRegressor_results.csv',
    'Ridge Regression': 'nn_RidgeRegression_results.csv',
    'MLP (Shallow Network)': 'nn_SimpleMLP_results.csv',
    'SVM (RBF Kernel)': 'nn_SVM_results.csv',
}

# Map display names to the algorithm keys used in the *_append_results.csv files
DISPLAY_TO_APPEND_KEY = {
    'Bayesian (Gaussian Process)': 'Bayesian',
    'Decision Tree': 'DecisionTree',
    'Deep Learning (Neural Network)': 'DeepLearning',
    'Elastic Net': 'Elastic',
    'Gradient Boosting': 'GradientBoost',
    'k-NN': 'k_Nearest',
    'Lasso Regression': 'Lasso',
    'PCR': 'PCR',
    'PLSR': 'PLSRegression',
    'Random Forest': 'RandomForestRegressor',
    'Ridge Regression': 'RidgeRegression',
    'MLP (Shallow Network)': 'SimpleMLP',
    'SVM (RBF Kernel)': 'SVM',
}


def best_mean_from_csv(path: str) -> float:
    df = pd.read_csv(path)
    # Take the minimum Mean Error as the best configuration's mean ΔE00
    return float(df['Mean Error'].min())


def ordinal(n: int) -> str:
    # 1 -> 1st, 2 -> 2nd, 3 -> 3rd, others -> th
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def mean_for_poly_degree(path: str, degree: int) -> float:
    df = pd.read_csv(path)
    # Configuration column holds degree as float (e.g., 3.0)
    deg_val = float(degree)
    row = df[df['Configuration'] == deg_val]
    if row.empty:
        return float('nan')
    return float(row.iloc[0]['Mean Error'])


def run_mean_from_append(path: str, algo_key: str) -> float:
    """Compute the mean of 'Mean Error' across runs for the given algorithm key
    from an *_append_results.csv file."""
    if not os.path.exists(path):
        return float('nan')
    df = pd.read_csv(path)
    sub = df[df['Algorithm'] == algo_key]
    if sub.empty:
        return float('nan')
    return float(sub['Mean Error'].mean())

def run_count_from_append(path: str, algo_key: str) -> int:
    """Number of runs/configs available for the given algorithm key in append file."""
    if not os.path.exists(path):
        return 0
    df = pd.read_csv(path)
    sub = df[df['Algorithm'] == algo_key]
    return int(len(sub)) if not sub.empty else 0

def run_std_from_append(path: str, algo_key: str) -> float:
    """Standard deviation of 'Mean Error' across runs for the algorithm key.
    If unavailable, returns NaN."""
    if not os.path.exists(path):
        return float('nan')
    df = pd.read_csv(path)
    sub = df[df['Algorithm'] == algo_key]
    if sub.empty:
        return float('nan')
    return float(sub['Mean Error'].std(ddof=1))


def build_summary(poly_degree: int, aggregate: str = 'best') -> pd.DataFrame:
    rows = []
    # First, ML models using their min Mean Error
    for display_name, filename in ALGO_FILES.items():
        row = {'Algorithm': display_name}
        for ds in DATASETS:
            if aggregate == 'run-mean':
                append_path = os.path.join(BASE_RESULTS_DIR, ds, f"{ds}_append_results.csv")
                algo_key = DISPLAY_TO_APPEND_KEY.get(display_name, display_name)
                row[ds] = run_mean_from_append(append_path, algo_key)
            else:
                csv_path = os.path.join(BASE_RESULTS_DIR, ds, filename)
                if not os.path.exists(csv_path):
                    row[ds] = float('nan')
                    continue
                row[ds] = best_mean_from_csv(csv_path)
        rows.append(row)

    # Then, Polynomial Regression for a fixed degree (e.g., 3rd)
    poly_label = f"Polynomial Regression ({ordinal(poly_degree)})"
    row = {'Algorithm': poly_label}
    for ds in DATASETS:
        csv_path = os.path.join(BASE_RESULTS_DIR, ds, 'polynomial_regression_results.csv')
        if not os.path.exists(csv_path):
            row[ds] = float('nan')
            continue
        row[ds] = mean_for_poly_degree(csv_path, poly_degree)
    rows.append(row)

    df = pd.DataFrame(rows)
    return df


def save_grouped_bar(df: pd.DataFrame, out_path: str, *, sort: bool = False, sort_by: str | None = None, error_bars: bool = False, aggregate: str = 'best', error_type: str = 'sd'):
    # Optional: rank algorithms by their overall average mean ΔE across datasets
    df = df.copy()
    title_suffix = ""
    if sort:
        if sort_by and sort_by in ['PC11', 'PC10', 'FOGRA']:
            df = df.sort_values(sort_by, ascending=True, na_position='last')
            title_suffix = f" (sorted by {sort_by})"
        else:
            df['Overall'] = df[['PC11', 'PC10', 'FOGRA']].mean(axis=1, skipna=True)
            df = df.sort_values('Overall', ascending=True).drop(columns=['Overall'])
            title_suffix = " (sorted by overall mean)"

    labels = df['Algorithm'].tolist()
    x = list(range(len(labels)))
    width = 0.22

    # Paper-friendly style and colorblind-safe palette
    plt.style.use('seaborn-v0_8-whitegrid')
    # Distinct, colorblind-friendly trio (Tableau 10 subset)
    colors = {
        'PC10': '#ff7f0e',   # orange
        'PC11': '#1f77b4',   # blue
        'FOGRA': '#2ca02c',  # green
    }

    fig, ax = plt.subplots(figsize=(16, 7))
    # Thicker error bars for visibility
    err_kw = dict(elinewidth=1.2, capthick=1.2, ecolor='black', alpha=0.95)
    # Optional error bars via run-std from append files when aggregate=='run-mean'
    yerr_pc10 = yerr_pc11 = yerr_fogra = None
    if error_bars and aggregate == 'run-mean':
        yerr_pc10, yerr_pc11, yerr_fogra = [], [], []
        for label in labels:
            key = DISPLAY_TO_APPEND_KEY.get(label, label)
            for ds, target in [('PC10', yerr_pc10), ('PC11', yerr_pc11), ('FOGRA', yerr_fogra)]:
                append_path = os.path.join(BASE_RESULTS_DIR, ds, f"{ds}_append_results.csv")
                std = run_std_from_append(append_path, key)
                n = run_count_from_append(append_path, key)
                if error_type == 'sem' and n > 0:
                    target.append(std / math.sqrt(n))
                elif error_type == 'ci95' and n > 0:
                    target.append(1.96 * std / math.sqrt(n))
                else:
                    target.append(std)

    # Bars ordered as PC10 (left), PC11 (center), FOGRA (right)
    ax.bar([i - width for i in x], df['PC10'], yerr=yerr_pc10, width=width, label='PC10', color=colors['PC10'], edgecolor='black', linewidth=0.3, capsize=4, error_kw=err_kw)
    ax.bar(x, df['PC11'], yerr=yerr_pc11, width=width, label='PC11', color=colors['PC11'], edgecolor='black', linewidth=0.3, capsize=4, error_kw=err_kw)
    ax.bar([i + width for i in x], df['FOGRA'], yerr=yerr_fogra, width=width, label='FOGRA51', color=colors['FOGRA'], edgecolor='black', linewidth=0.3, capsize=4, error_kw=err_kw)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Mean ΔE')
    ax.set_title('Mean ΔE Across PC10, PC11, and FOGRA51' + title_suffix)
    # Legend inside the plotting area (upper-left), with extra headroom
    ymax = float(pd.concat([df['PC10'], df['PC11'], df['FOGRA']]).max())
    ax.set_ylim(0, ymax * 1.18)
    ax.margins(x=0.06)
    ax.legend(ncols=1, frameon=True, loc='upper left', bbox_to_anchor=(0.01, 0.98),
              fontsize=10, fancybox=True, framealpha=0.95)
    # Subtle horizontal grid only
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.grid(axis='x', visible=False)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Figure 1 (Mean ΔE across datasets).')
    parser.add_argument('--sort', action='store_true', help='Sort algorithms (best to worst).')
    parser.add_argument('--sort-by', choices=['PC11', 'PC10', 'FOGRA', 'overall'], default='overall', help='Column to sort by when --sort is used.')
    parser.add_argument('--poly-degree', type=int, default=3, help='Polynomial regression degree to report (e.g., 3).')
    parser.add_argument('--aggregate', choices=['best', 'run-mean'], default='best', help="Aggregation: 'best' uses min Mean Error per model; 'run-mean' averages Mean Error across runs (append files).")
    parser.add_argument('--error-type', choices=['sd', 'sem', 'ci95'], default='sd', help='Error bar type used when --error-bars and --aggregate run-mean (sd, sem, or ci95).')
    parser.add_argument('--error-bars', action='store_true', help='Add error bars (std across runs in append files; only with --aggregate run-mean).')
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = build_summary(args.poly_degree, aggregate=args.aggregate)
    # Save CSV summary
    csv_path = os.path.join(OUT_DIR, 'Figure1_Mean_DeltaE_Summary.csv')
    df.to_csv(csv_path, index=False)

    # Plot PNG
    png_path = os.path.join(OUT_DIR, 'Figure1_Mean_DeltaE_Across_Datasets.png')
    sort_by = None if args.sort_by == 'overall' else args.sort_by
    save_grouped_bar(df, png_path, sort=args.sort, sort_by=sort_by, error_bars=args.error_bars, aggregate=args.aggregate, error_type=args.error_type)
    print(f"Wrote: {csv_path}\nWrote: {png_path}")


if __name__ == '__main__':
    main()
