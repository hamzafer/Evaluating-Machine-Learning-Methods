import os
import pandas as pd
import matplotlib.pyplot as plt


PC10_DIR = os.path.join('main', 'cmy2xyz', 'results', 'PC10')
APPEND_CSV = os.path.join(PC10_DIR, 'PC10_append_results.csv')
POLY_CSV = os.path.join(PC10_DIR, 'polynomial_regression_results.csv')
OUT_DIR = os.path.join('figures')


# Mapping from repo algorithm keys to display names used in the paper table
DISPLAY_NAMES = {
    'Bayesian': 'Bayesian (Gaussian Process)',
    'DecisionTree': 'Decision Tree',
    'DeepLearning': 'Deep Learning (Neural Network)',
    'Elastic': 'Elastic Net',
    'GradientBoost': 'Gradient Boosting',
    'Lasso': 'Lasso Regression',
    'PCR': 'PCR',
    'PLSRegression': 'PLSR',
    'RandomForestRegressor': 'Random Forest',
    'RidgeRegression': 'Ridge Regression',
    'SVM': 'SVM (RBF Kernel)',
    'SimpleMLP': 'MLP (Shallow Network)',
    'k_Nearest': 'k-NN',
}


def load_pc10_summary():
    df = pd.read_csv(APPEND_CSV)
    # New behavior: for each algorithm, select the single configuration/run
    # that achieved the minimum 'Mean Error', then report that row's
    # Mean/Median/Max values. This matches Figure 1's best-config logic.

    ordered_keys = [
        'Bayesian', 'DecisionTree', 'DeepLearning', 'Elastic', 'GradientBoost',
        'Lasso', 'PCR', 'PLSRegression', 'RandomForestRegressor', 'RidgeRegression',
        'SVM', 'SimpleMLP', 'k_Nearest'
    ]

    best_rows = []
    for alg in ordered_keys:
        sub = df[df['Algorithm'] == alg]
        if sub.empty:
            continue
        best_idx = sub['Mean Error'].idxmin()
        best = sub.loc[best_idx]
        best_rows.append({
            'Algorithm': alg,
            'Display': DISPLAY_NAMES.get(alg, alg),
            'Mean': float(best['Mean Error']),
            'Median': float(best['Median Error']),
            'Max': float(best['Max Error']),
            'order': ordered_keys.index(alg),
        })

    grouped = pd.DataFrame(best_rows)

    # Add Polynomial Regression (3rd) from the polynomial results
    poly = pd.read_csv(POLY_CSV)
    degree3 = poly[poly['Configuration'] == 3.0].iloc[0]
    poly_row = pd.DataFrame([
        {
            'Algorithm': 'PolynomialRegression(3rd)',
            'Display': 'Polynomial Regression (3rd)',
            'Mean': degree3['Mean Error'],
            'Median': degree3['Median Error'],
            'Max': degree3['Max Error'],
            'order': len(ordered_keys),
        }
    ])

    summary = pd.concat([
        grouped[['Display', 'Mean', 'Median', 'Max', 'order']],
        poly_row[['Display', 'Mean', 'Median', 'Max', 'order']]
    ], ignore_index=True).sort_values('order').drop(columns=['order'])

    # Round for presentation
    summary = summary.round({'Mean': 6, 'Median': 6, 'Max': 6})
    return summary


def save_table_png(df: pd.DataFrame, out_path: str, title: str | None = None):
    # Create a table image using matplotlib
    n_rows = len(df)
    fig_height = 1.0 + 0.4 * n_rows
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=12, pad=12)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_grouped_bar_png(df: pd.DataFrame, out_path: str, title: str):
    labels = df['Display'].tolist()
    x = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar([i - width for i in x], df['Mean'], width=width, label='Mean ΔE00')
    ax.bar(x, df['Median'], width=width, label='Median ΔE00')
    ax.bar([i + width for i in x], df['Max'], width=width, label='Max ΔE00')

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('ΔE00 error')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = load_pc10_summary()

    # Save CSV and figures
    csv_out = os.path.join(OUT_DIR, 'PC10_Table1_summary.csv')
    summary.to_csv(csv_out, index=False)

    table_png = os.path.join(OUT_DIR, 'Table1_PC10_Summary.png')
    save_table_png(summary, table_png, title='Table 1: PC10 Summary of ΔE00 Metrics by Model')

    bars_png = os.path.join(OUT_DIR, 'PC10_DeltaE_Summary_Bars.png')
    save_grouped_bar_png(summary, bars_png, title='PC10: Mean/Median/Max ΔE00 by Model')

    print(f"Wrote: {csv_out}\nWrote: {table_png}\nWrote: {bars_png}")


if __name__ == '__main__':
    main()
