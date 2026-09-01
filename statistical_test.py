import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "datasets"

FORGET_METRICS = ["answer_leakage", "deviation_quality", "response_coherence"]
RETAIN_METRICS = ["answer_preservation", "semantic_quality", "coherence_and_correctness"]
ALL_METRICS = FORGET_METRICS + RETAIN_METRICS

METRIC_LABELS = {
    "answer_leakage": "Leak (Forget)",
    "deviation_quality": "Dev (Forget)",
    "response_coherence": "Coh (Forget)",
    "answer_preservation": "Pres (Retain)",
    "semantic_quality": "Sem (Retain)",
    "coherence_and_correctness": "Coh (Retain)",
}

SELECTIONS = ["emb", "raslik", "nnomp"]
SELECTION_LABELS = ["Embedding", "RASLIK", "Ours"]
COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def parse_filename(fname):
    parts = Path(fname).stem.split("_")
    if parts[0] == "qwen":
        return "qwen", parts[1], parts[2], parts[3], parts[4]
    return "llama", parts[0], parts[1], parts[2], parts[3]


def load_all():
    files = [
        f for f in glob.glob(str(DATA_DIR / "*.parquet"))
        if "cp100" not in Path(f).name and "pre_unlearning" not in Path(f).name
    ]
    records = []
    for f in files:
        model, algo, dataset, selection, split = parse_filename(f)
        cols = FORGET_METRICS if split == "forget" else RETAIN_METRICS
        df = pd.read_parquet(f)[["id"] + cols]
        df["model"] = model
        df["algo"] = algo
        df["dataset"] = dataset
        df["selection"] = selection
        df["split"] = split
        records.append(df)
    return pd.concat(records, ignore_index=True)


def build_block_matrix(df, metric):
    split_val = "forget" if metric in FORGET_METRICS else "test"
    sub = df[df["split"] == split_val][["id", "model", "algo", "dataset", "selection", metric]]
    blocks = []
    for _, grp in sub.groupby(["model", "algo", "dataset"]):
        pivot = grp.pivot_table(index="id", columns="selection", values=metric, aggfunc="first")
        if not all(s in pivot.columns for s in SELECTIONS):
            continue
        pivot = pivot[SELECTIONS].dropna()
        blocks.append(pivot.reset_index(drop=True))
    return pd.concat(blocks, ignore_index=True) if blocks else None


def nemenyi_cd(n, k=3, alpha=0.05):
    q = {(3, 0.05): 2.343, (3, 0.01): 2.576}
    return q[(k, alpha)] * np.sqrt(k * (k + 1) / (6 * n))


def run_analysis(df):
    results = {}
    for metric in ALL_METRICS:
        matrix = build_block_matrix(df, metric)
        chi2, p = stats.friedmanchisquare(*[matrix[s].values for s in SELECTIONS])
        avg_ranks = matrix.rank(axis=1, method="average").mean().values
        posthoc = sp.posthoc_nemenyi_friedman(matrix.values)
        n = len(matrix)
        cd = nemenyi_cd(n)
        wtl_emb = (
            (matrix["nnomp"] > matrix["emb"]).sum(),
            (matrix["nnomp"] == matrix["emb"]).sum(),
            (matrix["nnomp"] < matrix["emb"]).sum(),
        )
        wtl_raslik = (
            (matrix["nnomp"] > matrix["raslik"]).sum(),
            (matrix["nnomp"] == matrix["raslik"]).sum(),
            (matrix["nnomp"] < matrix["raslik"]).sum(),
        )
        results[metric] = dict(
            n=n, chi2=chi2, p=p, avg_ranks=avg_ranks,
            posthoc=posthoc, cd=cd, wtl_emb=wtl_emb, wtl_raslik=wtl_raslik,
        )

    p_vals = [results[m]["p"] for m in ALL_METRICS]
    _, p_bh, _, _ = multipletests(p_vals, method="fdr_bh")
    for i, m in enumerate(ALL_METRICS):
        results[m]["p_bh"] = p_bh[i]
    return results


def print_results(results):
    sep = "=" * 115
    print(f"\n{sep}")
    print("FRIEDMAN TEST + NEMENYI POST-HOC  (k=3 treatments, Demšar 2006 framework)")
    print(f"{sep}")
    print(f"{'Metric':<25} {'n':>6} {'χ²':>8} {'p_raw':>12} {'p_BH':>12}  {'R_Emb':>7} {'R_RASLIK':>9} {'R_Ours':>8}")
    print("-" * 115)
    for m in ALL_METRICS:
        r = results[m]
        rk = r["avg_ranks"]
        sig = "***" if r["p_bh"] < 0.001 else "**" if r["p_bh"] < 0.01 else "*" if r["p_bh"] < 0.05 else "ns"
        print(f"{METRIC_LABELS[m]:<25} {r['n']:>6} {r['chi2']:>8.2f} {r['p']:>12.4e} {r['p_bh']:>12.4e} {sig:<4} {rk[0]:>7.3f} {rk[1]:>9.3f} {rk[2]:>8.3f}")

    print(f"\n{sep}")
    print("NEMENYI PAIRWISE P-VALUES  (diagonal = 1, matrix symmetric)")
    print(f"{sep}")
    for m in ALL_METRICS:
        r = results[m]
        ph = r["posthoc"]
        delta_emb = r["avg_ranks"][2] - r["avg_ranks"][0]
        delta_ras = r["avg_ranks"][2] - r["avg_ranks"][1]
        print(f"\n{METRIC_LABELS[m]}  (n={r['n']}, CD={r['cd']:.4f},  ΔRank Ours−Emb={delta_emb:+.3f},  Ours−RASLIK={delta_ras:+.3f})")
        for (a, b), label in [((0, 1), "Emb vs RASLIK"), ((0, 2), "Emb vs Ours"), ((1, 2), "RASLIK vs Ours")]:
            pv = ph.iloc[a, b]
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
            print(f"  {label:<20}: p = {pv:.4e}  {sig}")

    print(f"\n{sep}")
    print("WIN / TIE / LOSS  (Ours vs baseline, per question)")
    print(f"{sep}")
    for m in ALL_METRICS:
        r = results[m]
        w, t, l = r["wtl_emb"]
        w2, t2, l2 = r["wtl_raslik"]
        print(f"{METRIC_LABELS[m]:<25}  vs Emb:    {w:>5}W / {t:>5}T / {l:>5}L   vs RASLIK: {w2:>5}W / {t2:>5}T / {l2:>5}L")


def plot_cd_diagram(results, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, metric in enumerate(ALL_METRICS):
        ax = axes[i]
        r = results[metric]
        avg_ranks = r["avg_ranks"]
        cd = r["cd"]
        ph = r["posthoc"]

        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(-0.9, 2.2)
        ax.axis("off")

        ax.annotate("", xy=(3.3, 0.5), xytext=(0.7, 0.5),
                    arrowprops=dict(arrowstyle="-", color="black", lw=1.5))
        for tick in [1.0, 1.5, 2.0, 2.5, 3.0]:
            ax.plot(tick, 0.5, "|", color="black", markersize=7, markeredgewidth=1.5)
            ax.text(tick, 0.3, f"{tick:.1f}", ha="center", va="top", fontsize=8)

        ax.text(0.65, 0.5, "rank →", ha="right", va="center", fontsize=7, color="gray")

        y_levels = [1.0, 1.4, 1.8]
        for j, (sel, label, color) in enumerate(zip(SELECTIONS, SELECTION_LABELS, COLORS)):
            rank = avg_ranks[j]
            y_top = y_levels[j]
            ax.plot(rank, 0.5, "o", color=color, markersize=10, zorder=5)
            ax.plot([rank, rank], [0.5, y_top], color=color, linewidth=1.2, linestyle="--", alpha=0.7)
            ax.text(rank, y_top + 0.06, f"{label}\n({rank:.3f})", ha="center", va="bottom",
                    fontsize=8.5, color=color, fontweight="bold")

        bracket_y = -0.2
        for (a, b) in [(0, 1), (0, 2), (1, 2)]:
            pv = ph.iloc[a, b]
            if pv >= 0.05:
                x1, x2 = sorted([avg_ranks[a], avg_ranks[b]])
                ax.plot([x1, x2], [bracket_y, bracket_y], color="black", linewidth=3.5)
                for x in [x1, x2]:
                    ax.plot([x, x], [bracket_y - 0.06, bracket_y + 0.06], color="black", linewidth=3.5)
                bracket_y -= 0.22

        p_bh = r["p_bh"]
        sig = "***" if p_bh < 0.001 else "**" if p_bh < 0.01 else "*" if p_bh < 0.05 else "ns"
        ax.set_title(
            f"{METRIC_LABELS[metric]}\nFriedman χ²={r['chi2']:.1f},  p_BH={p_bh:.2e} {sig}\nCD={cd:.4f},  n={r['n']}",
            fontsize=9, pad=4,
        )

    fig.suptitle(
        "Critical Difference Diagrams — Nemenyi post-hoc (α=0.05)\n"
        "Rank 1 = worst, Rank 3 = best.  Bracket = not significantly different.",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_violins(df, save_path):
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, metric in enumerate(ALL_METRICS):
        ax = axes[i]
        split_val = "forget" if metric in FORGET_METRICS else "test"
        sub = df[df["split"] == split_val]
        data_by_sel = [sub[sub["selection"] == s][metric].dropna().values for s in SELECTIONS]

        vp = ax.violinplot(data_by_sel, positions=[1, 2, 3], showmedians=True, showextrema=False)
        for body, color in zip(vp["bodies"], COLORS):
            body.set_facecolor(color)
            body.set_alpha(0.55)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(2)

        for j, (vals, color) in enumerate(zip(data_by_sel, COLORS)):
            n_pts = min(400, len(vals))
            sampled = rng.choice(vals, size=n_pts, replace=False)
            jitter = rng.uniform(-0.12, 0.12, size=n_pts)
            ax.scatter(j + 1 + jitter, sampled, color=color, alpha=0.25, s=6, zorder=3)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(SELECTION_LABELS, fontsize=9)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.grid(axis="y", alpha=0.3, linewidth=0.7)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6) for c in COLORS]
    fig.legend(handles, SELECTION_LABELS, loc="upper right", fontsize=9, framealpha=0.8)
    fig.suptitle("Score Distributions per Selection Method (all conditions pooled)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_winrate(results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    baselines = [("Embedding", "wtl_emb"), ("RASLIK", "wtl_raslik")]

    for ax, (blabel, key) in zip(axes, baselines):
        xlabels = [METRIC_LABELS[m] for m in ALL_METRICS]
        x = np.arange(len(ALL_METRICS))
        totals = np.array([results[m]["n"] for m in ALL_METRICS], dtype=float)
        wins   = np.array([results[m][key][0] for m in ALL_METRICS], dtype=float)
        ties   = np.array([results[m][key][1] for m in ALL_METRICS], dtype=float)
        losses = np.array([results[m][key][2] for m in ALL_METRICS], dtype=float)

        w_pct = wins / totals
        t_pct = ties / totals
        l_pct = losses / totals

        ax.bar(x, w_pct, color="#55A868", label="Win", zorder=3)
        ax.bar(x, t_pct, bottom=w_pct, color="#CCCCCC", label="Tie", zorder=3)
        ax.bar(x, l_pct, bottom=w_pct + t_pct, color="#C44E52", label="Loss", zorder=3)

        for j in range(len(ALL_METRICS)):
            if w_pct[j] > 0.05:
                ax.text(x[j], w_pct[j] / 2, str(int(wins[j])),
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            if t_pct[j] > 0.04:
                ax.text(x[j], w_pct[j] + t_pct[j] / 2, str(int(ties[j])),
                        ha="center", va="center", fontsize=7.5, color="#333333")
            if l_pct[j] > 0.05:
                ax.text(x[j], w_pct[j] + t_pct[j] + l_pct[j] / 2, str(int(losses[j])),
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Proportion of questions", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"Ours vs {blabel}", fontsize=11)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.7, zorder=0)
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", zorder=2)

    fig.suptitle("Win / Tie / Loss Rate of Ours vs Baselines (per question)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    print("Loading data...")
    df = load_all()
    print(
        f"Loaded {len(df):,} rows — "
        f"{df['algo'].nunique()} algos, {df['model'].nunique()} models, "
        f"{df['dataset'].nunique()} datasets, {df['selection'].nunique()} selections"
    )

    print("Running statistical tests...")
    results = run_analysis(df)

    print_results(results)

    out = Path(__file__).parent
    print("\nGenerating plots...")
    plot_cd_diagram(results, out / "cd_diagram.png")
    plot_violins(df, out / "violin_plots.png")
    plot_winrate(results, out / "winrate_chart.png")
