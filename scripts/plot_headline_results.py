#!/usr/bin/env python3
"""
Paper-ready experiment plots and LaTeX tables from headline JSONL.

Design goals
------------
* **Non-DP utility**: one compact figure (test acc vs. K) + a small table.
* **DP runs**: main figures fix K (default 8) so curves are only
  (Algo 2 vs 3) × (q = 0.05 vs 0.10) = 4 lines + ceiling reference.
* **Hyperparameter grid**: tables at fixed σ=1 for (i) effect of q at each K
  and (ii) effect of K at q=0.10 — readable instead of 12-line plots.

Outputs (under --out-dir):
  utility_nondp.png
  dp_K{K}_<dataset>.png  (two panels: noise, Pareto)
  tables/tab_nondp_utility.tex
  tables/tab_dp_sigma1_q0p10_byK.tex   (vary K; DP σ=1, q=0.10)
  tables/tab_dp_sigma1_K8_byq.tex      (vary q; DP σ=1, K=8)
  dp_runs_summary.csv, nondp_subgraph_ceiling.csv

Usage:
  python scripts/plot_headline_results.py \\
    --data-dir ~/Downloads/headline \\
    --out-dir paper/figures/experiments \\
    --fixed-k 8
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def infer_q_from_path(path: Path) -> float | None:
    # Filenames use _q005_ → q=0.05, _q010_ → q=0.10 (hundredths, not thousandths).
    m = re.search(r"_q(\d{3})_", path.name)
    if m:
        return int(m.group(1)) / 100.0
    return None


def load_all_jsonl(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for p in sorted(root.rglob("*.jsonl")):
        if "dpmlp" in p.name.lower():
            continue
        q_file = infer_q_from_path(p)
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["_file"] = p.name
                r["_q_file"] = q_file
                rows.append(r)
    return pd.DataFrame(rows)


def prep_nondp(df: pd.DataFrame) -> pd.DataFrame:
    m = (df["dp"] == False) | (df["dp"].isna())
    m &= df["model"] == "gcn"
    m &= df["method"].isin(["Algo 2", "Algo 3"])
    m &= df["num_bins"].notna()
    return df.loc[m].copy()


def prep_dp(df: pd.DataFrame) -> pd.DataFrame:
    m = df["dp"] == True
    m &= df["model"] == "gcn"
    m &= df["method"].isin(["Algo 2", "Algo 3"])
    m &= df["noise_multiplier"].notna()
    m &= df["computed_epsilon"].notna()
    out = df.loc[m].copy()
    out["algo_label"] = out["method"].map({"Algo 2": "A2", "Algo 3": "A3"})
    return out


def prep_nondp_ceiling(nondp: pd.DataFrame) -> pd.Series:
    return nondp.groupby("dataset")["test_acc"].max()


def _dataset_title(ds: str) -> str:
    return {
        "ogbn-arxiv": "OGBN-Arxiv",
        "ogbn-products": "OGBN-Products",
        "reddit": "Reddit",
    }.get(ds, ds)


def plot_nondp_utility(nondp: pd.DataFrame, out_path: Path, dpi: int) -> None:
    """Three panels: test accuracy vs. number of bins K (no DP)."""
    nd = nondp[nondp["num_bins"].isin([4, 8, 16])].copy()
    g = (
        nd.groupby(["dataset", "num_bins", "method"])["test_acc"]
        .mean()
        .reset_index()
    )
    # Modest physical size so LaTeX scaling does not need aggressive shrink.
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=False)
    for ax, ds in zip(axes, ["ogbn-arxiv", "reddit", "ogbn-products"]):
        sub = g[g["dataset"] == ds].sort_values("num_bins")
        if sub.empty:
            ax.set_visible(False)
            continue
        for method, sty in [("Algo 2", "-"), ("Algo 3", "--")]:
            s2 = sub[sub["method"] == method]
            if s2.empty:
                continue
            ax.plot(
                s2["num_bins"],
                s2["test_acc"],
                marker="o",
                ms=7,
                lw=2.2,
                linestyle=sty,
                label=method.replace("Algo ", "Algo.~"),
            )
        ax.set_xticks(sorted(sub["num_bins"].unique()))
        ax.set_xlabel(r"Number of bins $K$")
        ax.set_ylabel("Test accuracy")
        ax.set_title(_dataset_title(ds))
        ax.legend(frameon=True)
    fig.suptitle("Non-private subgraph training (same GCN, no clipping or noise)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, format="png", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_dp_focus_k(
    dp: pd.DataFrame,
    ceiling: pd.Series,
    dataset: str,
    fixed_k: int,
    out_path: Path,
    dpi: int,
) -> None:
    """Noise + Pareto with only num_bins == fixed_k (four curves: A2/A3 × two q)."""
    d = dp[(dp["dataset"] == dataset) & (dp["num_bins"] == fixed_k)].copy()
    if d.empty:
        return

    fig, (ax_n, ax_p) = plt.subplots(2, 1, figsize=(5.8, 5.0), constrained_layout=True)
    fig.suptitle(
        rf"{_dataset_title(dataset)} — DP ($K={fixed_k}$ only)",
        fontsize=12,
        fontweight="600",
    )

    styles = {
        ("A2", 0.05): ("#1f77b4", (0, (4, 2)), "o"),
        ("A2", 0.1): ("#1f77b4", "-", "s"),
        ("A3", 0.05): ("#ff7f0e", (0, (4, 2)), "o"),
        ("A3", 0.1): ("#ff7f0e", "-", "s"),
    }

    for (algo, q), g in d.groupby(["algo_label", "_q_file"]):
        qf = float(q)
        # Normalize 0.1 vs 0.10 for dict lookup
        qk = 0.1 if abs(qf - 0.1) < 1e-6 else (0.05 if abs(qf - 0.05) < 1e-6 else round(qf, 2))
        key = (algo, qk)
        if key not in styles:
            continue
        color, ls, mk = styles[key]
        g = g.sort_values("noise_multiplier")
        lab = f"{algo}, $q={qf:g}$"
        ax_n.plot(
            g["noise_multiplier"],
            g["test_acc"],
            color=color,
            linestyle=ls,
            marker=mk,
            markersize=6,
            lw=2.0,
            label=lab,
        )
        g2 = g.sort_values("computed_epsilon")
        ax_p.plot(
            g2["computed_epsilon"],
            g2["test_acc"],
            color=color,
            linestyle=ls,
            marker=mk,
            markersize=6,
            lw=2.0,
            label=lab,
        )

    if dataset in ceiling.index:
        y0 = float(ceiling[dataset])
        ax_n.axhline(y0, color="0.25", lw=1.5, ls=":", label="Best non-DP subgraph (A2/A3)")
        ax_p.axhline(y0, color="0.25", lw=1.5, ls=":", label="_nolegend_")

    ax_n.set_xlabel(r"Noise multiplier $\sigma/C$ ($\log_2$ scale)")
    ax_n.set_ylabel("Test accuracy")
    ax_n.set_xscale("log", base=2)
    ax_n.set_title("Accuracy vs. noise")

    ax_p.set_xlabel(r"PRV $\varepsilon$ ($\delta=10^{-5}$, log scale)")
    ax_p.set_ylabel("Test accuracy")
    ax_p.set_xscale("log")
    ax_p.invert_xaxis()
    ax_p.set_title("Privacy--utility (left is tighter privacy)")

    h, lab = ax_n.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(out_path, dpi=dpi, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def write_tex_tables(
    nondp: pd.DataFrame,
    dp: pd.DataFrame,
    out_tables: Path,
    fixed_k: int,
) -> list[Path]:
    out_tables.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # --- tab_nondp_utility.tex: Dataset, K, A2, A3 (mean over seeds); $K\\in\\{4,8,16\\}$ only
    nd_tab = nondp[nondp["num_bins"].isin([4, 8, 16])].copy()
    g = (
        nd_tab.groupby(["dataset", "num_bins", "method"])["test_acc"]
        .mean()
        .unstack("method")
        .rename(columns={"Algo 2": "a2", "Algo 3": "a3"})
        .reset_index()
    )
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & $K$ & Algo.~2 & Algo.~3 \\",
        r"\midrule",
    ]
    ds_tex_map = {
        "ogbn-arxiv": r"\texttt{ogbn-arxiv}",
        "ogbn-products": r"\texttt{ogbn-products}",
        "reddit": r"\texttt{reddit}",
    }
    for ds in ["ogbn-arxiv", "reddit", "ogbn-products"]:
        sub = g[g["dataset"] == ds].sort_values("num_bins")
        ds_tex = ds_tex_map[ds]
        for _, row in sub.iterrows():
            k = int(row["num_bins"])
            a2 = _fmt(float(row["a2"])) if pd.notna(row.get("a2")) else "---"
            a3 = _fmt(float(row["a3"])) if pd.notna(row.get("a3")) else "---"
            lines.append(f"{ds_tex} & {k} & {a2} & {a3} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    p = out_tables / "tab_nondp_utility.tex"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(p)

    # --- DP: σ=1, q=0.10, vary K (one row per dataset × K)
    d_sig = dp[
        (dp["noise_multiplier"] == 1.0)
        & dp["_q_file"].notna()
        & ((dp["_q_file"] - 0.1).abs() < 0.02)
    ]
    rows_out = []
    for ds in ["ogbn-arxiv", "reddit", "ogbn-products"]:
        sub = d_sig[d_sig["dataset"] == ds]
        for k in sorted(sub["num_bins"].unique()):
            a2 = sub[(sub["num_bins"] == k) & (sub["algo_label"] == "A2")]
            a3 = sub[(sub["num_bins"] == k) & (sub["algo_label"] == "A3")]
            if len(a2) and len(a3):
                rows_out.append(
                    (
                        ds,
                        int(k),
                        float(a2["test_acc"].iloc[0]),
                        float(a3["test_acc"].iloc[0]),
                        float(a2["computed_epsilon"].iloc[0]),
                        float(a3["computed_epsilon"].iloc[0]),
                    )
                )
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Dataset & $K$ & Algo.~2 acc. & Algo.~3 acc. & $\varepsilon$ (A2) & $\varepsilon$ (A3) \\",
        r"\midrule",
    ]
    for ds, k, t2, t3, e2, e3 in rows_out:
        ds_tex = r"\texttt{" + ds + "}"
        lines.append(
            f"{ds_tex} & {k} & {_fmt(t2)} & {_fmt(t3)} & {_fmt(e2)} & {_fmt(e3)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    p = out_tables / "tab_dp_sigma1_q0p10_byK.tex"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(p)

    # --- DP: σ=1, K=fixed_k, vary q
    d_k = dp[(dp["noise_multiplier"] == 1.0) & (dp["num_bins"] == fixed_k)]
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Dataset & $q$ & Algo.~2 acc. & Algo.~3 acc. & $\varepsilon$ (A2) & $\varepsilon$ (A3) \\",
        r"\midrule",
    ]
    for ds in ["ogbn-arxiv", "reddit", "ogbn-products"]:
        ds_tex = r"\texttt{" + ds + "}"
        sub = d_k[d_k["dataset"] == ds].sort_values("_q_file")
        for qv in sorted(sub["_q_file"].dropna().unique()):
            a2 = sub[(sub["_q_file"] == qv) & (sub["algo_label"] == "A2")]
            a3 = sub[(sub["_q_file"] == qv) & (sub["algo_label"] == "A3")]
            if len(a2) and len(a3):
                lines.append(
                    f"{ds_tex} & {float(qv):g} & {_fmt(float(a2['test_acc'].iloc[0]))} & "
                    f"{_fmt(float(a3['test_acc'].iloc[0]))} & "
                    f"{_fmt(float(a2['computed_epsilon'].iloc[0]))} & "
                    f"{_fmt(float(a3['computed_epsilon'].iloc[0]))} \\\\"
                )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    p = out_tables / f"tab_dp_sigma1_K{fixed_k}_byq.tex"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(p)

    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("~/Downloads/headline").expanduser())
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument(
        "--fixed-k",
        type=int,
        default=8,
        help="Number of bins for the main DP figures (fewer curves).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir or (repo_root / "paper" / "figures" / "experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    root = args.data_dir.expanduser()
    df = load_all_jsonl(root)
    if df.empty:
        raise SystemExit(f"No jsonl found under {root}")

    nondp = prep_nondp(df)
    dp = prep_dp(df)
    ceiling = prep_nondp_ceiling(nondp)

    written: list[Path] = []

    p_util = out_dir / "utility_nondp.png"
    plot_nondp_utility(nondp, p_util, args.dpi)
    written.append(p_util)

    fk = args.fixed_k
    for ds in ["ogbn-arxiv", "reddit", "ogbn-products"]:
        slug = ds.replace("ogbn-", "ogbn_")
        outp = out_dir / f"dp_K{fk}_{slug}.png"
        plot_dp_focus_k(dp, ceiling, ds, fk, outp, args.dpi)
        if outp.exists():
            written.append(outp)

    written.extend(write_tex_tables(nondp, dp, tables_dir, fk))

    summary = (
        dp.groupby(["dataset", "method", "num_bins", "_q_file", "noise_multiplier"])[
            ["test_acc", "computed_epsilon", "train_acc"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(out_dir / "dp_runs_summary.csv", index=False)
    ceiling.to_csv(out_dir / "nondp_subgraph_ceiling.csv", header=["max_test_acc_non_dp_subgraph"])
    written.extend([out_dir / "dp_runs_summary.csv", out_dir / "nondp_subgraph_ceiling.csv"])

    print("Wrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
