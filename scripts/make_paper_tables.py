#!/usr/bin/env python3
"""LaTeX tables for the paper (paper/tables/*.tex), generated from results/ only.

  tab_main.tex   hard preset: row-level and ring-level metrics for every model
  tab_real.tex   AMLworld and IEEE-CIS
  tab_tvw.tex    takedown-value weighting: paired ring-level effect, all datasets/models
  tab_delay.tex  label delay x ring lifetime (synthetic) -- PR-AUC gain of label features
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "tables")

NAME = {"Transaction Baseline": "Transaction features", "+ Relationship counts": "\\quad + relationship counts",
        "+ Temporal relationships": "\\quad + temporal relationships",
        "Combined (full graph)": "Hand-crafted graph features", "Combined + delayed labels": "\\quad + delayed labels",
        "Neighbour-Aggregation XGBoost": "Neighbour-aggregation XGBoost", "Temporal GraphSAGE": "Temporal GraphSAGE",
        "Temporal GAT": "Temporal GAT", "Temporal R-GCN": "Temporal R-GCN", "CARE-style GNN": "Temporal CARE-style GNN",
        "TGN-style Memory": "TGN-style memory", "GraphSAGE (entity, inductive)": "Entity GraphSAGE (inductive)",
        "GraphSAGE (transductive)": "Entity GraphSAGE (transductive)$^\\dagger$",
        "CARE-GNN (original code)": "CARE-GNN, original code (tuned)", "PC-GNN (original code)": "PC-GNN, original code (tuned)",
        "Vesta features (XGBoost)": "Vesta features", "Vesta + our graph features": "Vesta + hand-crafted graph",
        "Vesta + graph + delayed labels": "Vesta + graph + delayed labels",
        "Vesta + Neighbour-Aggregation XGBoost": "Vesta + neighbour aggregation", "Vesta + Temporal GAT": "Vesta + Temporal GAT"}
GNN = {"Temporal GraphSAGE", "Temporal GAT", "Temporal R-GCN", "CARE-style GNN", "TGN-style Memory",
       "GraphSAGE (entity, inductive)", "GraphSAGE (transductive)", "CARE-GNN (original code)", "PC-GNN (original code)",
       "Vesta + Temporal GAT"}


def _w(name: str, body: str) -> None:
    with open(os.path.join(OUT, name), "w") as f:
        f.write(body)


def _bold_best(vals, higher=True):
    v = np.array(vals, dtype=float)
    best = np.nanmax(v) if higher else np.nanmin(v)
    return [abs(x - best) < 1e-12 for x in v]


def tab_main() -> None:
    d = os.path.join(RES, "main_hard")
    m = pd.read_csv(os.path.join(d, "metrics_by_seed.csv"))
    s = m.groupby("model", sort=False).agg(pr=("pr_auc", "mean"), prs=("pr_auc", "std"), rec=("recall_at_1pct_fpr", "mean"))
    r = pd.read_csv(os.path.join(d, "ring_summary.csv")).pivot(index="model", columns="budget", values="amount_lost_share")
    w = pd.read_csv(os.path.join(d, "warmup.csv"))
    first = w[w.k == 1].set_index("model").recall
    rows = []
    order = ["Transaction Baseline", "+ Relationship counts", "+ Temporal relationships", "Combined (full graph)",
             "Combined + delayed labels", "Neighbour-Aggregation XGBoost", "TGN-style Memory",
             "GraphSAGE (entity, inductive)", "CARE-style GNN", "Temporal GraphSAGE", "Temporal R-GCN", "Temporal GAT"]
    for k in order:
        rows.append([k, s.loc[k, "pr"], s.loc[k, "prs"], s.loc[k, "rec"], r.loc[k, 0.001], r.loc[k, 0.005], first.get(k)])
    ot = os.path.join(RES, "originals_tuning", "summary.csv")
    if os.path.exists(ot):
        o = pd.read_csv(ot)
        o = o[(o.preset == "hard") & (o.config == "validation-tuned")]
        for _, x in o.iterrows():
            rows.append([x.model, x["mean"], x["std"], np.nan, np.nan, np.nan, np.nan])
    df = pd.DataFrame(rows, columns=["model", "pr", "prs", "rec", "l1", "l5", "first"])
    bp, br = _bold_best(df.pr), _bold_best(df.rec)
    b1, b5, bf = _bold_best(df.l1, False), _bold_best(df.l5, False), _bold_best(df["first"])

    def f(x, b, pct=False):
        if pd.isna(x):
            return "--"
        t = f"{100 * x:.1f}" if pct else f"{x:.3f}"
        return f"\\textbf{{{t}}}" if b else t
    lines = []
    for i, x in df.iterrows():
        sep = "\\midrule\n" if x.model in ("Neighbour-Aggregation XGBoost", "TGN-style Memory", "CARE-GNN (original code)") else ""
        pr = f(x.pr, bp[i]) + f"\\,$\\pm$\\,{x.prs:.3f}"
        lines.append(sep + f"{NAME[x.model]} & {pr} & {f(x.rec, br[i])} & {f(x.l1, b1[i], True)} & "
                     f"{f(x.l5, b5[i], True)} & {f(x['first'], bf[i], True)} \\\\")
    _w("tab_main.tex", "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
       " & \\multicolumn{2}{c}{Row level} & \\multicolumn{2}{c}{Money lost (\\%)} & 1st-txn \\\\\n"
       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
       "Model & PR-AUC & R@1\\% & @0.1\\% & @0.5\\% & recall (\\%) \\\\\n\\midrule\n" + "\n".join(lines) +
       "\n\\bottomrule\n\\end{tabular}\n")


def _per_seed(path: str, cols) -> pd.DataFrame:
    """mean / std / n per model from the per-seed file (written incrementally by the runs)."""
    m = pd.read_csv(path)
    out = {}
    for c in cols:
        if c in m.columns:
            g = m.groupby("model", sort=False)[c]
            out[c + "_mean"], out[c + "_std"], out[c + "_n"] = g.mean(), g.std(), g.count()
    return pd.DataFrame(out)


def tab_real() -> None:
    a = _per_seed(os.path.join(RES, "amlworld", "metrics_by_seed.csv"), ["pr_auc"])
    i = _per_seed(os.path.join(RES, "ieee", "metrics_by_seed.csv"), ["pr_auc", "pr_auc_first_fraud"])
    order = ["Transaction Baseline", "Combined (full graph)", "Combined + delayed labels", "Neighbour-Aggregation XGBoost",
             "Temporal GraphSAGE", "Temporal GAT", "GraphSAGE (entity, inductive)", "GraphSAGE (transductive)",
             "Vesta features (XGBoost)", "Vesta + our graph features", "Vesta + graph + delayed labels"]

    def cell(df, k, c):
        if k not in df.index or c + "_mean" not in df.columns or pd.isna(df.loc[k, c + "_mean"]):
            return "n/a"
        sd, n = df.loc[k, c + "_std"], df.loc[k, c + "_n"]
        t = f"{df.loc[k, c + '_mean']:.3f}" + (f"\\,$\\pm$\\,{sd:.3f}" if n > 1 else "")
        return t + ("" if n >= 3 else f"$^{{({int(n)})}}$")
    lines = []
    for k in order:
        sep = "\\midrule\n" if k in ("Vesta features (XGBoost)", "GraphSAGE (entity, inductive)") else ""
        lines.append(sep + f"{NAME[k]} & {cell(a, k, 'pr_auc')} & {cell(i, k, 'pr_auc')} & "
                     f"{cell(i, k, 'pr_auc_first_fraud')} \\\\")
    _w("tab_real.tex", "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
       " & AMLworld & \\multicolumn{2}{c}{IEEE-CIS} \\\\\n\\cmidrule(lr){2-2}\\cmidrule(lr){3-4}\n"
       "Model & PR-AUC & PR-AUC & PR-AUC (1st fraud) \\\\\n\\midrule\n" + "\n".join(lines) +
       "\n\\bottomrule\n\\end{tabular}\n")


def tab_tvw(budget: float = 0.005) -> None:
    parts = [pd.read_csv(os.path.join(RES, s, f)) for s in ("tvw", "tvw_gnn") for f in ("paired.csv",)
             if os.path.exists(os.path.join(RES, s, f))]
    summ = [pd.read_csv(os.path.join(RES, s, "summary.csv")) for s in ("tvw", "tvw_gnn")
            if os.path.exists(os.path.join(RES, s, "summary.csv"))]
    if not parts:
        return
    p = pd.concat(parts, ignore_index=True)
    s = pd.concat(summ, ignore_index=True)
    s = s[s.budget == budget].pivot_table(index=["dataset", "model"], columns="variant", values="pr_auc")
    p = p[(p.budget == budget) & (p.variant == "TVW (beta=1)")]
    order_ds = ["synthetic (hard)", "AMLworld", "IEEE-CIS"]  # default preset: small gains, reported in the text
    p = p[p.model != "Transaction Baseline"]
    lines = []
    for ds in order_ds:
        g = p[p.dataset == ds]
        if g.empty:
            continue
        lines.append(f"\\midrule\n\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{ds}}} ({int(g.n_rings.iloc[0])} rings)}} \\\\")
        rank = {m: i for i, m in enumerate(["Combined (full graph)", "Combined + delayed labels",
                                             "Neighbour-Aggregation XGBoost", "Temporal GraphSAGE", "Temporal GAT",
                                             "CARE-style GNN", "Vesta features (XGBoost)", "Vesta + our graph features",
                                             "Vesta + graph + delayed labels"])}
        for _, x in g.assign(_r=g.model.map(rank)).sort_values("_r").iterrows():
            sig = x.ci_hi_pp < 0
            d = f"{x.delta_pp:+.2f} [{x.ci_lo_pp:+.2f}, {x.ci_hi_pp:+.2f}]"
            dpr = s.loc[(ds, x.model), "TVW (beta=1)"] - s.loc[(ds, x.model), "control (beta=0)"]
            lines.append(f"{NAME.get(x.model, x.model)} & {100 * x.lost_control:.1f} & {100 * x.lost_tvw:.1f} & "
                         f"{'\\textbf{' + d + '}' if sig else d} & {100 * dpr:+.1f} \\\\")
    _w("tab_tvw.tex", "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
       " & \\multicolumn{2}{c}{Money lost (\\%)} & & $\\Delta$PR-AUC \\\\\n\\cmidrule(lr){2-3}\n"
       "Model & ordinary & TVW & $\\Delta$ pp [95\\% CI] & (pp) \\\\\n" + "\n".join(lines) +
       "\n\\bottomrule\n\\end{tabular}\n")


def tab_delay(delays=(0.0, 1.0, 7.0, 30.0, 90.0), lifetimes=("1-3d", "7-14d", "30-60d", "60-120d")) -> None:
    from scipy import stats
    g = pd.read_csv(os.path.join(RES, "c2", "grid.csv"))
    g = g[(g.reuse == 0.0) & g.lifetime.isin(lifetimes)]
    base = g[g.model == "Combined (no labels)"].groupby(["lifetime", "seed"]).pr_auc.mean()  # label-free: delay-independent
    lab = g[g.model == "Combined + delayed labels"].set_index(["lifetime", "seed", "delay"]).pr_auc
    gain = (100 * (lab - base.reindex(lab.index.droplevel("delay")).to_numpy())).rename("gain").reset_index()
    lines = []
    for lt in lifetimes:
        cells = []
        for d in delays:
            x = gain[(gain.lifetime == lt) & (gain.delay == d)].gain
            ci = stats.t.ppf(0.975, len(x) - 1) * x.std() / np.sqrt(len(x))
            t = f"{x.mean():+.2f}\\,$\\pm$\\,{ci:.2f}"
            cells.append(f"\\textbf{{{t}}}" if x.mean() - ci > 0 else t)
        lines.append(f"{lt.replace('d', ' d')} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(f"$D$\\,=\\,{d:g}\\,d" for d in delays)
    _w("tab_delay.tex", "\\begin{tabular}{@{}l" + "c" * len(delays) + "@{}}\n\\toprule\n"
       f"Ring lifetime & {head} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (tab_main, tab_real, tab_tvw, tab_delay):
        f()
        print("ok", f.__name__)
