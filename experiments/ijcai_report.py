"""Pre-registered analysis (PREREGISTRATION_IJCAI.md) -> results/ijcai/REPORT.md

Runs exactly the tests written down before the experiments; every hypothesis is reported whether or not it
holds. Real-data intervals resample rings (all seeds of a ring together).
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from experiments.common import out_dir
from src.ring_bootstrap import paired_ring_bootstrap

R = out_dir("ijcai")
ROLL = {"AMLworld HI-Small": "rolling_hi_small", "AMLworld LI-Small": "rolling_li_small"}
PRIMARY_C = 0.15


def _load_rolling() -> pd.DataFrame:
    parts = []
    for ds, d in ROLL.items():
        p = os.path.join(out_dir(d), "ring_tables.csv")
        if os.path.exists(p):
            t = pd.read_csv(p).assign(dataset=ds)
            t["ring_key"] = ds + "|" + t.test_day.astype(str) + "|" + t.ring_id.astype(str)
            parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _sign_test(rt: pd.DataFrame, model: str, c: float, treat: str = "TVW") -> dict:
    g = rt[(rt.model == model) & (rt.budget_multiple == c)]
    a = g[g.variant == "control"].groupby("ring_key").txns_before_alert.mean()
    b = g[g.variant == treat].groupby("ring_key").txns_before_alert.mean().reindex(a.index)
    earlier, later = int((b < a - 1e-9).sum()), int((b > a + 1e-9).sum())
    n = earlier + later
    p = stats.binomtest(earlier, n, 0.5, alternative="greater").pvalue if n else float("nan")
    return {"rings": len(a), "earlier": earlier, "later": later, "p_one_sided": p}


def _holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        adj[k] = run
    return adj


def _md(df: pd.DataFrame) -> str:
    """Markdown table without the optional 'tabulate' dependency."""
    if not len(df):
        return "_no data_"
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    return "\n".join([head, sep] + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)])


def main() -> None:
    L = [f"# IJCAI confirmatory report\n\nGenerated {datetime.now():%Y-%m-%d %H:%M}. Analysis plan: "
         "`PREREGISTRATION_IJCAI.md` (written before these experiments).\n"]
    rt = _load_rolling()
    verdicts = []
    if len(rt):
        prim = rt[rt.primary]
        sec = rt[~rt.primary]
        # ---------------- H1
        L.append("## H1 (primary): neighbour-aggregation XGBoost + TVW catches rings earlier (c = 0.15)\n")
        h1 = []
        for label, sub in [("PRIMARY windows (pre-registered test)", prim), ("secondary windows", sec), ("all windows", rt)]:
            if len(sub):
                h1.append({"windows": label, **_sign_test(sub, "Neighbour-Aggregation XGBoost", PRIMARY_C)})
        h1 = pd.DataFrame(h1)
        L.append(_md(h1) + "\n")
        if len(h1):
            x = h1.iloc[0]
            v = "NO DATA" if x.rings == 0 else ("SUPPORTED" if x.p_one_sided < 0.05 else "NOT SUPPORTED")
            verdicts.append(("H1", v, f"{x.earlier} earlier vs {x.later} later, p = {x.p_one_sided:.4f}"))
        # ---------------- H3
        L.append("\n## H3 (secondary): temporal GNNs + TVW catch rings earlier (c = 0.15, primary windows, Holm)\n")
        h3 = pd.DataFrame([{"model": m, **_sign_test(prim, m, PRIMARY_C)} for m in ["Temporal GraphSAGE", "Temporal GAT"]])
        if len(h3):
            h3["p_holm"] = _holm(h3.p_one_sided.fillna(1).to_numpy())
            for _, x in h3.iterrows():
                v = "NO DATA" if x.rings == 0 else ("SUPPORTED" if x.p_holm < 0.05 else "NOT SUPPORTED")
                verdicts.append((f"H3 {x.model}", v, f"p_holm = {x.p_holm:.4f}"))
        L.append(_md(h3) + "\n")
        # ---------------- H2
        L.append("\n## H2: TVW money gain vs RECOVERABLE headroom (primary windows; Spearman, one-sided)\n")
        cells = []
        for (ds, day, m, c), g in prim.groupby(["dataset", "test_day", "model", "budget_multiple"]):
            a, b = g[g.variant == "control"], g[g.variant == "TVW"]
            if a.empty or b.empty:
                continue
            tot = a.ring_amount.sum()
            cells.append({"dataset": ds, "test_day": day, "model": m, "budget": c,
                          "recoverable": a.loc[a.detected, "amount_before_alert"].sum() / tot,
                          "gain": (a.amount_before_alert.sum() - b.amount_before_alert.sum()) / tot})
        cells = pd.DataFrame(cells)
        if len(cells) > 2:
            rho, p2 = stats.spearmanr(cells.recoverable, cells.gain)
            p2 = p2 / 2 if rho > 0 else 1 - p2 / 2
            L.append(f"cells = {len(cells)}, Spearman rho = {rho:.3f}, one-sided p = {p2:.4f}\n")
            verdicts.append(("H2", "SUPPORTED" if (rho > 0 and p2 < 0.05) else "NOT SUPPORTED", f"rho = {rho:.3f}, p = {p2:.4f}"))
        # ---------------- full real-data table (money + counts, ring-level CIs)
        L.append("\n## Real data: TVW and baselines vs control (all budgets; primary windows)\n")
        rows = []
        for (ds, m, c), g in prim.groupby(["dataset", "model", "budget_multiple"]):
            a = g[g.variant == "control"].assign(ring_id=lambda x: x.ring_key)
            for v in sorted(set(g.variant) - {"control"}):
                b = g[g.variant == v].assign(ring_id=lambda x: x.ring_key)
                if len(b):
                    rows.append({"dataset": ds, "model": m, "budget": c, "variant": v, **paired_ring_bootstrap(a, b)})
        full = pd.DataFrame(rows)
        if len(full):
            full.to_csv(os.path.join(R, "real_data_paired.csv"), index=False)
            L.append(_md(full[["dataset", "model", "budget", "variant", "n_rings", "delta_pp", "ci_lo_pp", "ci_hi_pp",
                               "rings_earlier", "rings_later", "largest_ring_money_share"]]) + "\n")
        runs = pd.concat([pd.read_csv(os.path.join(out_dir(d), "runs.csv")).assign(dataset=ds)
                          for ds, d in ROLL.items() if os.path.exists(os.path.join(out_dir(d), "runs.csv"))])
        L.append("\n### PR-AUC (primary windows, mean)\n")
        L.append(_md(runs[runs.primary].groupby(["dataset", "model", "variant"]).pr_auc.mean().reset_index()) + "\n")
    # ---------------- H4 (simulator objectives)
    sp = os.path.join(out_dir("objectives_sim"), "ring_tables.csv")
    if os.path.exists(sp):
        L.append("\n## H4: simulator, TVW vs reweighting baselines (hard preset, 0.5% budget)\n")
        s = pd.read_csv(sp)
        s = s[s.budget == 0.005]
        s["ring_key"] = s.seed.astype(str) + "|" + s.ring_id.astype(str)
        rows = []
        rng = np.random.default_rng(0)
        for m, g in s.groupby("model"):
            tv = g[g.variant == "TVW"].set_index("ring_key")
            for base in sorted(set(g.variant) - {"TVW", "control"}):  # H4 family: reweighting baselines only
                bb = g[g.variant == base].set_index("ring_key").reindex(tv.index)
                amt, lt, lb = tv.ring_amount.to_numpy(), tv.amount_before_alert.to_numpy(), bb.amount_before_alert.to_numpy()
                n = len(tv)
                boots = np.array([(lt[i].sum() - lb[i].sum()) / amt[i].sum() for i in (rng.integers(0, n, n) for _ in range(4000))])
                rows.append({"model": m, "TVW vs": base, "delta_pp": 100 * (lt.sum() - lb.sum()) / amt.sum(),
                             "ci_lo_pp": 100 * np.percentile(boots, 2.5), "ci_hi_pp": 100 * np.percentile(boots, 97.5),
                             "p_one_sided": float((boots >= 0).mean())})
        h4 = pd.DataFrame(rows)
        ctx = s.groupby(["model", "variant"]).apply(lambda g: g.amount_before_alert.sum() / g.ring_amount.sum(),
                                                    include_groups=False).unstack()
        L.append("Context (not part of H4): money lost before first alert by variant\n\n" + _md((100 * ctx).round(2).reset_index()) + "\n")
        if len(h4):
            h4["p_holm"] = _holm(h4.p_one_sided.to_numpy())
            L.append(_md(h4) + "\n")
            for _, x in h4.iterrows():
                verdicts.append((f"H4 {x.model} vs {x['TVW vs']}", "SUPPORTED" if x.p_holm < 0.05 else "NOT SUPPORTED",
                                 f"delta {x.delta_pp:+.2f} pp, p_holm = {x.p_holm:.4f}"))
    # ---------------- verdict summary at the top
    head = ["## Verdicts (pre-registered)\n", "| hypothesis | verdict | statistic |", "|---|---|---|"]
    head += [f"| {h} | **{v}** | {s} |" for h, v, s in verdicts]
    L.insert(1, "\n".join(head) + "\n")
    with open(os.path.join(R, "REPORT.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()


def run(**_) -> None:
    main()
