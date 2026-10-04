"""Paper figures (ACM single column, print-safe) from evaluation outputs.

Fonts: --font clean (Libertinus/Linux Libertine if installed to match acmart, else Arial/DejaVu)
or --font comic (Comic Sans MS). PDFs embed TrueType fonts (pdf.fonttype 42): acmart rejects Type 3.
Colors: validated categorical slots in fixed order (dataviz reference palette, light mode);
series also differ by hatch so figures survive grayscale printing and color-vision deficiency.
Every figure also writes the plotted numbers as CSV (table view). Missing inputs -> figure skipped.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from .config import Config  # noqa: E402

log = logging.getLogger(__name__)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
HATCH = ["", "////", "....", "xxxx", "\\\\\\\\", "++", "oo", "--"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
COL_W = 3.33   # ACM sigconf column width (in)
FONTS = {
    "clean": ["Libertinus Sans", "Linux Biolinum O", "Linux Libertine O", "Arial", "Helvetica", "DejaVu Sans"],
    "comic": ["Comic Sans MS", "Comic Neue", "DejaVu Sans"],
}
METHOD_LABEL = {"direct": "B0 Direct", "coverage": "B1 Coverage", "freqframe": "B2 Freq. prompt",
                "plan:frequency": "B3 Freq. plan", "plan:frequency_unverified": "B3 unverified"}


def setup_style(font: str = "clean") -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in FONTS[font] if f in installed), "DejaVu Sans")
    if chosen != FONTS[font][0]:
        log.info("font %s not installed; using %s", FONTS[font][0], chosen)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [chosen] + FONTS[font],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "hatch.linewidth": 0.5,
        "hatch.color": SURFACE,
    })
    return chosen


def _order(m: str) -> int:
    keys = list(METHOD_LABEL)
    return keys.index(m) if m in keys else len(keys)


def _label(m: str) -> str:
    return METHOD_LABEL.get(m, m)


def _save(fig, out: Path, name: str, data: pd.DataFrame, formats) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=300)
        paths.append(str(p))
    data.to_csv(out / f"{name}.csv", index=False)
    plt.close(fig)
    return paths


def fig_paired_differences(paths: list[Path], out, formats, title: str, name: str):
    """Forest plot: paired difference vs B0 (product bootstrap 95% CI) per metric facet."""
    frames = [pd.read_csv(p).assign(source=p.stem) for p in paths if p.exists()]
    if not frames:
        return []
    d = pd.concat(frames).dropna(subset=["diff"])
    metrics = list(dict.fromkeys(d.metric))
    methods = sorted(dict.fromkeys(d.b), key=_order)
    fig, axes = plt.subplots(1, len(metrics), figsize=(COL_W, 0.35 * len(methods) + 0.7),
                             sharey=True, squeeze=False)
    for ax, met in zip(axes[0], metrics):
        x = d[d.metric == met].set_index("b").reindex(methods)
        y = np.arange(len(methods))[::-1]
        ax.axvline(0, color=INK2, lw=0.8)
        ax.errorbar(x["diff"], y, xerr=[x["diff"] - x.ci_low, x.ci_high - x["diff"]], fmt="o",
                    ms=4, color=SERIES[0], ecolor=SERIES[0], elinewidth=1.2, capsize=0)
        ax.set_title(met.replace("_", " "), color=INK)
        ax.grid(axis="y", visible=False)
        ax.set_yticks(y, [_label(m) for m in methods])
    axes[0][0].set_xlabel(f"difference vs {_label(d.a.iloc[0])} (95% CI)")
    fig.subplots_adjust(wspace=0.15)  # the title belongs in the LaTeX caption
    return _save(fig, out, name, d, formats)


def fig_interventions(path: Path, out, formats, name="fig_interventions"):
    """Two panels (never a dual axis): joint coverage (target kept) and removal error (target removed)."""
    if not path.exists():
        return []
    d = pd.read_csv(path)
    models = list(dict.fromkeys(d.model_id))
    panels = [("target_preserved", "joint_coverage", "Joint coverage\n(higher is better)"),
              ("target_removed", "removal_error", "Removal error\n(lower is better)")]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.0))
    fig.subplots_adjust(wspace=0.35)
    rows = []
    for ax, (typ, col, ttl) in zip(axes, panels):
        x = d[d.type == typ]
        if x.empty or col not in x:
            ax.set_visible(False)
            continue
        g = x.groupby(["method", "model_id"]).apply(
            lambda s: np.average(s[col], weights=s.n_pairs), include_groups=False).unstack()
        g = g.reindex(columns=models)
        g = g.reindex(sorted(g.index, key=_order))
        meths = list(g.index)
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            ax.bar(np.arange(len(meths)) + (i - (len(models) - 1) / 2) * w, g[m], w,
                   color=SERIES[i], hatch=HATCH[i], edgecolor=SURFACE, linewidth=0.8, label=m)
        ax.set_xticks(np.arange(len(meths)), [_label(m) for m in meths], rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(ttl)
        ax.grid(axis="x", visible=False)
        rows.append(g.reset_index().assign(panel=col))
    axes[0].set_ylabel("share of pairs")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.25), ncol=len(models))
    return _save(fig, out, name, pd.concat(rows) if rows else d, formats)


def _boot_ci(v: np.ndarray, reps=2000, seed=0):
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(reps)]
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def fig_group_representation(path: Path, out, formats, cohort="test_disclosure", name="fig_du_representation"):
    """Dumbbell: RR_D vs RR_U per generator x method (product means, product-bootstrap 95% CI)."""
    if not path.exists():
        return []
    m = pd.read_parquet(path)
    m = m[(m.cohort == cohort) & (m.variant == "original")] if "cohort" in m else m
    if m.empty or "RR_D" not in m:
        return []
    rows = []
    for (model, meth), g in m.groupby(["model_id", "method"]):
        prod = g.groupby("product_id")[["RR_D", "RR_U"]].mean()
        for grp in ("D", "U"):
            mu, lo, hi = _boot_ci(prod[f"RR_{grp}"].to_numpy())
            rows.append({"model_id": model, "method": meth, "group": grp, "mean": mu, "ci_low": lo,
                         "ci_high": hi, "n_products": int(prod[f"RR_{grp}"].notna().sum())})
    d = pd.DataFrame(rows)
    keys = list(dict.fromkeys(zip(d.model_id, d.method)))
    fig, ax = plt.subplots(figsize=(COL_W, 0.3 * len(keys) + 0.8))
    y = np.arange(len(keys))[::-1]
    for yi, (model, meth) in zip(y, keys):
        s = d[(d.model_id == model) & (d.method == meth)].set_index("group")
        ax.plot([s.at["U", "mean"], s.at["D", "mean"]], [yi, yi], color=GRID, lw=2, zorder=1)
    for i, (grp, lab) in enumerate([("U", "No disclosure (U)"), ("D", "Disclosed (D)")]):
        s = d[d.group == grp].set_index(["model_id", "method"]).reindex(keys)
        ax.errorbar(s["mean"], y, xerr=[s["mean"] - s.ci_low, s.ci_high - s["mean"]], fmt="os"[i],
                    ms=5, color=SERIES[i], elinewidth=1, capsize=0, label=lab, zorder=2,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.set_yticks(y, [f"{mo} | {_label(me)}" for mo, me in keys])
    ax.set_xlabel("share of reviews linked to ≥1 summary claim (RR)")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    return _save(fig, out, name, d, formats)


def fig_judges(eval_csv: Path, curve_json: Path, out, formats, name="fig_judges"):
    """Left: SUPPORT / OPPOSE precision & recall per judge (weighted). Right: coverage-error curves."""
    if not eval_csv.exists():
        return []
    e = pd.read_csv(eval_csv)
    e = e[e.slice == "all_weighted"]
    judges = list(e.judge)
    cols = ["SUPPORT_precision", "SUPPORT_recall", "OPPOSE_precision", "OPPOSE_recall"]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.8), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.subplots_adjust(wspace=0.45)
    ax = axes[0]
    w = 0.8 / len(judges)
    for i, j in enumerate(judges):
        v = e.set_index("judge").loc[j, cols].astype(float)
        ax.bar(np.arange(len(cols)) + (i - (len(judges) - 1) / 2) * w, v, w, color=SERIES[i],
               hatch=HATCH[i], edgecolor=SURFACE, linewidth=0.8, label=j)
    ax.axhline(0.9, color=INK2, lw=0.6, ls="--")
    ax.text(len(cols) - 0.5, 0.91, "gate .90", ha="right", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(np.arange(len(cols)), ["SUP\nprec.", "SUP\nrec.", "OPP\nprec.", "OPP\nrec."])
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(judges))
    ax2 = axes[1]
    curves = json.loads(curve_json.read_text()) if curve_json.exists() else {}
    for i, j in enumerate(judges):
        c = pd.DataFrame(curves.get(j, []))
        if len(c):
            ax2.plot(c.coverage, c.error, color=SERIES[i], lw=2, marker="os^"[i % 3], ms=3)
    ax2.set_xlabel("coverage")
    ax2.set_ylabel("error rate")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(bottom=0)
    if not curves:
        ax2.set_visible(False)
    return _save(fig, out, name, e[["judge"] + cols + ["macro_f1"]], formats)


def make_all(cfg: Config, font: str = "clean", formats=("pdf", "png"), judge: str = "nimble") -> list[str]:
    chosen = setup_style(font)
    o, out = cfg.out_dir, cfg.run_dir / "figures" / font
    made = []
    made += fig_paired_differences([o / "preservation_paired_bootstrap.csv"], out, formats,
                                   "Human gold: paired difference vs B0", "fig_preservation_human")
    made += fig_paired_differences([o / f"auto_recall_paired_bootstrap_{judge}.csv"], out, formats,
                                   "Automatic recall: paired difference vs B0", "fig_recall_auto")
    iv = o / f"intervention_metrics_auto_{judge}.csv"
    made += fig_interventions(iv if iv.exists() else o / "intervention_metrics.csv", out, formats)
    reps = sorted(o.glob(f"metrics_by_summary_{judge}_all_*.parquet")) or sorted(o.glob(f"metrics_by_summary_{judge}_*.parquet"))
    if reps:
        made += fig_group_representation(reps[0], out, formats)
    made += fig_judges(o / "judge_eval.csv", o / "judge_coverage_error.json", out, formats)
    log.info("font used: %s; %d files", chosen, len(made))
    return made
