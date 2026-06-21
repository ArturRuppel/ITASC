"""Matplotlib figures for the replicate-level analyses. Headless (Agg) and saved
with editable text (``svg.fonttype='none'`` / ``pdf.fonttype=42``), the same
convention as :mod:`..iris_export.figures`, so the SVGs drop into a manuscript.

Each function takes already-computed result objects from :mod:`.stats` (so the
inference and the drawing stay separate) and writes ``<stem>.png`` + ``<stem>.svg``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .stats import CorrelationResult, EnrichmentResult

_EDIT_RC = {"svg.fonttype": "none", "pdf.fonttype": 42}
#: Per-replicate colour/marker cycle (mirrors the Iris SuperPlot replicate encoding).
_REP_STYLE = [("#E8A33D", "o"), ("#5BA7DC", "s"), ("#3DBC8A", "^"),
              ("#9B72C7", "D"), ("#D1495B", "v")]
_INK = "#1f2937"
_GREY = "#94a3b8"


def _stars(p: float) -> str:
    return ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns")


def _save(fig, out_stem: Path) -> list[Path]:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context(_EDIT_RC):
        for ext in (".png", ".svg"):
            p = out_stem.with_suffix(ext)
            fig.savefig(p, dpi=300)
            paths.append(p)
    plt.close(fig)
    return paths


def correlation_by_split(res: CorrelationResult, out_stem: Path, *,
                         x_label: str, title: str) -> list[Path]:
    """Per-replicate correlation coefficient, one column per split level, with the
    replicate-mean ± 95% CI and the one-sample p vs zero."""
    split_col = [c for c in res.per_replicate.columns if c not in ("r", "n")]
    split_col = [c for c in split_col if c != _replicate_col(res)]
    levels = list(res.by_split)
    reps = sorted(res.per_replicate[_replicate_col(res)].unique())

    fig, ax = plt.subplots(figsize=(1.7 + 1.7 * len(levels), 5.0))
    rng = np.random.default_rng(0)
    for xi, level in enumerate(levels):
        sub = res.per_replicate
        if split_col:
            sub = sub[sub[split_col[0]].astype(str) == level]
        for rep, (colour, marker) in zip(reps, _REP_STYLE):
            v = sub.loc[sub[_replicate_col(res)] == rep, "r"]
            if v.empty:
                continue
            ax.scatter(xi + rng.uniform(-0.1, 0.1), v.iloc[0], s=80, color=colour,
                       marker=marker, edgecolor="white", linewidth=0.6, zorder=3,
                       label=rep if xi == 0 else None)
        mean_r, p = res.by_split[level]
        rs = np.arctanh(np.clip(sub["r"].dropna().to_numpy(), -0.999999, 0.999999))
        if len(rs) > 1:
            from scipy import stats as ss
            ci = ss.t.ppf(0.975, len(rs) - 1) * rs.std(ddof=1) / np.sqrt(len(rs))
            ax.errorbar(xi, mean_r,
                        yerr=[[mean_r - np.tanh(rs.mean() - ci)],
                              [np.tanh(rs.mean() + ci) - mean_r]],
                        color=_INK, capsize=5, lw=1.4, zorder=4)
        ax.plot([xi - 0.22, xi + 0.22], [mean_r, mean_r], color=_INK, lw=2.2, zorder=4)
        ax.text(xi, max(sub["r"].max(), mean_r) + 0.03, f"p={p:.3f}", ha="center",
                va="bottom", fontsize=9, color=_INK)

    ax.axhline(0.0, ls="--", lw=1.2, color=_GREY, zorder=1)
    ax.text(len(levels) - 0.55, 0.0, "no relationship", va="center", ha="left",
            fontsize=8, color="#64748b")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_xlim(-0.5, len(levels) - 0.1 + 0.4)
    ax.set_ylabel(f"{res.method.capitalize()} r:  {x_label}")
    ax.set_title(title, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(title="replicate", frameon=False, loc="lower right", fontsize=8,
              title_fontsize=8)
    fig.tight_layout()
    return _save(fig, out_stem)


def _replicate_col(res: CorrelationResult) -> str:
    # the per_replicate frame's first column is the replicate
    return res.per_replicate.columns[0]


def enrichment_corroboration(results: dict[str, EnrichmentResult], out_stem: Path, *,
                             type_order: list[str], type_labels: dict[str, str],
                             title: str, replicate_col: str = "experiment_id",
                             type_col: str = "type") -> list[Path]:
    """Per-replicate enrichment per type from two (or more) independent nulls, drawn
    side by side — overlapping markers show the result is robust to the null model."""
    method_colours = ["#3b6fb5", "#E8A33D", "#3DBC8A"]
    markers = ["o", "D", "^"]
    methods = list(results)
    offsets = np.linspace(-0.17, 0.17, len(methods))
    fig, ax = plt.subplots(figsize=(2.0 + 1.5 * len(type_order), 5.0))
    rng = np.random.default_rng(0)
    from scipy import stats as ss
    for xi, t in enumerate(type_order):
        for mi, name in enumerate(methods):
            df = results[name].per_replicate
            vals = df.loc[df[type_col] == t, "enrichment"].to_numpy()
            colour, marker, dx = method_colours[mi], markers[mi], offsets[mi]
            ax.scatter(xi + dx + rng.uniform(-0.04, 0.04, len(vals)), vals, s=30,
                       color=colour, alpha=0.45, zorder=2, linewidths=0)
            z = np.log2(vals)
            m = 2 ** z.mean()
            ci = (ss.t.ppf(0.975, len(z) - 1) * z.std(ddof=1) / np.sqrt(len(z))
                  if len(z) > 1 else 0.0)
            ax.errorbar(xi + dx, m,
                        yerr=[[m - 2 ** (z.mean() - ci)], [2 ** (z.mean() + ci) - m]],
                        color=colour, marker=marker, ms=9, capsize=4, lw=1.6,
                        mec="white", mew=0.6, zorder=4,
                        label=name if xi == 0 else None)
            _, p = results[name].by_type[t]
            ax.text(xi + dx, max(vals.max(), m) * 1.012, _stars(p), ha="center",
                    va="bottom", fontsize=10, color=colour)
    ax.axhline(1.0, ls="--", lw=1.2, color=_GREY, zorder=1)
    ax.text(len(type_order) - 0.5, 1.0, "chance", va="center", ha="left",
            fontsize=8, color="#64748b")
    ax.set_yscale("log", base=2)
    ax.set_yticks([0.9, 1.0, 1.1])
    ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(range(len(type_order)))
    ax.set_xticklabels([type_labels.get(t, t) for t in type_order])
    ax.set_xlim(-0.5, len(type_order) - 0.1 + 0.45)
    ax.set_ylabel("contact enrichment  (observed / null)")
    ax.set_title(title, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    return _save(fig, out_stem)


def enrichment_vs_chance(res: EnrichmentResult, out_stem: Path, *,
                         type_order: list[str], type_labels: dict[str, str],
                         title: str, replicate_col: str = "experiment_id",
                         type_col: str = "type") -> list[Path]:
    """Per-replicate contact enrichment per type, with the chance line at 1.0 and
    the one-sample p vs chance."""
    df = res.per_replicate
    reps = sorted(df[replicate_col].unique())
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(type_order), 5.0))
    rng = np.random.default_rng(0)
    for xi, t in enumerate(type_order):
        vals = df.loc[df[type_col] == t, "enrichment"]
        for rep, (colour, marker) in zip(reps, _REP_STYLE):
            v = df[(df[type_col] == t) & (df[replicate_col] == rep)]["enrichment"]
            if v.empty:
                continue
            ax.scatter(xi + rng.uniform(-0.1, 0.1), v.iloc[0], s=80, color=colour,
                       marker=marker, edgecolor="white", linewidth=0.6, zorder=3,
                       label=rep if xi == 0 else None)
        m = vals.mean()
        from scipy import stats as ss
        z = np.log2(vals.to_numpy())
        ci = (ss.t.ppf(0.975, len(z) - 1) * z.std(ddof=1) / np.sqrt(len(z))
              if len(z) > 1 else 0.0)
        ax.plot([xi - 0.22, xi + 0.22], [m, m], color=_INK, lw=2.2, zorder=4)
        ax.errorbar(xi, 2 ** z.mean(),
                    yerr=[[2 ** z.mean() - 2 ** (z.mean() - ci)],
                          [2 ** (z.mean() + ci) - 2 ** z.mean()]],
                    color=_INK, capsize=5, lw=1.4, zorder=4)
        _, p = res.by_type[t]
        ax.text(xi, max(vals.max(), m) * 1.015, f"{_stars(p)}\np={p:.3f}",
                ha="center", va="bottom", fontsize=9, color=_INK)

    ax.axhline(1.0, ls="--", lw=1.2, color=_GREY, zorder=1)
    ax.text(len(type_order) - 0.5, 1.0, "chance", va="center", ha="left",
            fontsize=8, color="#64748b")
    ax.set_yscale("log", base=2)
    ax.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(range(len(type_order)))
    ax.set_xticklabels([type_labels.get(t, t) for t in type_order])
    ax.set_xlim(-0.5, len(type_order) - 0.1 + 0.45)
    ax.set_ylabel("contact enrichment  (observed / null)")
    ax.set_title(title, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(title="replicate", frameon=False, loc="upper right", fontsize=8,
              title_fontsize=8)
    fig.tight_layout()
    return _save(fig, out_stem)
