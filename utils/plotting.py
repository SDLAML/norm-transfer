from __future__ import annotations
from typing import Iterable, Sequence, Mapping, Any
from itertools import cycle

import math
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import plotly.graph_objects as go
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, FixedLocator, FixedFormatter, NullFormatter
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import FancyArrowPatch

from utils.fitting import (
    Config,
    compute_optimum,
    select_avg_window_at_horizon
)

# ---------------------------------------------------------------------------
# Shared legend handle and helper for batch-size min→max indicator
# ---------------------------------------------------------------------------
class _MinMaxHandle:
    """Legend helper storing the marker scale for batch-size extremes."""

    def __init__(self, smin, smax, color="0.5", hollow=True):
        """Capture marker style information for the batch-size legend entry."""
        self.smin, self.smax = float(smin), float(smax)
        self.edge = color
        self.face = "none" if hollow else color


class _MinMaxHorizHandler(HandlerBase):
    """Matplotlib legend handler that renders min→max markers with an arrow."""

    def create_artists(self, legend, orig, xdescent, ydescent, width, height, fs, trans):
        """Return artists visualising the minimum and maximum marker sizes."""
        y   = ydescent + 0.5*height
        xL  = xdescent + 0.85*width
        xR  = xdescent + 0.85*width

        msL = np.sqrt(orig.smin)
        msR = np.sqrt(orig.smax) * 0.8  # shrink right marker a bit

        left  = Line2D([xL],[y], marker="o", linestyle="None", markersize=msL,
                        markerfacecolor=orig.face, markeredgecolor='black',
                        transform=trans)
        right = Line2D([xR],[y], marker="o", linestyle="None", markersize=msR,
                        markerfacecolor=orig.face, markeredgecolor='black',
                        transform=trans)

        # arrow from left → right (with small offsets so it doesn't overlap circles)
        arr = FancyArrowPatch((xL + 0.6*msL, y), (xR + 0.6*msR, y),
                              arrowstyle="->", mutation_scale=fs, lw=1,
                              color='black', transform=trans)
        return [left, arr, right]


def add_bs_minmax_legend(
    ax: plt.Axes,
    bs_values: Iterable[int],
    bs_size: dict[int, float],
    *,
    title: str = "Batch size [samples]\n",
    bbox_to_anchor=(1.0, 0.65),
    loc: str = "upper left",
    legend_fontsize: float | int | None = None,
    title_fontsize: float | int | None = None,
):
    """Add a legend entry visualising min→max batch size marker scale.

    Parameters
    - ax: target axis
    - bs_values: iterable of batch sizes present in the plot
    - bs_size: mapping bs -> scatter size
    - title, bbox_to_anchor, loc: legend layout controls
    """
    bs_vals = np.sort(np.array(list(bs_values), dtype=int))
    if bs_vals.size == 0:
        return None
    bs_min, bs_max = int(bs_vals.min()), int(bs_vals.max())
    handle = _MinMaxHandle(smin=bs_size[bs_min], smax=bs_size[bs_max], color="0.5", hollow=True)
    leg = ax.legend(
        [handle], [f"{bs_min} → {bs_max}"],
        handler_map={_MinMaxHandle: _MinMaxHorizHandler()},
        title=title,
        bbox_to_anchor=bbox_to_anchor, loc=loc,
        borderpad=1.,
        handletextpad=1.8,
        prop=None if legend_fontsize is None else {"size": legend_fontsize},
        title_fontsize=title_fontsize if title_fontsize is not None else legend_fontsize,
    )
    ax.add_artist(leg)
    return leg

def build_bs_sizes(batch_sizes: Iterable[int], base: int, factor: float) -> dict[int, float]:
    """Map batch sizes to scatter marker areas via geometric scaling."""
    bs_sorted = sorted(batch_sizes)
    return {bs: base * (factor ** i) for i, bs in enumerate(bs_sorted)}


def make_horizon_style_maps(horizons: Sequence[int], markers: Sequence[str]):
    """Return marker and colour lookup dictionaries keyed by horizon."""
    if len(horizons) > len(markers):
        raise ValueError("Need more marker shapes for horizons.")
    palette = cm.get_cmap("viridis", len(horizons))
    h_sorted = sorted(horizons)
    h_to_marker = {h: markers[i] for i, h in enumerate(h_sorted)}
    h_to_color = {h: palette(i) for i, h in enumerate(h_sorted)}
    return h_to_marker, h_to_color


def nice_horizon_label(h: int) -> str:
    """Format a horizon as ``2^k`` plus its token count in billions."""
    exp = int(math.log2(h))
    billions = h / 1e9
    billions_str = f"{billions:.1f}" if billions < 10 else f"{int(round(billions))}"
    return rf'$2^{{{exp}}}$ ({billions_str} B)'

def plot_minima(minima_df: pd.DataFrame, cfg: Config):
    """Plot per-horizon minima as lines (over bs) + scatter with marker size by bs.

    Cosmetics (line width, alpha, legend positions, font sizes, layout) are
    read from ``cfg`` to match the configurability of
    ``plot_minima_at_horizon_across_models``.
    """
    if minima_df.empty:
        raise ValueError("minima_df is empty — nothing to plot.")

    # Pull plotting-related settings from cfg (shared with the model-comparison view)
    default_linestyle: str = cfg.default_linestyle
    default_alpha: float = cfg.default_alpha
    show_legends: bool = cfg.show_legends
    legend_h_loc: str = cfg.legend_models_loc
    legend_h_bbox: tuple[float, float] | None = cfg.legend_models_bbox
    legend_bs_loc: str = cfg.legend_bs_loc
    legend_bs_bbox: tuple[float, float] | None = cfg.legend_bs_bbox
    use_constrained_layout: bool = cfg.use_constrained_layout
    tight_layout_right: float = cfg.tight_layout_right
    line_width: float = cfg.line_width
    legend_fontsize: float | int | None = cfg.legend_fontsize
    axis_label_fontsize: float | int | None = cfg.axis_label_fontsize
    tick_label_fontsize: float | int | None = cfg.tick_label_fontsize

    unique_h = sorted(minima_df["horizon"].unique())
    unique_bs = sorted(minima_df["bs"].unique())

    h_marker, h_color = make_horizon_style_maps(unique_h, cfg.markers)
    bs_size = build_bs_sizes(unique_bs, cfg.bs_size_base, cfg.bs_size_factor)

    fig, ax = plt.subplots(figsize=cfg.figsize, constrained_layout=use_constrained_layout)
    bg = ax.get_facecolor()

    min_xs: list[float] = []
    for h in unique_h:
        sub = minima_df[minima_df["horizon"] == h].sort_values("log2_output_norm")
        sub_for_line = sub.sort_values("bs")

        # line through minima across batch sizes (cosmetic trend)
        ax.plot(
            sub_for_line["log2_output_norm"],
            sub_for_line["train_loss"],
            color=h_color[h],
            lw=float(line_width),
            linestyle=default_linestyle,
            alpha=default_alpha,
            zorder=1,
        )

        # scatter per point
        for _, row in sub.iterrows():
            ax.scatter(
                float(row["log2_output_norm"]),
                float(row["train_loss"]),
                s=bs_size[int(row["bs"])],
                marker=h_marker[h],
                facecolors=bg,
                edgecolors=h_color[h],
                linewidths=float(line_width),
                alpha=default_alpha,
                zorder=2,
            )

        # highlight best train_loss at this horizon
        best = sub.loc[sub["train_loss"].idxmin()]
        try:
            min_xs.append(float(best["log2_output_norm"]))
        except Exception:
            pass
        ax.scatter(
            float(best["log2_output_norm"]),
            float(best["train_loss"]),
            s=bs_size[int(best["bs"])],
            marker=h_marker[h],
            color=h_color[h],
            edgecolors="none",
            alpha=default_alpha,
            zorder=3,
        )

    # mean ± 1σ band of optimal log2 norms
    if min_xs:
        x_mean = float(np.mean(min_xs))
        x_std = float(np.std(min_xs))
        ax.axvspan(x_mean - x_std, x_mean + x_std, color="grey", alpha=0.2)
        ax.axvline(x=x_mean, linestyle="--", color="grey", linewidth=1)
    else:
        x_mean, x_std = float("nan"), float("nan")

    # legends
    if show_legends:
        # Horizon legend: show color + marker (and a short line) for each horizon
        handles_h: list[Line2D] = []
        labels_h: list[str] = []
        # choose a representative marker size (points) derived from bs_size_base
        try:
            ms_legend = float(np.sqrt(cfg.bs_size_base)) * 1.2
        except Exception:
            ms_legend = 8.0
        for h in unique_h:
            handle = Line2D(
                [0], [0],
                color=h_color[h],
                linestyle=default_linestyle,
                linewidth=float(line_width),
                marker=h_marker[h],
                markersize=ms_legend,
                markerfacecolor=bg,
                markeredgecolor=h_color[h],
                markeredgewidth=float(line_width),
            )
            handles_h.append(handle)
            labels_h.append(nice_horizon_label(int(h)))

        leg_h = ax.legend(
            handles=handles_h,
            labels=labels_h,
            title="Horizon [tokens]",
            loc=legend_h_loc if legend_h_bbox is None else "upper left",
            bbox_to_anchor=legend_h_bbox,
            prop=None if legend_fontsize is None else {"size": legend_fontsize},
            title_fontsize=legend_fontsize if legend_fontsize is not None else None,
            ncol=2
        )
        ax.add_artist(leg_h)

        # Batch size (min→max) legend via shared helper
        add_bs_minmax_legend(
            ax,
            bs_values=unique_bs,
            bs_size=bs_size,
            bbox_to_anchor=legend_bs_bbox,
            loc=legend_bs_loc,
            legend_fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
        )

        # # Small legend for mean ± 1σ band
        # if min_xs:
        #     fit_handles = [
        #         Line2D([0], [0], linestyle="--", color="grey", linewidth=1),
        #         Patch(facecolor="grey", edgecolor="none", alpha=0.2),
        #     ]
        #     fit_labels = [f"mean = {x_mean:.2f}", f"±1σ = {x_std:.2f}"]
        #     ax.legend(
        #         handles=fit_handles,
        #         labels=fit_labels,
        #         title="log₂(Optimal norm)",
        #         bbox_to_anchor=(1.02, 0.2),
        #         loc="upper left",
        #         prop=None if legend_fontsize is None else {"size": legend_fontsize},
        #         title_fontsize=legend_fontsize if legend_fontsize is not None else None,
        #     )

    # Axes, labels, title
    ax.set(
        xlabel=r"log₂||$W_\mathrm{out}$||$_{\mathrm{RMS} \to \infty}$",
        ylabel="Train loss",
    )
    ax.set_title(
        "Scaling horizon for proxy model (69M)", # \n(colour & marker ≙ horizon; size ≙ batch size)
        fontsize=axis_label_fontsize,
    )

    # Apply optional font sizes for axis labels and tick labels
    if axis_label_fontsize is not None:
        try:
            ax.xaxis.label.set_size(axis_label_fontsize)
            ax.yaxis.label.set_size(axis_label_fontsize)
        except Exception:
            pass
    if tick_label_fontsize is not None:
        try:
            ax.tick_params(axis='both', which='both', labelsize=tick_label_fontsize)
        except Exception:
            pass

    ax.grid(True, linestyle="--", linewidth=0.3)
    if not use_constrained_layout:
        fig.tight_layout(rect=[0, 0, float(tight_layout_right), 1])
    return fig, ax


def plot_minima_at_horizon_across_models(
    minima_by_model: dict[str, pd.DataFrame],
    cfg: Config,
    horizon: int,
):
    """Overlay per-batch minima for a single horizon across multiple models.

    Parameters
    ----------
    minima_by_model : dict[str, pandas.DataFrame]
        Mapping of model label -> minima_df as produced by ``build_minima_df``.
        Each dataframe must contain columns: ``horizon``, ``bs``,
        ``log2_output_norm``, and ``train_loss``.
    cfg : Config
        Plot configuration. All plotting-related knobs are read from ``cfg``
        (e.g., model styles, legend positions, line widths, font sizes, etc.).
    horizon : int
        The fixed horizon to display.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axis with the plot.
    """
    if not minima_by_model:
        raise ValueError("minima_by_model is empty — nothing to plot.")

    # Pull plotting-related settings from cfg
    model_styles: Mapping[str, Mapping[str, Any]] | None = cfg.model_styles
    default_marker: str = cfg.default_marker
    default_linestyle: str = cfg.default_linestyle
    default_alpha: float = cfg.default_alpha
    show_legends: bool = cfg.show_legends
    legend_models_loc: str = cfg.legend_models_loc  # kept for compatibility, not heavily used
    legend_models_bbox: tuple[float, float] | None = cfg.legend_models_bbox
    legend_bs_loc: str = cfg.legend_bs_loc
    legend_bs_bbox: tuple[float, float] | None = cfg.legend_bs_bbox
    use_constrained_layout: bool = cfg.use_constrained_layout
    tight_layout_right: float = cfg.tight_layout_right
    line_width: float = cfg.line_width
    legend_fontsize: float | int | None = cfg.legend_fontsize
    axis_label_fontsize: float | int | None = cfg.axis_label_fontsize
    tick_label_fontsize: float | int | None = cfg.tick_label_fontsize

    # Filter and validate
    filtered: dict[str, pd.DataFrame] = {}
    all_bs: set[int] = set()
    for label, df in minima_by_model.items():
        if df is None or df.empty:
            continue
        need = {"horizon", "bs", "log2_output_norm", "train_loss"}
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"minima_df for '{label}' missing columns: {miss}")
        sub = df[df["horizon"] == int(horizon)].copy()
        if sub.empty:
            continue
        sub["bs"] = sub["bs"].astype(int)
        sub = sub.sort_values(["bs", "log2_output_norm"])  # stable ordering
        filtered[label] = sub
        all_bs.update(map(int, sub["bs"].unique()))

    if not filtered:
        raise ValueError("No minima rows for the requested horizon across provided models.")

    bs_sorted = sorted(all_bs)
    bs_size = build_bs_sizes(bs_sorted, cfg.bs_size_base, cfg.bs_size_factor)

    # Colors/styles per model
    labels = list(filtered.keys())
    cmap = plt.get_cmap("tab10")
    base_colors = {lab: cmap(i % 10) for i, lab in enumerate(labels)}

    def _style_for(label: str):
        s = model_styles.get(label, {}) if model_styles else {}
        color = s.get("color", base_colors[label])
        marker = s.get("marker", default_marker)
        linestyle = s.get("linestyle", default_linestyle)
        try:
            alpha = float(s.get("alpha", default_alpha))
        except (TypeError, ValueError):
            alpha = default_alpha
        if not (0.0 <= alpha <= 1.0):
            alpha = max(0.0, min(1.0, alpha))
        legend_label = (
            s.get("legend_label")
            or s.get("legend")
            or s.get("label")
            or label
        )
        return color, marker, linestyle, alpha, legend_label

    fig, ax = plt.subplots(figsize=cfg.figsize, constrained_layout=use_constrained_layout)
    bg = ax.get_facecolor()

    # Draw each model: line over bs + scatter points
    best_xs: list[float] = []  # collect per‑model optimal log2 norms
    for label in labels:
        sub = filtered[label]
        color, marker, linestyle, alpha, legend_label = _style_for(label)
        # Ensure increasing bs for the trend line
        line_df = sub.sort_values("bs")
        ax.plot(
            line_df["log2_output_norm"],
            line_df["train_loss"],
            color=color,
            lw=float(line_width),
            linestyle=linestyle,
            alpha=alpha,
            zorder=1,
            label=legend_label,
        )

        for _, row in sub.iterrows():
            ax.scatter(
                float(row["log2_output_norm"]),
                float(row["train_loss"]),
                s=bs_size[int(row["bs"])],
                marker=marker,
                facecolors=bg,
                edgecolors=color,
                linewidths=float(line_width),
                alpha=alpha,
                zorder=2,
            )

        # Highlight best train_loss for this model at this horizon
        i_best = sub["train_loss"].idxmin()
        best = sub.loc[i_best]
        try:
            best_xs.append(float(best["log2_output_norm"]))
        except Exception:
            pass
        ax.scatter(
            float(best["log2_output_norm"]),
            float(best["train_loss"]),
            s=bs_size[int(best["bs"])],
            marker=marker,
            color=color,
            edgecolors="none",
            alpha=alpha,
            zorder=3,
        )

    # mean ± 1σ band of optimal log2 norms across models (at this horizon)
    if best_xs:
        x_mean = float(np.mean(best_xs))
        x_std = float(np.std(best_xs))
        ax.axvspan(x_mean - x_std, x_mean + x_std, color="grey", alpha=0.2)
        ax.axvline(x=x_mean, linestyle="--", color="grey", linewidth=1)

    # Legends: models and batch-size scale
    if show_legends:
        # Build custom handles so legend shows both line style and marker shape
        handles_models: list[Line2D] = []
        labels_models: list[str] = []
        # choose a representative marker size for legend (points, not area)
        try:
            ms_legend = float(np.sqrt(cfg.bs_size_base)) * 1.2
        except Exception:
            ms_legend = 8.0
        for label in labels:
            color, marker, linestyle, alpha, legend_label = _style_for(label)
            handle = Line2D(
                [0], [0],
                color=color,
                linestyle=linestyle,
                linewidth=float(line_width),
                alpha=alpha,
                marker=marker,
                markersize=ms_legend,
                markerfacecolor=bg,
                markeredgecolor=color,
                markeredgewidth=float(line_width),
            )
            handles_models.append(handle)
            labels_models.append(legend_label)

        # Custom legend layout:
        #  - first model on a centered first line
        #  - remaining models in two columns below (split into two halves)
        if legend_models_bbox is None:
            # Place two stacked legends INSIDE the axes, anchored to the top center.
            # Top: first model centered with the title
            top_anchor = (0.5, 1.0)  # axes coordinates
            leg_top = ax.legend(
                handles=handles_models[:1],
                labels=labels_models[:1],
                title=None,
                loc="upper center",
                bbox_to_anchor=top_anchor,
                prop=None if legend_fontsize is None else {"size": legend_fontsize},
                title_fontsize=legend_fontsize if legend_fontsize is not None else None,
            )
            ax.add_artist(leg_top)

            # Bottom grid for remaining models, two columns, just below the first row
            rest_handles = handles_models[1:]
            rest_labels  = labels_models[1:]
            if rest_handles:
                n = len(rest_handles)
                half = int(np.ceil(n / 2))
                left_h,  right_h  = rest_handles[:half], rest_handles[half:]
                left_lbl, right_lbl = rest_labels[:half],  rest_labels[half:]

                grid_handles: list[Line2D] = []
                grid_labels:  list[str]   = []
                for i in range(half):
                    if i < len(left_h):
                        grid_handles.append(left_h[i]); grid_labels.append(left_lbl[i])
                    if i < len(right_h):
                        grid_handles.append(right_h[i]); grid_labels.append(right_lbl[i])

                leg_bottom = ax.legend(
                    handles=grid_handles,
                    labels=grid_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.92),  # slightly below the top legend
                    ncol=2,
                    columnspacing=1.0,
                    prop=None if legend_fontsize is None else {"size": legend_fontsize},
                )
                ax.add_artist(leg_bottom)
        else:
            # Place two legends in the right margin.
            # Top: single centered entry (first model) with the title
            bx, by = legend_models_bbox

            # Heuristic: center within the right margin (~1/6 of axes width to the right
            # of the axes' right edge). This pairs well with tight_layout_right≈0.75.
            top_center_x = bx + 0.15

            # Top legend with the first model only
            leg_top = ax.legend(
                handles=handles_models[:1],
                labels=labels_models[:1],
                title=None,
                loc="upper center",
                bbox_to_anchor=(top_center_x - 0.227, by+0.01),
                prop=None if legend_fontsize is None else {"size": legend_fontsize},
                title_fontsize=legend_fontsize if legend_fontsize is not None else None,
            )
            ax.add_artist(leg_top)

            # Bottom legend: remaining models split into two columns
            rest_handles = handles_models[1:]
            rest_labels  = labels_models[1:]
            if rest_handles:
                n = len(rest_handles)
                half = int(np.ceil(n / 2))
                left_h,  right_h  = rest_handles[:half], rest_handles[half:]
                left_lbl, right_lbl = rest_labels[:half],  rest_labels[half:]

                grid_handles: list[Line2D] = []
                grid_labels:  list[str]   = []
                for i in range(half):
                    if i < len(left_h):
                        grid_handles.append(left_h[i]); grid_labels.append(left_lbl[i])
                    if i < len(right_h):
                        grid_handles.append(right_h[i]); grid_labels.append(right_lbl[i])

                leg_bottom = ax.legend(
                    handles=grid_handles,
                    labels=grid_labels,
                    loc="upper center",
                    bbox_to_anchor=(top_center_x, by - 0.07),  # a bit below the first row
                    ncol=2,
                    columnspacing=1.0,
                    prop=None if legend_fontsize is None else {"size": legend_fontsize},
                )
                ax.add_artist(leg_bottom)

        add_bs_minmax_legend(
            ax,
            bs_values=bs_sorted,
            bs_size=bs_size,
            bbox_to_anchor=legend_bs_bbox,
            loc=legend_bs_loc,
            legend_fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
        )

        # # Add small legend for the mean ± 1σ band
        # if best_xs:
        #     fit_handles = [
        #         Line2D([0], [0], linestyle="--", color="grey", linewidth=1),
        #         Patch(facecolor="grey", edgecolor="none", alpha=0.2),
        #     ]
        #     x_mean = float(np.mean(best_xs))
        #     x_std = float(np.std(best_xs))
        #     fit_labels = [f"mean = {x_mean:.2f}", f"±1σ = {x_std:.2f}"]
        #     ax.legend(
        #         handles=fit_handles,
        #         labels=fit_labels,
        #         title="log₂(Optimal norm)",
        #         bbox_to_anchor=(1.02, 0.2),
        #         loc="upper left",
        #         prop=None if legend_fontsize is None else {"size": legend_fontsize},
        #         title_fontsize=legend_fontsize if legend_fontsize is not None else None,
        #     )

    # Axes, labels, title
    ax.set(
        xlabel=r"log₂||$W_\mathrm{out}$||$_{\mathrm{RMS} \to \infty}$",
        ylabel="Train loss",
    )
    ax.set_title(f"Scaling width/depth for {nice_horizon_label(int(horizon))} token horizon", fontsize=axis_label_fontsize) # , colour/marker ≙ model; size ≙ batch size

    # Apply optional font sizes for axis labels and tick labels
    if axis_label_fontsize is not None:
        try:
            ax.xaxis.label.set_size(axis_label_fontsize)
            ax.yaxis.label.set_size(axis_label_fontsize)
        except Exception:
            pass
    if tick_label_fontsize is not None:
        try:
            ax.tick_params(axis='both', which='both', labelsize=tick_label_fontsize)
        except Exception:
            pass
    ax.grid(True, linestyle="--", linewidth=0.3)
    if not use_constrained_layout:
        fig.tight_layout(rect=[0, 0, float(tight_layout_right), 1])
    return fig, ax


def plot_parabola_grid(raw_df: pd.DataFrame, cfg: Config):
    """Plot quadratic fits per horizon/batch-size pair across a grid of axes."""

    horizons_sorted = sorted(cfg.horizons)

    work = raw_df.copy()
    work["output_norm"] = work["output_norm"].astype(float)
    work["train_loss"]  = work["train_loss"].astype(float)
    work["lr"]          = work["lr"].astype(float)

    unique_bs = sorted(work["bs"].unique())
    rows = len(horizons_sorted)
    cols = len(unique_bs)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2), squeeze=False)

    palette = cm.get_cmap("viridis", len(unique_bs))
    bs_to_color = {bs: palette(i) for i, bs in enumerate(unique_bs)}

    skip_pairs = {(int(hh), int(bb)) for hh, bb in cfg.skip_fit}

    for r, h in enumerate(horizons_sorted):
        # Representative points for this horizon (already averaged + stds)
        row_points = (
            work.sort_values("step")
            .groupby(["bs", "lr"], as_index=False)
            .apply(lambda g, h=h: select_avg_window_at_horizon(g, h, cfg=cfg))
            .dropna(how="any")
        )

        # ---- Compute a common x-limit for the entire row (include ±1σ if present) ----
        if not row_points.empty:
            if {"log2_output_norm", "log2_output_norm_std"}.issubset(row_points.columns):
                mu  = row_points["log2_output_norm"].astype(float).to_numpy()
                sig = row_points["log2_output_norm_std"].astype(float).fillna(0.0).to_numpy()
                x_lo = np.power(2.0, mu - sig)
                x_hi = np.power(2.0, mu + sig)
            else:
                x_lin = row_points["output_norm"].astype(float).to_numpy()
                x_lo = x_lin
                x_hi = x_lin

            xmin = float(np.nanmin(x_lo))
            xmax = float(np.nanmax(x_hi))

            # small padding in log2 space (±0.15 ≈ ±10%)
            pad_log2 = 0.15
            xmin /= 2**pad_log2
            xmax *= 2**pad_log2

            # avoid degenerate/invalid limits on log scale
            if not np.isfinite(xmin) or xmin <= 0:
                xmin = max(1e-6, np.nanmin(row_points["output_norm"]))
            if not np.isfinite(xmax):
                xmax = np.nanmax(row_points["output_norm"]) * 1.1
        else:
            xmin, xmax = 1.0, 2.0  # harmless fallback

        # -------------------------------------------------------------------------

        for c, bs in enumerate(unique_bs):
            ax = axes[r][c]
            ax.set_xscale("log", base=2)
            ax.set_xlim(xmin, xmax)  # unify the row’s x-range

            sub = row_points[row_points["bs"] == bs]
            sub = sub[sub["train_loss"] < cfg.max_loss]

            # scatter + error bars
            if not sub.empty:
                ax.scatter(
                    sub["output_norm"], sub["train_loss"],
                    color=bs_to_color[bs], s=12, marker="x"
                )

                has_xstd = {"log2_output_norm", "log2_output_norm_std", "output_norm"}.issubset(sub.columns)
                has_ystd = "train_loss_std" in sub.columns

                if has_xstd or has_ystd:
                    xvals = sub["output_norm"].to_numpy(dtype=float)
                    xerr = None
                    if has_xstd:
                        mu   = sub["log2_output_norm"].to_numpy(dtype=float)
                        sig  = sub["log2_output_norm_std"].astype(float).fillna(0.0).to_numpy()
                        x_lo = np.power(2.0, mu - sig)
                        x_hi = np.power(2.0, mu + sig)
                        xerr = np.vstack([
                            np.clip(xvals - x_lo, 0.0, None),
                            np.clip(x_hi - xvals, 0.0, None),
                        ])
                    yerr = sub["train_loss_std"].astype(float).fillna(0.0).to_numpy() if has_ystd else None

                    ax.errorbar(
                        sub["output_norm"], sub["train_loss"],
                        xerr=xerr, yerr=yerr,
                        fmt="none", elinewidth=0.8, alpha=0.6, capsize=2, color=bs_to_color[bs]
                    )

            # handle small n (no quadratic fit)
            if len(sub) < 3:
                if not sub.empty:
                    idx_min = sub["train_loss"].idxmin()
                    x0_lin = float(sub.loc[idx_min, "output_norm"])
                    y0     = float(sub.loc[idx_min, "train_loss"])
                    x0_log2 = float(sub.loc[idx_min, "log2_output_norm"]) if "log2_output_norm" in sub.columns else float(np.log2(x0_lin))
                    ax.axvline(x0_lin, lw=0.8, ls="--")
                    ax.axhline(y0,     lw=0.8, ls="--")
                    min_handle = Line2D([], [], ls="--", lw=0.8, label=f"min: log2 x≈{x0_log2:.3g}, y≈{y0:.3g}")
                    ax.legend(handles=[min_handle], fontsize=6, loc="best")

                ax.grid(True, linestyle="--", linewidth=0.3)
                if r == 0:
                    ax.set_title(f"bs={bs}", fontsize=8)
                if c == 0:
                    ax.set_ylabel(f"{nice_horizon_label(h)}", fontsize=8)
                ax.tick_params(labelsize=7)
                continue

            # ---- FIT (weighted if stds are present) ----
            x = sub["log2_output_norm"].to_numpy(dtype=float)
            y = sub["train_loss"].to_numpy(dtype=float)

            y_std = sub["train_loss_std"].astype(float).fillna(0.0).to_numpy()       if "train_loss_std"       in sub.columns else None

            # If all stds are exactly zero, treat as None to avoid no-op weighting
            if y_std is not None and not np.any(np.isfinite(y_std) & (y_std > 0)):
                y_std = None

            do_fit = cfg.from_fit and (h, int(bs)) not in skip_pairs
            fit = compute_optimum(
                x, y,
                from_fit=do_fit,
                c_fixed=cfg.c_fixed,
                optimum_from_closest=cfg.optimum_from_closest,
                fit_k=cfg.fit_k, fit_k_by=cfg.fit_k_by,
                y_std=y_std,
            )

            # plot the fitted parabola if we actually did a quadratic fit
            if fit is not None and fit.a is not None:
                x_fit = np.linspace(float(x.min()), float(x.max()), 200)
                y_fit = fit.a * x_fit**2 + fit.b * x_fit + fit.c
                # annotate whether weighting was used
                used_wls = y_std is not None
                label = (f"a={fit.a:.3f}, b={fit.b:.3f}, c={fit.c:.3f}"
                         + (" (WLS)" if used_wls else ""))
                ax.plot(2**x_fit, y_fit, lw=1.0, label=label)

            # draw optimum lines (from fitted or empirical)
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            if fit is not None:
                x0_log2 = float(fit.x0)
                x0_lin  = float(2**x0_log2)
                y0      = float(fit.y0)
                ax.axvline(x0_lin, lw=0.8, ls="--")
                ax.axhline(y0,     lw=0.8, ls="--")
                min_handle = Line2D([], [], ls="--", lw=0.8, label=f"min: log2 x≈{x0_log2:.3g}, y≈{y0:.3g}")
                legend_handles.append(min_handle); legend_labels.append(min_handle.get_label())

            # cosmetics
            if r == 0:
                ax.set_title(f"bs={bs}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{nice_horizon_label(h)}", fontsize=8)
            ax.grid(True, linestyle="--", linewidth=0.3)
            ax.tick_params(labelsize=7)
            if legend_handles:
                ax.legend(handles=legend_handles, labels=legend_labels, fontsize=6, loc="best")

    fig.tight_layout()
    return fig, axes

def plot_lr_bs_fit(
    df,
    *,
    min_alpha: float = 0.2,
    lr_col: str = "lr",
    bs_col: str = "bs",
    horizon_col: str = "horizon",
    loss_col: str = "train_loss",
    figsize=(8, 6),
    ax=None,
    marker_size: float = 80,
    best_marker_size: float = 350,
    legend_marker_size: float | None = None,
    legend_fontsize: float = 9.0,
    axis_label_fontsize: float | None = None,
    tick_label_fontsize: float | None = None,
    errorbar_linewidth: float = 1.0,
    errorbar_capsize: float = 3.0,
):
    """
    Fit log2(lr) ~ alpha*log2(bs) + beta*log2(horizon) + C and
    plot per-horizon data & regression lines.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing columns for learning rate, batch size, and horizon.
    min_alpha : float, optional
        Minimum transparency for the worst loss (if loss_col exists). Default 0.2.
    lr_col, bs_col, horizon_col, loss_col : str
        Column names in `df`. `loss_col` is optional.
    figsize : tuple, optional
        Figure size when creating a new axis. Ignored if `ax` is provided.
    ax : matplotlib.axes.Axes or None
        If provided, draw on this axis; otherwise create a new one.
    marker_size : float, optional
        Marker area for data points when not using error bars (matplotlib scatter `s`).
    best_marker_size : float, optional
        Marker area for the highlighted "best" point.
    legend_marker_size : float or None, optional
        Marker area for legend entries. If None, uses `marker_size`.
    legend_fontsize : float, optional
        Font size for the legends (model lines, batch sizes, fit summary). Default ``9.0``.
    axis_label_fontsize : float or None, optional
        Font size for x/y axis labels. ``None`` uses Matplotlib defaults.
    tick_label_fontsize : float or None, optional
        Font size for tick labels on both axes. ``None`` uses Matplotlib defaults.
    errorbar_linewidth : float, optional
        Line width of error bars when `lr_std` is present.
    errorbar_capsize : float, optional
        Cap size of error bars when `lr_std` is present.

    Returns
    -------
    result : dict
        {
          "fig": fig,
          "ax": ax,
          "model": model,
          "slope": slope,
          "slope_se": slope_se,
          "intercepts": {h: (intercept, intercept_se), ...},
          "horizons": [sorted unique horizons]
        }
    """

    # ---------- Prep ----------
    if df is None:
        raise ValueError("df must be a pandas DataFrame")

    needed = {lr_col, bs_col, horizon_col}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    # keep only positive lr, bs
    d = df[(df[lr_col] > 0) & (df[bs_col] > 0)].copy()
    if d.empty:
        raise ValueError("No rows with positive learning rate and batch size.")

    # model frame with canonical names to keep the formula simple
    mf = pd.DataFrame({
        "lr": d[lr_col].values,
        "bs": d[bs_col].values,
        "h":  d[horizon_col].values,
    })
    if loss_col in d.columns:
        mf["loss"] = d[loss_col].values
    # carry std for plotting error bars if available
    if "lr_std" in d.columns:
        mf["lr_std"] = d["lr_std"].values

    mf["x"] = np.log2(mf["bs"])
    mf["y"] = np.log2(mf["lr"])
    # continuous horizon regressor
    mf["z"] = np.log2(mf["h"].astype(float))

    # ---------- Fit ----------
    # log2(lr) ~ alpha*log2(bs) + beta*log2(h) + C
    model = smf.ols("y ~ x + z", data=mf).fit()
    slope, slope_se = model.params["x"], model.bse["x"]

    # ---------- Intercepts ----------
    intercepts = {}
    base = float(model.params["Intercept"])  # C
    beta = float(model.params["z"])          # beta for log2(h)
    cov = model.cov_params()
    horizons = sorted(mf["h"].unique())

    # For fixed horizon h, the line is: log2(lr) = (C + beta*log2(h)) + alpha*log2(bs)
    # Propagate uncertainty for the intercept term: var(C + beta*z) = var(C) + z^2 var(beta) + 2 z cov(C,beta)
    var_C = float(cov.loc["Intercept", "Intercept"]) if "Intercept" in cov.index else 0.0
    var_beta = float(cov.loc["z", "z"]) if "z" in cov.index else 0.0
    cov_C_beta = float(cov.loc["Intercept", "z"]) if ("Intercept" in cov.index and "z" in cov.index) else 0.0

    for h in horizons:
        z_val = float(np.log2(float(h)))
        inter_h = base + beta * z_val
        var_inter_h = var_C + (z_val ** 2) * var_beta + 2.0 * z_val * cov_C_beta
        se_inter_h = float(np.sqrt(max(var_inter_h, 0.0)))
        intercepts[h] = (inter_h, se_inter_h)

    # ---------- Plot ----------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    n_h = len(horizons)
    colors = plt.cm.viridis(np.linspace(0., 1., n_h))
    marker_pool = ["o", "s", "D", "^", "v", "P", "*", "X", "<", ">"]
    markers = [marker_pool[i % len(marker_pool)] for i in range(n_h)]
    log2_offsets = np.linspace(-0.08, 0.08, n_h)

    for color, marker, off, h in zip(colors, markers, log2_offsets, horizons):
        exp = int(np.log2(h))  # assumes horizons are powers of two
        sub = mf[mf["h"] == h].copy()
        bs_shift = sub["bs"] * (2 ** off)

        # ---------- Alpha scaling ----------
        if "loss" in sub.columns:
            min_loss = sub["loss"].min()
            max_loss = sub["loss"].max()
            if max_loss == min_loss:
                alphas = np.ones(len(sub))
            else:
                alphas = 1 - (sub["loss"] - min_loss) / (max_loss - min_loss) * (1 - min_alpha)
        else:
            alphas = np.ones(len(sub))

        use_errorbar = ("lr_std" in sub.columns)
        if use_errorbar:
            for x_val, y_val, y_err, alpha_val in zip(bs_shift, sub["lr"], sub["lr_std"], alphas):
                ax.errorbar(
                    x_val, y_val,
                    yerr=y_err,
                    fmt=marker,
                    ecolor=color,
                    elinewidth=float(errorbar_linewidth),
                    capsize=float(errorbar_capsize),
                    alpha=float(alpha_val),
                    markersize=marker_size,
                    markerfacecolor=color,
                    markeredgecolor=color,
                )
        else:
            for x_val, y_val, alpha_val in zip(bs_shift, sub["lr"], alphas):
                ax.scatter(
                    x_val, y_val, marker=marker, s=float(marker_size),
                    edgecolors="black", linewidths=0.6,
                    color=color, alpha=float(alpha_val),
                )

        # best point
        if "loss" in sub.columns:
            best_idx = sub["loss"].idxmin()
            best_x = bs_shift.loc[best_idx]
            best_y = sub.loc[best_idx, "lr"]
            ax.scatter(best_x, best_y, s=float(best_marker_size), facecolors="none",
                       edgecolors=color, linewidths=2.5, marker="o")

        # regression line
        xs = np.linspace(sub["bs"].min() * 0.9, sub["bs"].max() * 1.1, 200)
        # xs = np.linspace(8 * 0.9, 2048 * 1.1, 200)
        log_lr = intercepts[h][0] + slope * np.log2(xs)
        ax.plot(xs, 2 ** log_lr, linestyle="--", color=color)

        # legend entry
        billions = h / 1e9
        billions_str = f"{billions:.1f}" if billions < 10 else f"{int(round(billions))}"
        label = rf"$2^{{{exp}}}$ ({billions_str} B)"
        _leg_ms = float(legend_marker_size) if legend_marker_size is not None else float(marker_size)
        ax.scatter([], [], marker=marker, color=color, s=_leg_ms, label=label)

    # axes & labels
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=2))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=2))
    if axis_label_fontsize is None:
        ax.set_xlabel(r"$B$ [samples]")
        ax.set_ylabel(r"$\eta_{\mathrm{optimal}}$")
    else:
        ax.set_xlabel(r"$B$ [samples]", fontsize=float(axis_label_fontsize))
        ax.set_ylabel(r"$\eta_{\mathrm{optimal}}$", fontsize=float(axis_label_fontsize))

    # tick label sizes
    if tick_label_fontsize is not None:
        ax.tick_params(axis="both", which="major", labelsize=float(tick_label_fontsize))
        ax.tick_params(axis="both", which="minor", labelsize=float(tick_label_fontsize))

    # grid
    ax.grid(which="major", linestyle=":", linewidth=1, alpha=0.4)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.25)

    # Horizon legend (only horizons)
    hlg = ax.legend(fontsize=float(legend_fontsize), frameon=True, loc="upper left",
                    title="Horizon [tokens]")
    hlg.get_title().set_fontsize(float(legend_fontsize))
    ax.add_artist(hlg)

    # Add fit results as a separate legend (alpha, beta, constant)
    try:
        beta = float(model.params["z"]) if "z" in model.params else float('nan')
        beta_se = float(model.bse["z"]) if "z" in model.bse else float('nan')
        C = float(model.params["Intercept"]) if "Intercept" in model.params else float('nan')
        C_se = float(model.bse["Intercept"]) if "Intercept" in model.bse else float('nan')
        r2 = float(getattr(model, 'rsquared', float('nan')))
    except Exception:
        beta = beta_se = C = C_se = r2 = float('nan')

    fit_lines = [
        f"$\\alpha$ = {slope:.1g} ± {slope_se:.1g}, $\\beta$ = {beta:.1g} ± {beta_se:.1g}",
        f"$\\gamma$ = {C:.2g} ± {C_se:.2g}; $R^2$ = {r2:.2f}",
        # f"",
    ]
    # Build dummy handles for the fit legend
    fit_handles = [
        # Add a black dashed line to represent the fitted curves with the formula as label
        Line2D([], [], linestyle='--', color='black', linewidth=2, label="$\\log_2 \\eta = \\alpha\\,\\log_2 B + \\beta\\,\\log_2 H + \\gamma$"),
        Line2D([], [], linestyle='None', marker=None, color='none', label=fit_lines[0]),
        Line2D([], [], linestyle='None', marker=None, color='none', label=fit_lines[1]),
    ]

    # Fit legend at bottom-right
    flg = ax.legend(handles=fit_handles, labels=["$\\log_2 \\eta = \\alpha\\,\\log_2 B + \\beta\\,\\log_2 D + \\gamma$", fit_lines[0], fit_lines[1]],
                    fontsize=float(legend_fontsize), frameon=True,
                    loc="lower right", title=None)
    flg.get_title().set_fontsize(float(legend_fontsize))
    # ax.set_title('Optimal learning rate vs. batch size, per token horizon', fontsize=20)

    fig.tight_layout()

    return {
        "fig": fig,
        "ax": ax,
        "model": model,
        "slope": float(slope),
        "slope_se": float(slope_se),
        "intercepts": intercepts,
        "horizons": horizons,
    }


def plot_interactive_horizon_scatter(
    df: pd.DataFrame,
    horizon: int,
    *,
    step_min: int | None = None,
    step_max: int | None = None,
    color_scale: str = "Viridis",
    loss_range: tuple[float, float] = (3.0, 6.0),
    scatter_frac: float = 1.0,
    marker_size_px: int = 5,
    min_marker_size: int = 10,
    fig_height_px: int = 800,
    fig_width_px: int = 950,
    export_html: bool = False,
    norm_col: str = "output_norm",
) -> go.Figure:
    """Interactive 3‑D scatter of training loss vs. batch size and norm.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing columns for learning rate, batch size, step,
        output norm, train loss, and horizon. This is the same dataframe used
        as input for :func:`build_minima_df`.
    horizon : int
        Horizon value to visualise.
    step_min, step_max : int or None, optional
        Inclusive bounds on the ``step`` column. ``None`` disables the bound.
    color_scale : str, optional
        Plotly continuous colourscale name. Default ``"Viridis"``.
    loss_range : tuple of float, optional
        ``(lo, hi)`` range for the colour scale and z‑axis. Default ``(3.0, 6.0)``.
    scatter_frac : float, optional
        Fraction of points to sample for the background scatter. Default ``1.0``
        keeps all points.
    marker_size_px : int, optional
        Base marker size for regular points. Default ``5``.
    min_marker_size : int, optional
        Marker size for per‑batch‑size minima. Default ``10``.
    fig_height_px, fig_width_px : int, optional
        Dimensions of the resulting figure in pixels.
    export_html : bool, optional
        If ``True``, write ``interactive_3d_plot.html`` in the current
        directory using :meth:`plotly.graph_objects.Figure.write_html`.
    norm_col : str, optional
        Name of the column containing positive norms. Default ``"output_norm"``.

    Returns
    -------
    plotly.graph_objects.Figure
        The created figure.
    """

    required = {"lr", "bs", "step", norm_col, "train_loss", "horizon"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    mask = np.ones(len(df), dtype=bool)
    if step_min is not None:
        mask &= df["step"] >= step_min
    if step_max is not None:
        mask &= df["step"] <= step_max
    sub_df = df.loc[mask].copy()
    if sub_df.empty:
        raise ValueError("No data left after step filtering. Adjust step_min / step_max or verify data.")

    sub_df = sub_df[sub_df["horizon"] == horizon].copy()
    if sub_df.empty:
        raise ValueError("No data for the requested horizon after filtering.")

    if 0 < scatter_frac < 1:
        sub_df = sub_df.sample(frac=scatter_frac, random_state=0)

    log_norm_col = f"log2_{norm_col}"
    sub_df[log_norm_col] = np.where(
        sub_df[norm_col] > 0, np.log2(sub_df[norm_col]), np.nan
    )
    sub_df["log2_bs"] = np.log2(sub_df["bs"].astype(float))
    sub_df.dropna(subset=[log_norm_col, "log2_bs"], inplace=True)

    minima_idx = sub_df.groupby("bs")["train_loss"].idxmin()
    minima_df = sub_df.loc[minima_idx]
    regular_df = sub_df.drop(index=minima_idx)

    z_base = loss_range[0]
    y_axis_base = float(np.floor(sub_df["log2_bs"].min()))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=regular_df[log_norm_col],
            y=regular_df["log2_bs"],
            z=regular_df["train_loss"],
            mode="markers",
            name=f"horizon={horizon:g}",
            marker=dict(
                size=marker_size_px,
                symbol="circle",
                opacity=0.8,
                color=regular_df["train_loss"],
                colorscale=color_scale,
                cmin=loss_range[0],
                cmax=loss_range[1],
                showscale=True,
                colorbar=dict(title=f"train_loss ({loss_range[0]}–{loss_range[1]})"),
            ),
            customdata=np.stack([
                regular_df[c] for c in ("lr", "bs", "step", norm_col)
            ], axis=-1),
            hovertemplate=(
                "bs=%{customdata[1]}<br>"
                "lr=%{customdata[0]:.2g}<br>"
                "step=%{customdata[2]}<br>"
                f"norm=%{{customdata[3]:.3g}}<br>"
                "train_loss=%{z:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=minima_df[log_norm_col],
            y=minima_df["log2_bs"],
            z=minima_df["train_loss"],
            mode="markers",
            name="min per batch size",
            marker=dict(
                size=min_marker_size,
                symbol="circle",
                opacity=1.0,
                color=minima_df["train_loss"],
                colorscale=color_scale,
                cmin=loss_range[0],
                cmax=loss_range[1],
                line=dict(color="white", width=1),
            ),
            customdata=np.stack([
                minima_df[c] for c in ("lr", "bs", "step", norm_col)
            ], axis=-1),
            hovertemplate=(
                "<b>Batch-min loss</b><br>"
                "bs=%{customdata[1]}<br>"
                "lr=%{customdata[0]:.2g}<br>"
                "step=%{customdata[2]}<br>"
                f"norm=%{{customdata[3]:.3g}}<br>"
                "train_loss=%{z:.3f}<extra></extra>"
            ),
            showlegend=True,
        )
    )

    xs, ys, zs = [], [], []
    xh, yh, zh = [], [], []
    for _, row in minima_df.iterrows():
        xs.extend([row[log_norm_col], row[log_norm_col], None])
        ys.extend([row["log2_bs"], row["log2_bs"], None])
        zs.extend([row["train_loss"], z_base, None])

        xh.extend([row[log_norm_col], row[log_norm_col], None])
        yh.extend([row["log2_bs"], y_axis_base, None])
        zh.extend([z_base, z_base, None])

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color="grey", width=1, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=xh,
            y=yh,
            z=zh,
            mode="lines",
            line=dict(color="grey", width=1, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=f"log₂ ({norm_col})",
            yaxis_title="log₂ (batch_size)",
            zaxis_title="train_loss",
            zaxis=dict(range=list(loss_range)),
        ),
        height=fig_height_px,
        width=fig_width_px,
        margin=dict(t=80, b=60, l=60, r=20),
        title=(
            f"3-D scatter · horizon={horizon:g} · train_loss {loss_range[0]}–{loss_range[1]} window"
        ),
    )

    if export_html:
        fig.write_html("interactive_3d_plot.html")

    fig.show()
    return fig


def plot_parallel_coordinates(
    df: pd.DataFrame,
    *,
    horizon: int,
    bs: int,
    lr_exp_min: int,
    lr_exp_max: int,
    top_k: int = 3,
    top_quantile_pct: float = 10.0,
    top_opacity: float = 1.0,
    other_opacity: float = 0.05,
    topk_opacity_levels: Sequence[float] = (1.0, 0.75, 0.5),
    jitter_std: float = 0.007,
    figsize: tuple[float, float] = (12, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot learning-rate configurations as parallel coordinates.

    Parameters
    ----------
    df:
        DataFrame with columns ``lr_in``, ``lr_hidden``, ``lr_out``,
        ``train_loss_mean`` and optionally a corresponding standard deviation
        column (e.g. ``train_loss_std``).  ``horizon`` and ``bs`` columns are
        used for filtering.
    horizon:
        Required horizon (in tokens) of the runs to plot.
    bs:
        Required batch size of the runs to plot.
    lr_exp_min, lr_exp_max:
        Bounds on the ``log₂`` learning-rate axis.
    top_k:
        Number of best configurations (lowest loss) to highlight in orange.
    top_quantile_pct:
        Percentage of best-performing rows (by lowest loss) to draw at full opacity.
        Example: ``10`` means top 10% are fully opaque.
    top_opacity:
        Opacity for the top quantile set (default ``1.0``).
    other_opacity:
        Opacity for the remaining rows (default ``0.05``).
    topk_opacity_levels:
        Opacities for the highlighted Top‑K configurations from best to worst.
    jitter_std:
        Standard deviation of Gaussian jitter applied to non-Top‑K lines.
    figsize:
        Figure size passed to ``plt.figure``. Defaults to ``(12, 6)``.

    Returns
    -------
    (fig, ax):
        The Matplotlib figure and axes for further customisation.
    """

    if lr_exp_min >= lr_exp_max:
        raise ValueError("lr_exp_min must be < lr_exp_max")

    hp_cols = ["lr_in", "lr_hidden", "lr_out"]
    hp_cols_labels = [r"$\eta_\mathrm{input}$", r"$\eta_\mathrm{hidden}$", r"$\eta_\mathrm{output}$"]
    work = df.copy()

    for col in ["horizon", "bs", "lr_in", "lr_hidden", "lr_out", "train_loss_mean"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    std_candidates = [
        c
        for c in work.columns
        if "train_loss" in c and any(s in c.lower() for s in ["std", "stderr", "se", "sd"])
    ]
    preferred = [
        "train_loss_std",
        "train_loss_stderr",
        "train_loss_se",
        "train_loss_sd",
    ]
    std_col = next((c for c in preferred if c in work.columns), None)
    if std_col is None and std_candidates:
        std_col = std_candidates[0]

    lower = 2 ** lr_exp_min
    upper = 2 ** lr_exp_max
    mask = (
        (work["horizon"] == horizon)
        & (work["bs"] == bs)
        & work[hp_cols].ge(lower).all(axis=1)
        & work[hp_cols].le(upper).all(axis=1)
    )
    select_cols = hp_cols + ["train_loss_mean"]
    if std_col is not None and not all(work.loc[mask, std_col].isna()):
        select_cols.append(std_col)
    filtered = work.loc[mask, select_cols].dropna().reset_index(drop=True)

    if filtered.empty:
        raise ValueError(
            f"No rows after filtering: horizon={horizon}, bs={bs}, "
            f"LRs in [2^{{{lr_exp_min}}}, 2^{{{lr_exp_max}}}]."
        )

    log_df = filtered[hp_cols].apply(lambda s: np.log2(s.astype(float)))
    exp_min, exp_max = float(lr_exp_min), float(lr_exp_max)
    norm_df = (log_df - exp_min) / (exp_max - exp_min)
    norm_df = norm_df.clip(0.0, 1.0)

    perf = filtered["train_loss_mean"].values
    n = len(perf)
    order = np.argsort(perf)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)

    # Build alpha mask using top percentage (best = lowest loss)
    pct = float(top_quantile_pct)
    if not np.isfinite(pct):
        pct = 10.0
    pct = max(0.0, min(100.0, pct))
    k_top = int(np.ceil((pct / 100.0) * n)) if n > 0 else 0
    if pct > 0.0 and k_top == 0 and n > 0:
        k_top = 1
    mask_top = ranks < k_top
    alpha_per_line = np.where(mask_top, float(top_opacity), float(other_opacity)).astype(float)

    # --------------------------------------------------------------
    # Best-quantile LR frequency summary (per layer group)
    # Also collect data to draw an overlaid histogram later.
    # --------------------------------------------------------------
    best_hist_data = None  # (all_lrs_sorted_desc, counts_by_group, group_labels)
    try:
        # Use the same selection as the top‑percentage mask
        if np.any(mask_top):
            best_rows = filtered.loc[mask_top, hp_cols]
            print(
                f"[plot_parallel_coordinates] Top-{pct:.0f}% counts (horizon={horizon}, bs={bs}, rows={len(best_rows)}):"
            )
            group_labels = ["input", "hidden", "output"]
            counts_by_group: dict[str, dict[float, int]] = {}
            for col, glabel in zip(hp_cols, group_labels):
                # Sort by LR value (numeric), not by frequency
                values = best_rows[col].astype(float).to_numpy()
                uniq, cnts = (np.unique(values, return_counts=True) if values.size > 0 else (np.array([]), np.array([])))
                # Sort by LR descending: largest LR first
                order_idx = np.argsort(uniq)[::-1]
                uniq_sorted = uniq[order_idx]
                cnts_sorted = cnts[order_idx]
                if uniq_sorted.size == 0:
                    print(f"  - {glabel}: (no entries)")
                    counts_by_group[glabel] = {}
                    continue
                print(f"  - {glabel}:")
                counts_by_group[glabel] = {float(lr): int(c) for lr, c in zip(uniq_sorted, cnts_sorted)}
                for lr_val, cnt in zip(uniq_sorted, cnts_sorted):
                    # Prefer pretty 2^k formatting when applicable
                    try:
                        pretty = format_lr_value_as_pow2_or_float(float(lr_val))
                    except Exception:
                        pretty = f"{float(lr_val):.3g}"
                    print(f"      {pretty}: {int(cnt)}")
            # Union of all LR values across groups, sorted descending
            all_lrs = set()
            for d in counts_by_group.values():
                all_lrs.update(d.keys())
            all_lrs_sorted_desc = sorted(all_lrs, reverse=True)
            best_hist_data = (all_lrs_sorted_desc, counts_by_group, group_labels)
        else:
            print(
                f"[plot_parallel_coordinates] No rows in the top-{pct:.0f}% for horizon={horizon}, bs={bs}."
            )
    except Exception as _e:
        # Don't break plotting if printing fails for any reason
        pass

    top_k = order[: min(top_k, n)]
    top_k_set = set(top_k.tolist())
    if not topk_opacity_levels:
        topk_opacity_levels = [1.0] * len(top_k)
    elif len(topk_opacity_levels) < len(top_k):
        topk_opacity_levels = list(topk_opacity_levels) + [topk_opacity_levels[-1]] * (
            len(top_k) - len(topk_opacity_levels)
        )
    else:
        topk_opacity_levels = list(topk_opacity_levels[: len(top_k)])

    fig = plt.figure(figsize=figsize, dpi=150)
    ax = plt.gca()
    xs = np.arange(len(hp_cols))

    blue = "dimgrey"
    orange = "darkorange"
    grid_grey = "grey"

    for e in range(int(lr_exp_min), int(lr_exp_max) + 1):
        y = (e - exp_min) / (exp_max - exp_min)
        ax.hlines(y, xs.min(), xs.max(), colors=grid_grey, linestyles="dashed", linewidth=0.6, alpha=0.4, zorder=0)
    for j in xs:
        ax.vlines(j, 0, 1, colors=grid_grey, linestyles="dashed", linewidth=0.6, alpha=0.4, zorder=0)

    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, jitter_std, size=(n, len(hp_cols)))

    non_top_indices = [i for i in range(n) if i not in top_k_set]
    non_top_sorted = sorted(non_top_indices, key=lambda i: alpha_per_line[i])
    for i in non_top_sorted:
        ys = norm_df.iloc[i].values.astype(float)
        ys_j = np.clip(ys + noise[i], 0.0, 1.0)
        ax.plot(xs, ys_j, linewidth=1.4, alpha=float(alpha_per_line[i]), color=blue, zorder=2)

    top_k_sorted = top_k[np.argsort(perf[top_k])]
    rank_of = {idx: rank for rank, idx in enumerate(top_k_sorted, start=1)}
    for i in reversed(top_k_sorted):
        rank = rank_of[i]
        alpha_topk = topk_opacity_levels[rank - 1]
        ys = norm_df.iloc[i].values.astype(float)
        ax.plot(xs, ys, linewidth=3.0, alpha=float(alpha_topk), color=orange, zorder=3)

    j = 0
    for e in range(int(lr_exp_min) + 1, int(lr_exp_max) + 1, 2):
        y = (e - exp_min) / (exp_max - exp_min)
        label = rf"$2^{{{e}}}$"
        ax.text(j - 0.03, y, label, ha="right", va="center", fontsize=18, color="black")

    ax.set_xticks(xs)
    ax.set_xticklabels(hp_cols_labels)
    ax.set_yticks([])
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(0, 1)
    ax.grid(False)

    title = (
        f"Best configurations for horizon={nice_horizon_label(horizon)}, B={bs}"
    )
    ax.set_title(title, fontsize=18)

    # --------------------------------------------------------------
    # Inset: grouped (side-by-side) histograms of best-quantile LR counts
    # per layer group. Each LR bin shows three adjacent bars (input/hidden/output)
    # rather than overlapping bars.
    # --------------------------------------------------------------
    try:
        if best_hist_data is not None:
            all_lrs_sorted_desc, counts_by_group, group_labels_hist = best_hist_data
            # Common bins on LR scale using powers of two in the requested range
            bin_edges = 2.0 ** np.arange(int(lr_exp_min), int(lr_exp_max) + 1 + 1)
            num_bins = len(bin_edges) - 1

            # Build per-group counts per bin from the frequency maps
            group_counts = {}
            for glabel in group_labels_hist:
                counts_map = counts_by_group.get(glabel, {})
                # Build an array by summing counts of LR values that fall in each bin
                arr = np.zeros(num_bins, dtype=float)
                if counts_map:
                    lrs = np.array(list(counts_map.keys()), dtype=float)
                    cnts = np.array(list(counts_map.values()), dtype=float)
                    # Determine bin index for each LR using np.digitize
                    # bins: left-inclusive, right-exclusive except last bin
                    idx = np.digitize(lrs, bin_edges, right=False) - 1
                    # Valid indices are [0, num_bins-1]
                    m = (idx >= 0) & (idx < num_bins)
                    for k, c in zip(idx[m], cnts[m]):
                        arr[int(k)] += float(c)
                group_counts[glabel] = arr

            # Create an inset axes placed in the right margin, under the legend
            pos = ax.get_position()  # figure-normalised coords
            inset_w = 0.26
            inset_h = 0.33
            inset_x = min(pos.x1 + 0.02, 0.98 - inset_w)
            inset_y = max(0.05, pos.y1 - inset_h - 0.28)  # sit just below top of main axes
            ax_hist = fig.add_axes([inset_x, inset_y, inset_w, inset_h])
            colors_map = {"input": "#EEE5E9", "hidden": "#287ACC", "output": "#3C9537"}
            # colors_map = {"input": "#F5F3F7", "hidden": "#BC8DA0", "output": "#632C41"}
            # colors_map = {"input": "#B7C3F3", "hidden": "#9FAFBC", "output": "#404E5C"}

            # Grouped bars within each bin with equal widths in log2 space.
            # We compute positions and widths in log2 units, then convert to linear x.
            groups = list(group_labels_hist)
            G = len(groups)
            # Fraction of each bin's log2 width to occupy with bars (leave symmetric margins)
            frac_occupied = 0.85
            # Precompute index of the hidden group (fallback to middle if absent)
            hidden_idx = groups.index("hidden") if "hidden" in groups else (G // 2)
            # Per-group width in log2 units inside a single [2^k, 2^{k+1}] bin
            group_log_w = (frac_occupied / max(G, 1))  # bin log2 width is 1
            for b in range(num_bins):
                left = float(bin_edges[b])
                right = float(bin_edges[b + 1])
                # Bin center in log2 space (k + 0.5 for [2^k, 2^{k+1}])
                bin_center_log2 = 0.5 * (np.log2(left) + np.log2(right))
                for gi, glabel in enumerate(groups):
                    # Center of this group's bar in log2 units, align hidden at bin center
                    center_log2 = bin_center_log2 + (gi - hidden_idx) * group_log_w
                    # Convert symmetric log2 width to linear x-span
                    left_log2 = center_log2 - 0.5 * group_log_w
                    right_log2 = center_log2 + 0.5 * group_log_w
                    x_left = float(2.0 ** left_log2)
                    x_right = float(2.0 ** right_log2)
                    width_lin = max(0.0, x_right - x_left)
                    h = float(group_counts[glabel][b])
                    if h <= 0 or width_lin <= 0:
                        continue
                    ax_hist.bar(
                        x_left,
                        h,
                        width=width_lin,
                        align="edge",
                        color=colors_map.get(glabel, None),
                        edgecolor="black",
                        linewidth=0.3,
                        label=glabel if (b == 0) else None,  # legend once
                        alpha=0.85,
                    )

            # Cosmetics for the inset histogram
            ax_hist.set_xscale("log", base=2)
            # Use fixed major ticks at bin centers with fixed formatter to avoid
            # auto-generated labels overlaying. Label centers as powers of two
            # using the left-edge exponent for each bin (visual consistency with
            # the previous version's tick style, but placed at centers).
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])[1::2]
            left_exponents = np.arange(int(lr_exp_min)+1, int(lr_exp_max), 2)
            ax_hist.xaxis.set_major_locator(FixedLocator(bin_centers))
            ax_hist.xaxis.set_major_formatter(FixedFormatter([rf"$2^{{{e}}}$" for e in left_exponents]))
            # No minor tick labels
            ax_hist.xaxis.set_minor_formatter(NullFormatter())
            ax_hist.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
            ax_hist.set_ylabel("count", fontsize=12)
            ax_hist.tick_params(axis="both", which="major", labelsize=12)
            ax_hist.set_title(f"Top-{pct:.0f}% loss counts per layer group", fontsize=12)
            # Put a compact legend below the histogram inset
            legend_handles = [Patch(facecolor=colors_map[g], edgecolor="black") for g in group_labels_hist]
            ax_hist.legend(
                legend_handles,
                list(group_labels_hist),
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=1,
                frameon=True,
                fontsize=12,
                borderaxespad=0.0,
            )
    except Exception:
        # Never break the main plot if inset creation fails
        pass

    legend_handles = []
    legend_labels = []
    # Two opacity groups: Top‑X% and Others (include loss ranges)
    # Compute ranges over the full filtered set.
    def _range_str(vals: np.ndarray) -> str:
        if vals.size == 0 or not np.any(np.isfinite(vals)):
            return "(none)"
        return f"[{np.nanmin(vals):.2f}, {np.nanmax(vals):.2f}]"

    top_range = _range_str(perf[mask_top])
    oth_mask = ~mask_top if n > 0 else np.array([], dtype=bool)
    oth_range = _range_str(perf[oth_mask])

    h_top = Line2D([0], [0], color=blue, lw=3, alpha=float(top_opacity))
    legend_handles.append(h_top)
    legend_labels.append(f"Top {pct:.0f}% loss: {top_range}")

    if np.any(oth_mask):
        h_oth = Line2D([0], [0], color=blue, lw=3, alpha=float(other_opacity))
        legend_handles.append(h_oth)
        legend_labels.append(f"Others: {oth_range}")

    std_vals = filtered[std_col].values if std_col is not None and std_col in filtered.columns else np.full_like(perf, np.nan)
    for rank, idx in enumerate(top_k_sorted, start=1):
        mean_val = perf[idx]
        std_val = std_vals[idx]
        label = f"Top {rank}: {mean_val:.2f}"
        if np.isfinite(std_val):
            label += f" ± {std_val:.2f}"
        legend_handles.append(Line2D([0], [0], color=orange, lw=3.0, alpha=float(topk_opacity_levels[rank - 1])))
        legend_labels.append(label)

    topk_lr_out = filtered.loc[top_k, "lr_out"].values
    if len(topk_lr_out) > 0:
        vals, counts = np.unique(topk_lr_out, return_counts=True)
        mf_lr_out = vals[np.argmax(counts)]
        mask_equal = (
            (filtered["lr_in"] == mf_lr_out)
            & (filtered["lr_hidden"] == mf_lr_out)
            & (filtered["lr_out"] == mf_lr_out)
        )
        equal_rows = filtered[mask_equal]
        exp_est = np.log2(mf_lr_out)
        if np.isclose(exp_est, round(exp_est)):
            lr_label = rf"$2^{{{int(round(exp_est))}}}$"
        else:
            lr_label = f"{mf_lr_out:.3g}"
        if not equal_rows.empty:
            best_equal_idx = equal_rows["train_loss_mean"].idxmin()
            best_equal_loss = float(filtered.loc[best_equal_idx, "train_loss_mean"])
            if std_col is not None and std_col in filtered.columns and np.isfinite(filtered.loc[best_equal_idx, std_col]):
                best_equal_std = float(filtered.loc[best_equal_idx, std_col])
                eq_label = f"Equal LRs at {lr_label}: {best_equal_loss:.2f} ± {best_equal_std:.2f}"
            else:
                eq_label = f"Equal LRs at {lr_label}: {best_equal_loss:.2f}"
        else:
            eq_label = f"Equal LRs at {lr_label}: no matching row"
        legend_handles.append(Line2D([0], [0], color="white", lw=0, alpha=0.0))
        legend_labels.append(eq_label)

    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=12,
    )

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Two‑predictor global fit (log2(lr) = A·log2(h) + B·log2(bs) + C)
# ---------------------------------------------------------------------------
def plot_global_two_param_fit(
    df: pd.DataFrame,
    *,
    band_center_log2: float = 7.0,
    band_eps: float = 0.2,
    lr_min: float = 0.0,
    lr_max: float = 1.0,
    horizon_min: float | None = 2.0 ** 29,
    horizon_max: float | None = 2.0 ** 37,
    bs_min: float | int | None = 32,
    bs_max: float | int | None = 2048,
    target_log2_horizons: Sequence[float] | None = None,
    coef_csv_path: str | Path | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
    title: str | None = None,
    A_fixed: float | None = None,
    B_fixed: float | None = None,
    C_fixed: float | None = None,
    overlay_solid_A: float | None = None,
    overlay_solid_B: float | None = None,
    legend_fontsize: int | float | None = None,
    axis_label_fontsize: int | float | None = None,
    tick_label_fontsize: int | float | None = None,
    figsize: tuple[float, float] | None = None,
    star_size: float | int = 16,
    marker_size: float | int = 6,
) -> dict:
    """Fit ``log₂(lr) = A·log₂(h) + B·log₂(bs) + C`` and visualise the result.

    For every (batch size, learning rate) pair, the routine finds the first
    horizon whose output norm lies inside the band ``[2**(c - eps), 2**(c + eps)]``
    determined by ``band_center_log2`` (``c``) and ``band_eps`` (``eps``).
    Those points are regressed to obtain the global coefficients. Solid lines
    show the free fit; when both ``overlay_solid_A`` and ``overlay_solid_B`` are
    provided, dashed lines add the constrained fit with those slopes. Legends
    summarise the fitted parameters and batch‑size colours.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing batch size, learning rate, horizon, output norm,
        and train loss columns (resolved case-insensitively).
    band_center_log2, band_eps : float
        Define the output-norm band used to pick the per-configuration horizon.
    lr_min, lr_max : float
        Keep only learning rates within this open interval before searching for
        band entry.
    horizon_min, horizon_max : float or None
        Restrict the fitted horizons to this interval (set to ``None`` to skip a bound).
    bs_min, bs_max : float or int or None
        Restrict the batch sizes considered for the fit.
    target_log2_horizons : Sequence[float] | None
        Reserved for the (currently disabled) star overlay; kept for compatibility.
    coef_csv_path : str or pathlib.Path or None
        Optional path where a one-row CSV with the fitted ``A``, ``B``, ``C`` and
        their standard errors is written.
    ax : matplotlib.axes.Axes or None
        Existing axis to draw on. A new figure/axis pair is created when ``None``.
    show : bool
        Call ``plt.show()`` if a new figure was created.
    title : str or None
        Reserved for future custom titles; currently unused.
    A_fixed, B_fixed, C_fixed : float or None
        Coefficients to hold fixed during the least-squares solve. Unspecified
        coefficients are estimated from the data.
    overlay_solid_A, overlay_solid_B : float or None
        If both are provided, draw dashed curves using those slopes and a fitted
        intercept for comparison with the free fit.
    legend_fontsize : int or float or None
        Font size applied to the batch-size and fit-summary legends.
    axis_label_fontsize : int or float or None
        Font size for the axis labels.
    tick_label_fontsize : int or float or None
        Font size for tick labels on both axes.
    figsize : tuple[float, float] or None
        Figure size to use when creating a new axis. Defaults to ``(9, 8)``.
    star_size : float or int
        Reserved for the star overlay (currently unused, kept for compatibility).
    marker_size : float or int
        Marker size for the unfilled circle data points.

    Returns
    -------
    dict
        Dictionary with the Matplotlib objects and fit diagnostics. Keys:
        ``fig``, ``ax``, ``A``, ``B``, ``C``, ``A_se``, ``B_se``, ``C_se``,
        ``R2``, ``A_constrained``, ``B_constrained``, ``C_constrained``,
        ``C_constrained_se``, ``R2_constrained``, and ``firsts`` (the
        per-(bs, lr) DataFrame of horizons entering the band).
    """

    # ---- Column resolution ----
    def _resolve_columns(frame: pd.DataFrame):
        lc = {c.lower(): c for c in frame.columns}

        def pick(options=None, contains_all=None, fallback=None, required=True, name=""):
            if options:
                for opt in options:
                    if opt in lc:
                        return lc[opt]
            if contains_all:
                for k in lc:
                    if all(token in k for token in contains_all):
                        return lc[k]
            if fallback:
                for k in lc:
                    if fallback in k:
                        return lc[k]
            if required:
                raise KeyError(f"Could not find a column for {name or options or contains_all or fallback}")
            return None

        bs_col = pick(options=["bs", "batch_size"], contains_all=["batch", "size"], fallback="batch", name="batch size (bs)")
        lr_col = pick(options=["lr", "learning_rate"], contains_all=["learning", "rate"], fallback="lr", name="learning rate (lr)")
        horizon_col = pick(options=["horizon"], contains_all=["horizon"], fallback="horizon", name="horizon")
        output_norm_col = pick(options=["output_norm", "norm"], contains_all=["output", "norm"], fallback="norm", name="output_norm")
        train_loss_col = pick(options=["train_loss", "loss"], contains_all=["train", "loss"], fallback="train_loss", name="train_loss")
        return bs_col, lr_col, horizon_col, output_norm_col, train_loss_col

    bs_col, lr_col, horizon_col, output_norm_col, train_loss_col = _resolve_columns(df)

    # Ensure numeric types for used columns
    for col in [bs_col, lr_col, horizon_col, output_norm_col, train_loss_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Parameters & derived ----
    c = float(band_center_log2)
    eps = float(band_eps)
    low = float(2 ** (c - eps))
    high = float(2 ** (c + eps))

    if target_log2_horizons is None:
        target_log2_horizons = list(np.linspace(29, 37, 5))
    target_horizons = (2.0 ** np.array(target_log2_horizons, dtype=float)).astype(float)

    # ---- Filter valid lr and bs ----
    _mask = (df[lr_col] > lr_min) & (df[lr_col] < lr_max)
    if bs_min is not None:
        _mask &= (df[bs_col] >= float(bs_min))
    if bs_max is not None:
        _mask &= (df[bs_col] <= float(bs_max))
    df_lr = df[_mask].copy()
    df_lr = df_lr.sort_values([bs_col, lr_col, horizon_col])

    # ---- Compute per (bs, lr) the first horizon entering the band ----
    def first_enter_band(group: pd.DataFrame):
        in_band = group[(group[output_norm_col] >= low) & (group[output_norm_col] <= high)]
        if in_band.empty:
            return np.nan
        return float(in_band.iloc[0][horizon_col])

    firsts = (
        df_lr.groupby([bs_col, lr_col], sort=True)
        .apply(first_enter_band)
        .dropna()
        .reset_index()
    )
    firsts.columns = ["bs", "lr", "first_horizon_in_band"]
    firsts = firsts[firsts["first_horizon_in_band"] > 0]
    # Apply horizon cuts for fitting/plotting, similar to lr bounds
    if horizon_min is not None:
        firsts = firsts[firsts["first_horizon_in_band"] >= float(horizon_min)]
    if horizon_max is not None:
        firsts = firsts[firsts["first_horizon_in_band"] <= float(horizon_max)]
    # Ensure bs bounds are applied (already filtered in df_lr, but keep for safety)
    if bs_min is not None:
        firsts = firsts[firsts["bs"] >= float(bs_min)]
    if bs_max is not None:
        firsts = firsts[firsts["bs"] <= float(bs_max)]
    if firsts.empty:
        raise ValueError("No (bs, lr) pairs with a first horizon entering the band.")

    # ---- Prepare regression variables & fit A, B, C ----
    firsts["log2_h"] = np.log2(firsts["first_horizon_in_band"])
    firsts["log2_lr"] = np.log2(firsts["lr"])
    firsts["log2_bs"] = np.log2(firsts["bs"])

    # Model: log2_lr ~ A*log2_h + B*log2_bs + C (with intercept)
    # If no fixed parameters are provided, use statsmodels OLS as before.
    if A_fixed is None and B_fixed is None and C_fixed is None:
        ols_res = smf.ols("log2_lr ~ log2_h + log2_bs", data=firsts).fit()
        A = float(ols_res.params.get("log2_h", np.nan))
        B = float(ols_res.params.get("log2_bs", np.nan))
        C = float(ols_res.params.get("Intercept", np.nan))
        A_se = float(ols_res.bse.get("log2_h", np.nan))
        B_se = float(ols_res.bse.get("log2_bs", np.nan))
        C_se = float(ols_res.bse.get("Intercept", np.nan))
        r2 = float(ols_res.rsquared)
    else:
        # Optionally fix A/B/C to provided values and fit the rest by least squares.
        y = firsts["log2_lr"].astype(float).values
        x_h = firsts["log2_h"].astype(float).values
        x_bs = firsts["log2_bs"].astype(float).values

        # Subtract contributions from any fixed parameters
        y_adj = y.copy()
        if A_fixed is not None:
            y_adj = y_adj - float(A_fixed) * x_h
        if B_fixed is not None:
            y_adj = y_adj - float(B_fixed) * x_bs
        if C_fixed is not None:
            y_adj = y_adj - float(C_fixed)

        # Build design matrix for free parameters
        design_cols = []
        names = []
        if A_fixed is None:
            design_cols.append(x_h)
            names.append("A")
        if B_fixed is None:
            design_cols.append(x_bs)
            names.append("B")
        if C_fixed is None:
            design_cols.append(np.ones_like(y_adj))
            names.append("C")

        if len(design_cols) > 0:
            X = np.column_stack(design_cols)
            # Least squares estimation for free parameters
            beta, residuals, rank, s = np.linalg.lstsq(X, y_adj, rcond=None)
            # Map estimates to A, B, C
            est = {name: float(val) for name, val in zip(names, beta)}
        else:
            # All parameters fixed; no fitting performed
            est = {}

        A = float(A_fixed) if A_fixed is not None else float(est.get("A", np.nan))
        B = float(B_fixed) if B_fixed is not None else float(est.get("B", np.nan))
        C = float(C_fixed) if C_fixed is not None else float(est.get("C", np.nan))

        # Compute residuals and R^2 using the assembled model
        y_pred = A * x_h + B * x_bs + C
        resid = y - y_pred
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y - float(np.mean(y)))**2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        # Standard errors: only available for free parameters when n > p and (X^T X) invertible
        n = y.shape[0]
        p = len(design_cols)
        if p > 0 and n > p:
            sigma2 = ss_res / (n - p)
            try:
                XtX_inv = np.linalg.inv(X.T @ X)
            except np.linalg.LinAlgError:
                XtX_inv = np.linalg.pinv(X.T @ X)
            se_free = np.sqrt(np.clip(np.diag(XtX_inv) * sigma2, a_min=0.0, a_max=None))
            se_map = {name: float(val) for name, val in zip(names, se_free)}
        else:
            se_map = {name: float("nan") for name in names}

        A_se = float("nan") if A_fixed is not None else float(se_map.get("A", float("nan")))
        B_se = float("nan") if B_fixed is not None else float(se_map.get("B", float("nan")))
        C_se = float("nan") if C_fixed is not None else float(se_map.get("C", float("nan")))

    # optionally save coefficients
    if coef_csv_path is not None:
        p = Path(coef_csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {
                "A": A,
                "B": B,
                "C": C,
                "A_se": A_se,
                "B_se": B_se,
                "C_se": C_se,
                "R2": r2,
            }
        ]).to_csv(p, index=False)

    # ---- Optional constrained overlay fit (solid lines) ----
    constrained_present = overlay_solid_A is not None and overlay_solid_B is not None
    if constrained_present:
        A_constrained = float(overlay_solid_A)
        B_constrained = float(overlay_solid_B)
        # Fit only C with A,B fixed
        y2 = firsts["log2_lr"].astype(float).values
        xh2 = firsts["log2_h"].astype(float).values
        xbs2 = firsts["log2_bs"].astype(float).values
        y2_adj = y2 - A_constrained * xh2 - B_constrained * xbs2
        X2 = np.ones_like(y2_adj)[:, None]
        beta2, *_ = np.linalg.lstsq(X2, y2_adj, rcond=None)
        C_constrained = float(beta2[0])
        y2_pred = A_constrained * xh2 + B_constrained * xbs2 + C_constrained
        resid2 = y2 - y2_pred
        ss_res2 = float(np.sum(resid2**2))
        ss_tot2 = float(np.sum((y2 - float(np.mean(y2)))**2))
        r2_constrained = float(1.0 - ss_res2 / ss_tot2) if ss_tot2 > 0 else float("nan")
        # SE for C only, if n>1
        n2 = y2.shape[0]
        if n2 > 1:
            sigma2_2 = ss_res2 / (n2 - 1)
            XtX_inv_2 = np.array([[1.0 / float(n2)]])
            C_constrained_se = float(np.sqrt(np.clip(XtX_inv_2[0, 0] * sigma2_2, 0.0, None)))
        else:
            C_constrained_se = float("nan")
    else:
        A_constrained = B_constrained = C_constrained = float("nan")
        C_constrained_se = float("nan")
        r2_constrained = float("nan")

    # ---- Plot ----
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 8) if figsize is None else figsize)
        created_fig = True
    else:
        fig = ax.figure

    bs_color: dict[float, Any] = {}
    bs_values = np.sort(firsts["bs"].unique())
    palette = cm.get_cmap("magma_r", len(bs_values) if len(bs_values) > 0 else 1)
    bs_to_color = {bs: palette(i) for i, bs in enumerate(bs_values)}
    if len(bs_values) > 0:
        min_bs = bs_values[0]
        cont = cm.get_cmap("magma_r")
        bs_to_color[min_bs] = cont(0.07)

    # Unfilled circle markers for data points (X=horizon, Y=lr)
    for bs, g in firsts.groupby("bs"):
        g_sorted = g.sort_values("first_horizon_in_band")
        x = g_sorted["first_horizon_in_band"].values
        y = g_sorted["lr"].values
        color = bs_to_color.get(bs)
        pts, = ax.plot(
            x, y,
            linestyle="None",
            marker="o",
            markersize=marker_size,
            markerfacecolor="none",
            markeredgewidth=2.5,
            color=color,
        )
        # 's' in scatter is in points^2; keep visual size consistent with markersize
        ax.scatter(x, y, facecolors="none", edgecolors=color, s=(marker_size ** 2))
        bs_color[bs] = color

    # Per‑bs solid lines from the free global fit (X=horizon → predict Y=lr)
    handles: list[Line2D] = []
    labels: list[str] = []
    for bs, g in firsts.groupby("bs"):
        x_obs = np.sort(g["first_horizon_in_band"].values)
        x_log2_line = np.linspace(np.log2(x_obs.min()), np.log2(x_obs.max()), 200)
        y_log2_pred = A * x_log2_line + B * np.log2(bs) + C
        x_line = 2.0 ** x_log2_line
        y_line = 2.0 ** y_log2_pred
        h, = ax.plot(x_line, y_line, linestyle="-", color=bs_color[bs], linewidth=2.5)
        handles.append(h)
        try:
            labels.append(f"{int(bs)}")
        except Exception:
            labels.append(f"{bs}")

        # Overlay constrained fit (dashed) if requested
        if constrained_present:
            y_log2_pred2 = A_constrained * x_log2_line + B_constrained * np.log2(bs) + C_constrained
            y_line2 = 2.0 ** y_log2_pred2
            ax.plot(x_line, y_line2, linestyle="--", color=bs_color[bs], linewidth=2.5)

    # Legend with batch sizes
    legend1 = ax.legend(
        handles, labels, title="Batch size [samples]", loc="lower left", ncol=2,
        prop=None if legend_fontsize is None else {"size": legend_fontsize},
        title_fontsize=legend_fontsize if legend_fontsize is not None else None,
    )
    # Legend for fit styles: solid (free fit), dashed (slope-constrained), star (horizon optimal)
    style_handles: list[Line2D] = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2),
    ]
    style_labels: list[str] = [
        "Free fit",
    ]
    if constrained_present:
        style_handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=2))
        style_labels.append("Heuristic fit")
    style_handles.append(Line2D([0], [0], color="black", marker="*", linestyle="None", markersize=14))
    # style_labels.append("Horizon-optimal value")

    legend2 = ax.legend(
        style_handles,
        style_labels,
        loc=(0.42, 0.017),
        frameon=True,
        prop=None if legend_fontsize is None else {"size": legend_fontsize},
    )
    ax.add_artist(legend1)

    # # ---- Stars at fixed horizons (best train_loss across all bs/lr) ----
    # df_lr2 = df_lr.dropna(subset=[train_loss_col, horizon_col, lr_col, bs_col]).copy()
    # # Apply horizon cuts to candidate rows for stars as well
    # if horizon_min is not None:
    #     df_lr2 = df_lr2[df_lr2[horizon_col] >= float(horizon_min)]
    # if horizon_max is not None:
    #     df_lr2 = df_lr2[df_lr2[horizon_col] <= float(horizon_max)]
    # df_lr2["log2_h"] = np.log2(df_lr2[horizon_col])

    # star_rows: list[dict[str, float]] = []
    # for target_log2, target_h in zip(target_log2_horizons, target_horizons):
    #     # Skip targets outside horizon bounds
    #     if horizon_min is not None and not (target_h >= float(horizon_min)):
    #         continue
    #     if horizon_max is not None and not (target_h <= float(horizon_max)):
    #         continue
    #     exact = df_lr2[df_lr2[horizon_col] == target_h]
    #     if exact.empty:
    #         idx = (df_lr2["log2_h"] - target_log2).abs().idxmin()
    #         nearest_h = float(df_lr2.loc[idx, horizon_col])
    #         cand = df_lr2[df_lr2[horizon_col] == nearest_h]
    #     else:
    #         cand = exact
    #     if cand.empty:
    #         continue
    #     min_idx = cand[train_loss_col].idxmin()
    #     row = cand.loc[min_idx]
    #     star_rows.append({
    #         "target_log2_horizon": float(target_log2),
    #         "target_horizon": float(target_h),
    #         "used_horizon": float(row[horizon_col]),
    #         "bs": float(row[bs_col]),
    #         "lr": float(row[lr_col]),
    #         "train_loss_min": float(row[train_loss_col]),
    #     })

    # stars = pd.DataFrame(star_rows)
    # for _, r in stars.iterrows():
    #     color = bs_color.get(r["bs"], None)
    #     ax.plot([r["target_horizon"]], [r["lr"]], marker="*", markersize=star_size, linestyle="None", color=color)

    # Axes, labels, cosmetics
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel(
        f"First horizon to reach optimal norm [tokens]",
        fontsize=axis_label_fontsize,
    )
    ax.set_ylabel(r"$\eta$", fontsize=axis_label_fontsize)
    # ax.set_ylim(2**-12.5, 2**0.)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    if tick_label_fontsize is not None:
        ax.tick_params(axis="both", which="both", labelsize=tick_label_fontsize)

    if created_fig:
        fig.tight_layout()
        if show:
            plt.show()

    return {
        "fig": fig,
        "ax": ax,
        "A": float(A),
        "B": float(B),
        "C": float(C),
        "A_se": float(A_se),
        "B_se": float(B_se),
        "C_se": float(C_se),
        "R2": float(r2),
        "A_constrained": float(A_constrained),
        "B_constrained": float(B_constrained),
        "C_constrained": float(C_constrained),
        "C_constrained_se": float(C_constrained_se),
        "R2_constrained": float(r2_constrained),
        "firsts": firsts,
        # "stars": stars,
    }

def format_horizon_label(h: int) -> str:
    """Render a horizon value as ``2^k`` with an accompanying token count."""
    exp = int(round(np.log2(h))) if h > 0 else 0
    billions = h / 1e9
    if billions >= 10:
        b_txt = f"{billions:.0f} B"
    else:
        b_txt = f"{billions:.1f} B"
    return rf"$2^{{{exp}}}$ ({b_txt})"

def format_lr_value_as_pow2_or_float(x: float) -> str:
    """Express a positive learning rate as ``2^k`` when close to a power of two."""
    if x <= 0 or not np.isfinite(x):
        return f"{x:.3g}"
    e = np.log2(x)
    if np.isclose(e, round(e), rtol=1e-10, atol=1e-12):
        return rf"$2^{{{int(round(e))}}}$"
    return f"{x:.3g}"

def plot_top1_across_horizons(
    df: pd.DataFrame,
    *,
    horizons: Iterable[int],
    bs: int,
    lr_exp_min: int,
    lr_exp_max: int,
    jitter_std: float = 0.007,
    jitter_nonbest_only: bool = True,
    rng_seed: int = 7,
    colors: Mapping[int, object] | Sequence[object] | None = None,
    default_cmap: str = "tab10",
) -> tuple[plt.Figure, plt.Axes]:
    """Visualise the best loss per horizon across a grid of learning-rate triples.

    The dataframe is filtered to the requested batch size and horizons, keeping
    only rows whose learning-rate components fall within the provided
    ``[2**lr_exp_min, 2**lr_exp_max]`` slab. For each horizon the lowest
    ``train_loss_mean`` row is selected and plotted, with optional jitter and a
    legend describing the associated equal-learning-rate baseline.
    """
    if lr_exp_min >= lr_exp_max:
        raise ValueError("lr_exp_min must be < lr_exp_max")

    hp_cols = ["lr_in", "lr_hidden", "lr_out"]
    hp_cols_labels = [r"$\eta_\mathrm{input}$", r"$\eta_\mathrm{hidden}$", r"$\eta_\mathrm{output}$"]

    work = df.copy()
    for col in ["horizon", "bs", "lr_in", "lr_hidden", "lr_out", "train_loss_mean"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    std_candidates = [
        c for c in work.columns
        if "train_loss" in c and any(s in c.lower() for s in ["std", "stderr", "se", "sd"])
    ]
    preferred = ["train_loss_std", "train_loss_stderr", "train_loss_se", "train_loss_sd"]
    std_col = next((c for c in preferred if c in work.columns), None)
    if std_col is None and std_candidates:
        std_col = std_candidates[0]

    lower = 2 ** lr_exp_min
    upper = 2 ** lr_exp_max

    horizons = list(dict.fromkeys(int(h) for h in horizons))  # unique, preserve order
    top_rows = []
    for h in horizons:
        mask = (
            (work["horizon"] == h)
            & (work["bs"] == bs)
            & work[hp_cols].ge(lower).all(axis=1)
            & work[hp_cols].le(upper).all(axis=1)
        )
        cols = hp_cols + ["train_loss_mean"]
        if std_col is not None and not all(work.loc[mask, std_col].isna()):
            cols.append(std_col)
        sub = work.loc[mask, cols].dropna().reset_index(drop=True)
        if sub.empty:
            continue
        i_best = int(sub["train_loss_mean"].idxmin())
        best_row = sub.loc[i_best].copy()
        best_row["horizon"] = h
        top_rows.append(best_row)

    if not top_rows:
        raise ValueError("No rows matched the filters for the requested horizons.")

    best_df = pd.DataFrame(top_rows)

    # Precompute equal-LR losses for each horizon: lr_in=lr_hidden=lr_out=best lr_out
    # Use tolerant equality to find the matching row(s) in the original filtered space.
    eq_losses = {}
    for _, row in best_df.iterrows():
        h = int(row["horizon"])
        lr_eq = float(row["lr_out"])
        mask_base = (work["horizon"] == h) & (work["bs"] == bs)
        # Apply LR bounds
        mask_base &= work[hp_cols].ge(lower).all(axis=1) & work[hp_cols].le(upper).all(axis=1)
        # Tolerant equality on LRs
        m_in  = np.isclose(work.loc[mask_base, "lr_in"].astype(float),    lr_eq, rtol=1e-9, atol=1e-12)
        m_hid = np.isclose(work.loc[mask_base, "lr_hidden"].astype(float), lr_eq, rtol=1e-9, atol=1e-12)
        m_out = np.isclose(work.loc[mask_base, "lr_out"].astype(float),    lr_eq, rtol=1e-9, atol=1e-12)
        mask_eq = mask_base.copy()
        # Align boolean index for mask_eq (same index as work)
        mask_subset = work.index[mask_base]
        eq_index = mask_subset[m_in & m_hid & m_out]
        # Choose the minimum train_loss_mean among matches
        if len(eq_index) > 0:
            eq_vals = work.loc[eq_index, "train_loss_mean"].astype(float)
            if not eq_vals.dropna().empty:
                eq_losses[h] = float(eq_vals.min())
                continue
        eq_losses[h] = np.nan  # no match / no value

    # Normalise to log2 scale in [0,1] for plotting the best rows
    log_df = best_df[hp_cols].apply(lambda s: np.log2(s.astype(float)))
    exp_min, exp_max = float(lr_exp_min), float(lr_exp_max)
    norm_df = (log_df - exp_min) / (exp_max - exp_min)
    norm_df = norm_df.clip(0.0, 1.0)

    # Build horizon->color mapping
    present_horizons_sorted = sorted(best_df["horizon"].astype(int).unique().tolist())
    cmap = plt.get_cmap(default_cmap)
    if colors is None:
        horizon_to_color = {h: cmap(i % 10) for i, h in enumerate(present_horizons_sorted)}
    elif isinstance(colors, dict):
        horizon_to_color = {h: colors.get(h, cmap(i % 10)) for i, h in enumerate(present_horizons_sorted)}
    else:
        seq = list(colors)
        if len(seq) == 0:
            horizon_to_color = {h: cmap(i % 10) for i, h in enumerate(present_horizons_sorted)}
        else:
            cyc = cycle(seq)
            horizon_to_color = {h: next(cyc) for h in present_horizons_sorted}

    # Figure
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = plt.gca()
    xs = np.arange(len(hp_cols))

    grid_grey = "grey"
    for e in range(int(lr_exp_min), int(lr_exp_max) + 1):
        y = (e - exp_min) / (exp_max - exp_min)
        ax.hlines(y, xs.min(), xs.max(), colors=grid_grey, linestyles="dashed",
                  linewidth=0.6, alpha=0.4, zorder=0)
    for j in xs:
        ax.vlines(j, 0, 1, colors=grid_grey, linestyles="dashed",
                  linewidth=0.6, alpha=0.4, zorder=0)

    # Exponent labels along left axis
    j0 = 0
    for e in range(int(lr_exp_min) + 1, int(lr_exp_max) + 1, 1):
        y = (e - exp_min) / (exp_max - exp_min)
        ax.text(j0 - 0.03, y, rf"$2^{{{e}}}$", ha="right", va="center",
                fontsize=18, color="black")

    # Jitter
    n = len(best_df)
    rng = np.random.default_rng(rng_seed)
    noise = rng.normal(0.0, jitter_std, size=(n, len(hp_cols)))

    # Draw from worst to best so the best stays on top
    order_draw = np.argsort(best_df["train_loss_mean"].values)
    global_best_idx = int(best_df["train_loss_mean"].idxmin())

    for idx in order_draw:
        row = best_df.iloc[idx]
        ys = norm_df.iloc[idx].values.astype(float)
        apply_jitter = not (jitter_nonbest_only and best_df.index[idx] == global_best_idx)
        ys_plot = np.clip(ys + (noise[idx] if apply_jitter else 0.0), 0.0, 1.0)
        color = horizon_to_color[int(row["horizon"])]
        ax.plot(xs, ys_plot, linewidth=5.0, alpha=0.95, color=color, zorder=2)

    # Legend: sorted by horizon ascending, now with equal-LR loss info
    legend_handles, legend_labels = [], []
    std_col_found = next((c for c in ["train_loss_std","train_loss_stderr","train_loss_se","train_loss_sd"] if c in best_df.columns), None)
    for h in present_horizons_sorted:
        row = best_df[best_df["horizon"] == h].iloc[0]
        color = horizon_to_color[h]
        loss = float(row["train_loss_mean"])
        label_h = format_horizon_label(int(h))

        # Best line loss ± std
        if std_col_found is not None and np.isfinite(row.get(std_col_found, np.nan)):
            best_text = f"loss = {loss:.2f} ± {float(row[std_col_found]):.2f}"
        else:
            best_text = f"loss = {loss:.2f}"

        # Equal-LR @ best lr_out
        lr_out_best = float(row["lr_out"])
        lr_label = format_lr_value_as_pow2_or_float(lr_out_best)
        eq_loss = eq_losses.get(int(h), np.nan)
        eq_text = f"equal @ {lr_label}: {eq_loss:.2f}" if np.isfinite(eq_loss) else f"equal @ {lr_label}: N/A"

        lbl = f"{label_h}: {best_text} | {eq_text}"
        legend_handles.append(Line2D([0], [0], color=color, lw=3.0))
        legend_labels.append(lbl)

    ax.set_xticks(xs)
    ax.set_xticklabels(hp_cols_labels)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_yticks([])
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(0, 1)
    ax.grid(False)
    ax.set_title(r"Best $\eta_{\mathrm{input}} = \eta_{\mathrm{output}}$ configuration per horizon" + f", B={bs}", fontsize=16)
    ax.legend(
        legend_handles, legend_labels,
        loc="upper center",
        frameon=True,
        ncol=1,
        fontsize=12,
    )

    fig.tight_layout()
    return fig, ax


def plot_norm_vs_horizon_by_lr_bs(df, lr_values=None, bs_values=None, x_col='horizon', figsize=(12, 8)):
    """
    Plot output_norm vs x_col for different (lr, bs) combinations
    
    Parameters:
    - df: DataFrame with columns 'output_norm', x_col, 'lr', 'bs'
    - lr_values: list of specific lr values to plot, or None for all values
    - bs_values: list of specific bs values to plot, or None for all values
    - x_col: column name for x-axis (default: 'horizon')
    """
    
    # Filter data based on specific values
    filtered_df = df.copy()
    if lr_values is not None:
        filtered_df = filtered_df[filtered_df['lr'].isin(lr_values)]
    if bs_values is not None:
        filtered_df = filtered_df[filtered_df['bs'].isin(bs_values)]
    
    # Get unique BS and LR values
    unique_bs = sorted(filtered_df['bs'].unique())
    unique_lr = sorted(filtered_df['lr'].unique())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create colormap for BS values
    bs_colors = plt.cm.magma_r(np.linspace(0.1, 1, len(unique_bs)))
    
    # Create opacity levels for LR values within each BS group
    alpha_levels = np.linspace(0.2, 1.0, len(unique_lr))
    line_styles = ['dotted', 'dashed', 'dashdot', 'solid', ]
    
    for bs_idx, bs_val in enumerate(unique_bs):
        bs_color = bs_colors[bs_idx]
        
        # Get all LR values for this BS
        bs_data = filtered_df[filtered_df['bs'] == bs_val]
        lr_values_for_bs = sorted(bs_data['lr'].unique())
        
        for lr_val in lr_values_for_bs:
            # Find the alpha level for this LR value
            lr_idx = unique_lr.index(lr_val)
            alpha = alpha_levels[lr_idx]
            line_style = line_styles[lr_idx]
            
            # Get data for this specific (lr, bs) combination
            subset = filtered_df.query(f'lr == {lr_val} and bs == {bs_val}')
            
            if len(subset) > 0:
                # Sort by x_col for proper line plotting
                subset = subset.sort_values(x_col)
                
                ax.plot(subset[x_col], subset['output_norm'], 
                       color=bs_color, marker='none', markersize=4,
                       linewidth=2, linestyle=line_style,
                       label=r"$\eta=$"+f'{lr_val:.1e}, '+r"B="+f'{int(bs_val)}', 
                    #    alpha=alpha,
                       )
    
    ax.tick_params(axis="both", which="both", labelsize=22)
    ax.set_xlabel(x_col.capitalize(), fontsize=22)
    ax.set_ylabel(r"log₂||$W_\mathrm{out}$||$_{\mathrm{RMS} \to \infty}$", fontsize=22)
    # ax.set_ylabel('Output Norm', fontsize=22)
    ax.set_title(f"Output norm vs {x_col}" + r" for different $(\eta, B)$ combinations", fontsize=22)
    
    # Set log scale for x-axis if it's horizon, otherwise use linear
    if x_col.lower() == 'horizon':
        ax.set_xscale('log', base=2)
    else:
        ax.set_xscale('log')  # or 'linear' depending on your preference
    
    ax.set_yscale('log', base=2)
    ax.grid(True, alpha=0.3)
    
    # Create custom legend
    n_combinations = len(unique_bs) * len(unique_lr)
    if n_combinations <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
    else:
        # Create a simplified legend showing BS values only
        legend_elements = []
        for bs_idx, bs_val in enumerate(unique_bs):
            legend_elements.append(plt.Line2D([0], [0], color=bs_colors[bs_idx], 
                                            linewidth=3, label=r"B="+f'{int(bs_val)}'))
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
                 title='Batch Size\n(opacity = LR)', fontsize=15)
    
    plt.tight_layout()
    return fig, ax
