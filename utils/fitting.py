from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Mapping, Any

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Config & small helpers
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    """Configuration bundle for loading sweeps, fitting minima, and styling plots."""
    csv_path: str
    horizons: Sequence[int]
    max_loss: float = 11.8
    from_fit: bool = True
    c_fixed: float | None = None
    optimum_from_closest: bool = False

    # plot styling
    figsize: tuple[float, float] = (12, 7)
    markers: Sequence[str] = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "*")
    bs_size_base: int = 40
    bs_size_factor: float = 1.5

    # plotting overrides for model-comparison view (taken from cfg rather than function args)
    model_styles: Mapping[str, Mapping[str, Any]] | None = None  # per-model styles
    default_marker: str = "o"
    default_linestyle: str = "solid"
    default_alpha: float = 1.0
    show_legends: bool = True
    legend_models_loc: str = "upper left"
    legend_models_bbox: tuple[float, float] | None = (1.02, 1.0)
    legend_bs_loc: str = "upper left"
    legend_bs_bbox: tuple[float, float] | None = (1.02, 0.65)
    use_constrained_layout: bool = False
    tight_layout_right: float = 0.75
    line_width: float = 4.0
    legend_fontsize: float | int | None = None
    axis_label_fontsize: float | int | None = None
    tick_label_fontsize: float | int | None = None

    # generalized averaging window (over steps within a trajectory)
    avg_rel_from: int = 0
    avg_rel_to:   int = 0
    strict_avg: bool = False

    # refit only around the optimum
    fit_k: int | None = None       # e.g., 5 => refit using 5 nearest points
    fit_k_by: str = "x"            # "x" (distance in x) or "xy" (2D distance)

    # control where (h, bs) the step-window averaging is applied
    # None => apply to all. Empty sequence => apply to none.
    average_h: Sequence[int] | None = None
    average_bs: Sequence[int] | None = None

    # skip quadratic fitting for specific (horizon, batch_size) pairs
    # use the empirical minimum for these combinations
    skip_fit: Sequence[tuple[int, int]] = ()


@dataclass(frozen=True)
class Fit:
    """Result of a quadratic fit exposing the minimum and polynomial coefficients."""
    x0: float
    y0: float
    a: float | None
    b: float | None
    c: float | None

def _quad_fit_w(xs: np.ndarray, ys: np.ndarray, w: np.ndarray, cfix: float | None):
    """Weighted LS for y = a x^2 + b x + c (c may be fixed)."""
    # scale rows by sqrt(w) to avoid forming a big diagonal W
    sw = np.sqrt(w)
    if cfix is None:
        X = np.column_stack((xs**2, xs, np.ones_like(xs)))
        beta, *_ = np.linalg.lstsq(X * sw[:, None], ys * sw, rcond=None)
        a_, b_, c_ = beta
    else:
        X = np.column_stack((xs**2, xs))
        Y = ys - cfix
        beta, *_ = np.linalg.lstsq(X * sw[:, None], Y * sw, rcond=None)
        a_, b_ = beta
        c_ = cfix
    return float(a_), float(b_), float(c_)
    
# ──────────────────────────────────────────────
# Core logic
# ──────────────────────────────────────────────
def compute_optimum(
    x: np.ndarray,
    y: np.ndarray,
    *,
    from_fit: bool = True,
    c_fixed: float | None = None,
    optimum_from_closest: bool = False,
    fit_k: int | None = None,          # number of points to refit on
    fit_k_by: str = "x",               # "x" or "xy"
    y_std: np.ndarray | None = None,   # std of y in loss space
    min_var: float = 1e-12,            # guard to avoid zero/neg variances
) -> Fit | None:
    """
    Find ``(x0, y0)`` of minimum. If standard deviations for ``y`` are provided,
    fit a parabola using weighted least squares with weights ``1/σ_y²``.
    """

    # Use the empirical minimum as a seed and guaranteed fallback.
    i_emp = int(np.argmin(y))
    x_emp, y_emp = float(x[i_emp]), float(y[i_emp])

    # Honor configurations that request the empirical optimum directly.
    if not from_fit:
        return Fit(x_emp, y_emp, None, None, None)

    # Normalize input arrays to flat float vectors and validate optional noise estimates.
    x  = np.asarray(x, dtype=float).ravel()
    y  = np.asarray(y, dtype=float).ravel()
    nx = len(x)
    if y_std is not None:
        y_std = np.asarray(y_std, dtype=float).ravel()
        if len(y_std) != nx:
            raise ValueError("y_std must have the same length as x and y.")

    # ----- choose local subset around the empirical min -----
    # Either use all points or restrict to the nearest ``fit_k`` observations.
    if fit_k is None or fit_k >= len(x):
        idx = np.arange(len(x))
    else:
        k = max(3, min(int(fit_k), len(x)))
        if fit_k_by.lower() == "xy":
            d = (x - x_emp) ** 2 + (y - y_emp) ** 2
        else:
            d = np.abs(x - x_emp)
        order = np.argsort(d)
        idx = order[:k]

        # try to bracket x_emp
        left = (x[idx] < x_emp).any()
        right = (x[idx] > x_emp).any()
        if not (left and right) and k < len(x):
            for j in order[k:]:
                need_left = not left and x[j] < x_emp
                need_right = not right and x[j] > x_emp
                if need_left or need_right:
                    # replace farthest in |x - x_emp|
                    far = idx[np.argmax(np.abs(x[idx] - x_emp))]
                    idx = np.array([t for t in idx if t != far] + [j])
                    left = (x[idx] < x_emp).any()
                    right = (x[idx] > x_emp).any()
                    if left and right:
                        break

    # Slice arrays down to the selected subset for fitting.
    xs, ys = x[idx], y[idx]
    ys_std = y_std[idx] if y_std is not None else None

    # ----- build initial weights from y-uncertainty (or uniform) -----
    # Convert y-standard deviations into inverse-variance weights when provided.
    if ys_std is None:
        w = np.ones_like(xs)
    else:
        var = np.where(np.isfinite(ys_std), ys_std, 0.0) ** 2
        var = np.maximum(var, min_var)
        w = 1.0 / var

    # Fit the quadratic; fall back to the best empirical point if it is nearly flat.
    a, b, c = _quad_fit_w(xs, ys, w, c_fixed)
    if abs(a) < 1e-10:
        j = int(np.argmin(ys))
        return Fit(float(xs[j]), float(ys[j]), None, None, None)

    # Vertex of the parabola: compute analytically from the coefficients.
    x0 = -b / (2.0 * a)
    y0 = c - (b ** 2) / (4.0 * a)

    # Warning if extrapolated minima fall outside the local window.
    xmin, xmax = float(xs.min()), float(xs.max())
    if not (xmin <= x0 <= xmax):
        print('\n(xmin <= x0 <= xmax) violated; ignoring and fitting anyway\n')

    # Optionally snap to the closest observed point when configured.
    if optimum_from_closest:
        j = int(np.argmin((xs - x0) ** 2 + (ys - y0) ** 2))
        x0, y0 = float(xs[j]), float(ys[j])

    # Return the fitted minimum and coefficients for downstream consumers.
    return Fit(float(x0), float(y0), float(a), float(b), float(c))


def select_avg_window_at_horizon(
    g: pd.DataFrame,
    horizon: int,
    *,
    cfg: Config,
) -> pd.Series | None:
    """Pick the horizon-reaching row and apply the configured step window average.

    Returns the aggregated row with derived statistics (e.g. geometric mean norm
    and log-space standard deviations) or ``None`` if the horizon is never
    reached.
    """
    # Bail out if there are no observations for this trajectory.
    if g.empty:
        return None

    # Sort by step so the window arithmetic operates on monotonically increasing steps.
    g_sorted = g.sort_values("step")
    reached = g_sorted[g_sorted["horizon"] == horizon]
    if reached.empty:
        return None

    # Identify the first row that reaches the horizon and remember its position.
    horizon_row = reached.iloc[0]
    pos = g_sorted.index.get_loc(horizon_row.name)
    if isinstance(pos, (slice, np.ndarray)):
        pos = int(np.atleast_1d(pos)[0])

    # Reuse the batch size recorded for this trajectory.
    bs_val = int(g_sorted["bs"].iloc[0])

    # decide whether to apply the step-window averaging for this (h, bs)
    def _selected(vals: Sequence[int] | None, value: int) -> bool:
        if vals is None:
            return True
        return value in set(vals)

    # Check if averaging is enabled for both the horizon and batch selections.
    apply_window = _selected(cfg.average_h, horizon) and _selected(cfg.average_bs, bs_val)

    if not apply_window:
        # Without averaging, return the raw horizon row with zeroed spreads.
        out = float(horizon_row["output_norm"])
        if out <= 0:
            raise ValueError("output_norm must be > 0 to take log2.")
        return pd.Series(
            {
                "bs": bs_val,
                "lr": float(g_sorted["lr"].iloc[0]),
                "output_norm": out,
                "log2_output_norm": float(np.log2(out)),
                "train_loss": float(horizon_row["train_loss"]),
                "step": int(horizon_row["step"]),
                "log2_output_norm_std": 0.0,
                "train_loss_std": 0.0,
                "n_window": 1,
            }
        )

    # Window-averaging path: translate the relative bounds into absolute indices around ``pos``.
    A = int(cfg.avg_rel_from)
    B = int(cfg.avg_rel_to)
    if A > 0:
        raise ValueError("avg_rel_from (A) must be <= 0.")
    start_rel, end_rel = sorted((A, B))
    start_pos = pos + start_rel
    end_pos   = pos + end_rel

    # Clamp the window to the data range and optionally fall back when empty.
    n = len(g_sorted)
    start_pos_clamped = max(0, start_pos)
    end_pos_clamped   = min(n - 1, end_pos)

    if start_pos_clamped > end_pos_clamped:
        if cfg.strict_avg:
            raise IndexError(f"Window [{A}, {B}] around pos={pos} is empty after clamping.")
        start_pos_clamped = end_pos_clamped = pos

    # Extract the desired window (or the single horizon row as a fallback).
    window = g_sorted.iloc[start_pos_clamped : end_pos_clamped + 1]
    if window.empty:
        if cfg.strict_avg:
            raise IndexError("Averaging window is empty.")
        window = reached

    # Averages and stds: aggregate norms in log space to produce a geometric mean and spread.
    out_vals = window["output_norm"].astype(float).to_numpy()
    if np.any(out_vals <= 0):
        raise ValueError("output_norm must be > 0 to take log2.")
    log2_vals = np.log2(out_vals)
    log2_out_norm_avg = float(np.mean(log2_vals))
    log2_out_norm_std = float(np.std(log2_vals))          # population std (ddof=0)

    out_norm_geom_mean = float(2.0 ** log2_out_norm_avg)

    # Average the loss values within the window and record their dispersion.
    loss_vals = window["train_loss"].astype(float).to_numpy()
    loss_avg  = float(np.mean(loss_vals))
    loss_std  = float(np.std(loss_vals))                  # population std (ddof=0)

    lr_val   = float(g_sorted["lr"].iloc[0])

    # Emit the aggregated statistics for downstream analysis.
    return pd.Series(
        {
            "bs": bs_val,
            "lr": lr_val,
            "output_norm": out_norm_geom_mean,       # geometric mean
            "log2_output_norm": log2_out_norm_avg,   # mean in log2 space
            "train_loss": loss_avg,
            "step": int(window["step"].max()),
            "log2_output_norm_std": log2_out_norm_std,
            "train_loss_std": loss_std,
            "n_window": int(len(window)),
        }
    )


def build_minima_df(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Collect per-horizon minima, optionally fitted with quadratics, into a dataframe."""
    frames: list[dict] = []

    # Work on a float-typed copy so downstream math is predictable.
    work = df.copy()
    work["output_norm"] = work["output_norm"].astype(float)
    work["train_loss"]  = work["train_loss"].astype(float)
    work["lr"]          = work["lr"].astype(float)

    # Precompute (horizon, batch size) pairs where fitting should be skipped.
    skip_pairs = {(int(hh), int(bb)) for hh, bb in cfg.skip_fit}

    for h in cfg.horizons:
        # Collect one averaged row per (bs, lr) that actually reaches this horizon.
        rows = (
            work.sort_values("step")
            .groupby(["bs", "lr"], as_index=False)
            .apply(lambda g, h=h: select_avg_window_at_horizon(g, h, cfg=cfg))
            .dropna(how="any")
        )

        for bs, sub in rows.groupby("bs"):
            # Ignore trajectories with large losses and require enough points for a fit.
            sub = sub[sub["train_loss"] < cfg.max_loss]
            if len(sub) < 3:
                continue

            # Prepare arrays for fitting the loss as a function of log2 output norm.
            x = sub["log2_output_norm"].to_numpy()
            y = sub["train_loss"].to_numpy()
            y_std = sub["train_loss_std"].to_numpy()       if "train_loss_std" in sub.columns else None

            # Optionally run the quadratic fit unless this (h, bs) combination is skipped.
            do_fit = cfg.from_fit and (h, int(bs)) not in skip_pairs
            fit = compute_optimum(
                x, y,
                from_fit=do_fit,
                c_fixed=cfg.c_fixed,
                optimum_from_closest=cfg.optimum_from_closest,
                fit_k=cfg.fit_k,
                fit_k_by=cfg.fit_k_by,
                y_std=y_std,
            )

            if fit is None:
                continue

            # Find the observed point closest to the fitted minimum to report empirical stats.
            dx = sub["log2_output_norm"] - fit.x0
            dy = sub["train_loss"] - fit.y0
            idx_closest = (dx**2 + dy**2).idxmin()
            lr_closest = float(sub.at[idx_closest, "lr"])

            # Record the per-horizon result for downstream plotting/analysis.
            frames.append(
                dict(
                    horizon=h,
                    bs=int(bs),
                    log2_output_norm=float(fit.x0),
                    train_loss=float(fit.y0),
                    lr=lr_closest,
                    log2_output_norm_std=float(sub.at[idx_closest, "log2_output_norm_std"]),
                    train_loss_std=float(sub.at[idx_closest, "train_loss_std"]),
                    n_window=int(sub.at[idx_closest, "n_window"]),
                )
            )

    return pd.DataFrame(frames)
