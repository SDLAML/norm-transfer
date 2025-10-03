import os
import re
import glob
from typing import List, Pattern, Tuple, Optional

import numpy as np
import pandas as pd


def build_output_filename(
    scaling_prefix: str,
    momentum: float,
    decayed: bool,
    seeds: Optional[List[int]],
    base: str,
    postfix: str = "",
) -> str:
    """
    Create the output CSV filename stem that mirrors the original naming logic.
    """
    if seeds is None:
        seed_part = "seeds"
    elif len(seeds) == 1:
        seed_part = f"seed-{seeds[0]}"
    else:
        seed_part = "seeds-" + "-".join(str(s) for s in seeds)
    parts = [
        base,
        scaling_prefix,
        f"momentum-{momentum}",
        "decayed-" if decayed else "",
        "preprocessed",
        seed_part,
        postfix
    ]
    # collapse repeated dashes and strip
    name = "-".join(p for p in parts if p != "").replace("--", "-").strip("-")
    return name


def collect_csv_files(
    data_path: str,
    base: str,
    scaling_prefix: str,
    momentum: float,
    seeds: Optional[List[int]],
    decayed: bool,
) -> List[str]:
    """
    Return a list of CSV file paths that match the original glob pattern.

    If ``seeds`` is provided, only files whose names contain ``-seed-<seed>``
    for one of the specified seeds are returned. Otherwise all seed files are
    collected.
    """
    decayed_bit = "-decayed" if decayed else ""
    pattern = f"{data_path}/{base}-{scaling_prefix}*-momentum-{momentum}-seed-*{decayed_bit}.csv"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matched pattern:\n  {pattern}")
    
    if seeds is not None:
        seed_re = re.compile(r"-seed-(\d+)")
        selected = []
        for f in files:
            match = seed_re.search(os.path.basename(f))
            if match and int(match.group(1)) in seeds:
                selected.append(f)
        if not selected:
            raise FileNotFoundError(
                f"No files matched seeds {seeds} with pattern:\n  {pattern}"
            )
        files = selected

    return sorted(files)


def load_and_merge_csv(files: List[str]) -> pd.DataFrame:
    """
    Load the list of CSV files and concatenate them into a single DataFrame.
    """
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df


def rename_metrics(
    df: pd.DataFrame,
    norm_name: str,
    norm_label: str,
    loss_name: str,
    loss_label: str,
) -> pd.DataFrame:
    """
    Apply the column renames for 'step', norm metric, and loss metric.
    """
    rename_map = {"_step": "step", norm_name: norm_label, loss_name: loss_label}
    present = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=present)


def build_runname_pattern(momentum: float) -> Pattern:
    """
    Build a regex pattern with named groups for run_name filtering.
    - For momentum==1.0, expects: lr-<float|i{int}h{int}o{int}>-bs-<int>[-seed-<int>]
    - For other momentum, expects: lr-<float|i{int}h{int}o{int}>-bs-<int>-momentum-<float>[...]
    """
    float_re = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    lr_re = rf"(?:{float_re}|i\d+h\d+o\d+)"

    if momentum == 1.0:
        pattern = rf"^lr-(?P<lr>{lr_re})-bs-(?P<bs>\d+)(?:-seed-(?P<seed>\d+))?$"
    else:
        pattern = rf"^lr-(?P<lr>{lr_re})-bs-(?P<bs>\d+)-momentum-(?P<momentum>{float_re})(?:-.*)?$"

    return re.compile(pattern)
    

def filter_runs_by_name(
    df: pd.DataFrame,
    pattern: Pattern,
    decayed: bool,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filter rows by 'run_name' matching the supplied pattern.
    Also validates the 'decayed' flag expectation.
    """
    if "run_name" not in df.columns:
        raise KeyError("Expected column 'run_name' not found in DataFrame.")

    if decayed:
        # Validate that all runs indicate decayed in their name (mirrors original assert)
        missing = df[~df["run_name"].str.contains("decayed", na=False)]
        if not missing.empty:
            raise ValueError(
                f"DECAYED=True but {len(missing)} runs lack 'decayed' in run_name. "
                f"Examples: {missing['run_name'].head(3).tolist()}"
            )

    # Apply pattern filter
    name_mask = df["run_name"].str.match(pattern)
    filtered_out = len(df) - int(name_mask.sum())
    if verbose:
        print(f"Filtered {filtered_out} runs")
        if filtered_out:
            print("Unique run names (filtered out):", np.unique(df.loc[~name_mask, "run_name"]))
    df = df.loc[name_mask].copy()

    return df


def ensure_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast 'step' to int if present. If 'bs' is missing, attempt to extract it from run_name.
    """
    if "step" in df.columns:
        df["step"] = df["step"].astype(int)

    # Extract bs from run_name if it's not already a column
    if "bs" not in df.columns:
        print("bs column not found; attempting to extract from run_name")
        # Try to parse from run_name: look for '-bs-<int>'
        bs_extract = df["run_name"].str.extract(r"-bs-(?P<bs>\d+)")
        if bs_extract.isna().any().any():
            raise KeyError("Could not infer 'bs' from run_name; 'bs' column is missing.")
        df["bs"] = bs_extract["bs"].astype(int)

    return df


def add_horizon_column(df: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    """
    Add 'horizon' = bs * step * seq_len as int.
    """
    required = {"bs", "step"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for horizon calculation: {missing}")
    df["horizon"] = (df["bs"] * df["step"] * seq_len).astype(int)
    return df


def save_dataframe(df: pd.DataFrame, output_data_path: str, filename_stem: str) -> str:
    """
    Save DataFrame to <output_data_path>/<filename_stem>.csv and return the path.
    """
    os.makedirs(output_data_path, exist_ok=True)
    out_path = os.path.join(output_data_path, f"{filename_stem}.csv")
    df.to_csv(out_path, index=False)
    return out_path


def process_runs(
    data_path: str,
    base: str,
    scaling_prefix: str,
    momentum: float,
    decayed: bool,
    seeds: Optional[List[int]],
    postfix: str,
    seq_len: int,
    norm_name: str,
    norm_label: str,
    loss_name: str,
    loss_label: str,
    output_data_path: str,
    output_filename: str,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    End-to-end pipeline:
    1) Collect matching CSVs
    2) Merge
    3) Rename columns (step, norm, loss)
    4) Filter runs by run_name pattern
    5) Ensure numeric cols (step, bs)
    6) Compute 'horizon'
    7) Save to CSV; return (df, path)
    """
    files = collect_csv_files(data_path, base, scaling_prefix, momentum, seeds, decayed)
    if verbose:
        print(f"Found {len(files)} files")

    df = load_and_merge_csv(files)
    df = rename_metrics(df, norm_name, norm_label, loss_name, loss_label)

    pattern = build_runname_pattern(momentum)
    df = filter_runs_by_name(df, pattern, decayed, verbose=verbose)

    df = ensure_numeric_columns(df)
    df = add_horizon_column(df, seq_len)

    if output_filename is None:
        output_filename = build_output_filename(
            scaling_prefix=scaling_prefix,
            momentum=momentum,
            decayed=decayed,
            seeds=seeds,
            base=base,
            postfix=postfix
        )

    out_path = save_dataframe(df, output_data_path, output_filename)
    if verbose:
        print(f"Saved data to: {out_path}")

    return df, out_path


def deduplicate_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Post-process preprocessed data by removing duplicates and aggregating.

    The function performs several checks and transformations:

    1. Ensure required columns exist and are numeric.
    2. Drop duplicate rows that share the same hyperparameters, step, seed and
       train loss.
    3. Validate that each (hyperparameters + step + seed) combination has a
       single train loss entry.
    4. Ensure that ``horizon`` is consistent within each hyperparameter group.
    5. Aggregate metrics across seeds for each hyperparameter group.

    Parameters
    ----------
    df:
        The DataFrame produced by :func:`process_runs`.

    Returns
    -------
    pd.DataFrame
        Aggregated statistics for each hyperparameter group.
    """

    required_cols = [
        "lr_in",
        "lr_hidden",
        "lr_out",
        "bs",
        "step",
        "seed",
        "train_loss",
        "horizon",
        "output_norm",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    group_cols = ["lr_in", "lr_hidden", "lr_out", "bs", "step"]
    key_with_seed = group_cols + ["seed"]

    before_rows = len(df)
    df_filtered = df.drop_duplicates(
        subset=key_with_seed + ["train_loss"], keep="last"
    ).copy()
    after_rows = len(df_filtered)
    removed_rows = before_rows - after_rows
    print(
        f"Removed {removed_rows} rows with exact same loss for (hparams+step+seed) duplicates."
    )

    per_seed_counts = (
        df_filtered.groupby(key_with_seed, dropna=False)
        .size()
        .reset_index(name="rows_per_seed")
    )
    remaining_multi = per_seed_counts[per_seed_counts["rows_per_seed"] > 1].copy()
    remaining_multi = remaining_multi.sort_values(group_cols + ["seed"]).reset_index(
        drop=True
    )
    assert (
        remaining_multi.empty
    ), "There are still seeds with multiple rows having different losses."

    horizon_nunique = (
        df_filtered.groupby(group_cols, dropna=False)["horizon"]
        .nunique(dropna=False)
        .reset_index(name="horizon_nunique")
    )
    horizon_values = (
        df_filtered.groupby(group_cols, dropna=False)["horizon"]
        .apply(lambda s: sorted(pd.Series(s.dropna().unique()).tolist()))
        .reset_index(name="horizon_values")
    )
    horizon_conflicts = (
        horizon_nunique.merge(horizon_values, on=group_cols, how="left")
        .query("horizon_nunique > 1")
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    assert (
        horizon_conflicts.empty
    ), "There are groups with multiple horizon values. Please resolve this before proceeding."

    agg = (
        df_filtered.groupby(group_cols, dropna=False)
        .agg(
            train_loss_mean=("train_loss", "mean"),
            train_loss_std=("train_loss", "std"),
            output_norm_mean=("output_norm", "mean"),
            output_norm_std=("output_norm", "std"),
            n=("train_loss", "size"),
            horizon=("horizon", "first"),
        )
        .reset_index()
        .sort_values(group_cols + ["train_loss_mean"])
    )

    return agg


if __name__ == "__main__":
    NORM_NAME, NORM_LABEL = "track_param_rms_to_l1/model_part_0/output", "output_norm"
    # NORM_NAME, NORM_LABEL = "track_param_l1_to_rms/model_part_0/output", "output_norm"
    # NORM_NAME, NORM_LABEL = "track_param_rms_to_l1/model_part_0/tok_embeddings", "output_norm"
    LOSS_NAME, LOSS_LABEL = "loss_metrics/global_avg_loss", "train_loss"

    SEQ_LEN = 4096
    MOMENTUM = 1.0
    DECAYED = False

    DATA_PATH = "data/raw"
    BASE = "lr-bs-scan"
    SCALING_PREFIX = "base"
    SEEDS = [30] # None for all seeds
    POSTFIX = ""

    OUTPUT_DATA_PATH = "data"
    OUTPUT_FILENAME = build_output_filename(
        base=BASE,
        scaling_prefix=SCALING_PREFIX,
        momentum=MOMENTUM,
        decayed=DECAYED,
        seeds=SEEDS,
        postfix=POSTFIX
    )

    df, path = process_runs(
        data_path=DATA_PATH,
        base=BASE,
        scaling_prefix=SCALING_PREFIX,
        momentum=MOMENTUM,
        decayed=DECAYED,
        seeds=SEEDS,
        postfix=POSTFIX,
        seq_len=SEQ_LEN,
        norm_name=NORM_NAME,
        norm_label=NORM_LABEL,
        loss_name=LOSS_NAME,
        loss_label=LOSS_LABEL,
        output_data_path=OUTPUT_DATA_PATH,
        output_filename=OUTPUT_FILENAME,
        verbose=True,
    )
