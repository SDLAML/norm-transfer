import os
import re
import argparse

import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm

WANDB_PROJECT = "atmlaml/norm-transfer"
WANDB_GROUP = "eta-BS-scan-per-layer"
METRIC_NAMES = [
                "track_param_rms_to_l1/model_part_0/output", 
                "track_param_rms_to_rms/model_part_0/output",
                "track_param_l1_to_rms/model_part_0/tok_embeddings",
                "track_param_rms_to_rms/model_part_0/layers.3.feed_forward.w2",
                "loss_metrics/global_avg_loss", 
                "lr/opt_0/group_0", "lr/opt_0/group_1", "lr/opt_0/group_2", 
                "_step",
                ]

# Learning rate and batch size grids
LR_GRID = [0.00048828, 0.00069053, 0.00097656, 0.00138107, 0.00195312, 0.00276214, 
           0.00390625, 0.00552427, 0.0078125, 0.01104854, 0.015625, 0.02209709, 
           0.03125, 0.04419417, 0.0625, 0.08838835, 0.125, 0.1767767, 0.25, 
           0.35355339, 0.5, 0.70710678, 1.0, 1.41421356, 2.0, 2.82842712, 4.0,
           7.62939453e-06, 1.07895932e-05, 1.52587891e-05, 2.15791864e-05,
           3.05175781e-05, 4.31583729e-05, 6.10351562e-05, 8.63167458e-05,
           1.22070312e-04, 1.72633492e-04, 2.44140625e-04,
           0.00006103515, 0.00024414, 0.00034527, 0.00012207, 1.5, 3.0, 6.0, 11.0, 14.0,
           8.0, 10.0, 12.0, 16.0,
           ]
BS_GRID = np.logspace(3, 11, 9, base=2)

# Define the range of runs to process
RUN_I, RUN_J = 0, 1000
MOMENTUM = 1.0
DECAYED = False
SEED = None
BS_FILTER = None

# Construct the output file name based on the parameters
parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=None, help="Filter runs by batch size")
parser.add_argument("--seed", type=int, default=None, help="Filter runs by seed")
parse_args = parser.parse_args()

BS_FILTER = parse_args.bs
SEED = parse_args.seed

MOMENTUM_POSTFIX = f"momentum-{MOMENTUM}"
SEED_POSTFIX = f"seed-{SEED}" if SEED is not None else "seed-all"
BS_POSTFIX = f"bs-{BS_FILTER}" if BS_FILTER is not None else "bs-all"
OUTPUT_FILE = (
    f"wandb_logs/{WANDB_GROUP}-history-{RUN_I}-{RUN_J}-"
    f"{MOMENTUM_POSTFIX}-{SEED_POSTFIX}-{BS_POSTFIX}{'-decayed' if DECAYED else ''}.csv"
)

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Initialize the Weights & Biases API
api = wandb.Api(timeout=120)
filters = {"group": WANDB_GROUP}
# if DECAYED: 
#     filters["display_name"] = {"$regex": "decayed"}
runs = api.runs(WANDB_PROJECT, filters=filters)
df_list = []

# Iterate through each run and extract the specified metrics
for run in tqdm(runs[RUN_I:RUN_J], desc="Processing runs"):
    run_name = run.name
    if 'test' in run_name:
        continue
    if DECAYED != ('decayed' in run_name):
        continue
    if (MOMENTUM != 1.0) ^ ('momentum' in run_name):
        continue

    if run.metadata is None:
        tqdm.write(f"Skipping run {run_name} due to missing metadata.")
        continue
    args = run.metadata['args']
    seed = next(
        int(arg.split("=")[1]) for arg in args if arg.startswith("--training.seed=")
    )
    if SEED is not None and seed != SEED:
        continue

    bs = next(
        int(arg.split("=")[1]) for arg in args if arg.startswith("--training.global_batch_size=")
    )
    assert bs in BS_GRID, f"Unexpected batch size: {bs}"
    if BS_FILTER is not None and bs != BS_FILTER:
        continue

    momentum = next(
        float(arg.split("=")[1]) for arg in args if arg.startswith("--optimizer.momentum=")
    )
    if momentum != MOMENTUM:
        continue

    run_history = run.scan_history(keys=METRIC_NAMES)
    run_df = pd.DataFrame(list(run_history))
    run_df['run_name'] = run_name
    run_df['bs'] = bs
    run_df['seed'] = seed
    run_df['momentum'] = momentum

    if all(run_df["lr/opt_0/group_0"] == run_df["lr/opt_0/group_1"]) and all(run_df["lr/opt_0/group_1"] == run_df["lr/opt_0/group_2"]):
        assert len(np.unique(run_df["lr/opt_0/group_0"])) == 1
        run_df['lr'] = run_df["lr/opt_0/group_0"].iloc[0]
        run_df['lr_hidden'] = run_df['lr']
        run_df['lr_in'] = run_df['lr']
        run_df['lr_out'] = run_df['lr']
    else:
        run_df['lr'] = None
        run_df['lr_hidden'] = run_df["lr/opt_0/group_0"]
        run_df['lr_in'] = run_df["lr/opt_0/group_1"]
        run_df['lr_out'] = run_df["lr/opt_0/group_2"]

    df_list.append(run_df)
    tqdm.write(f"Processed run: {run_name}, Number of records: {len(run_df)}")

# Concatenate all run dataframes into a single dataframe
run_df = pd.concat(df_list, ignore_index=True)

# Save the combined dataframe to a CSV file
run_df.to_csv(OUTPUT_FILE, index=False)
