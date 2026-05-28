"""Convert datagenerationworker.py JSONL outputs to Parquet for verl RLVR.

Output schema follows verl's RLHFDataset convention:
    prompt:       list[{"role": "user", "content": str}]
    reward_model: {"style": "rule", "ground_truth": str}
    data_source:  str
    extra_info:   {"index": int, "op": int, "mode": str}

Example:
    python prepare_rlvr_data.py \\
        --input-dir data/rlvr-simple-forward \\
        --op-min 20 --op-max 30 \\
        --output-dir data/rlvr/parquet \\
        --val-frac 0.02
"""

import argparse
import glob
import json
import os
import random
import re

import pandas as pd

ANSWER_RE = re.compile(r"Answer:\s*(-?\d+)")


def extract_ground_truth(solution: str):
    matches = ANSWER_RE.findall(solution)
    return matches[-1] if matches else None


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def to_record(item, idx, data_source):
    gt = extract_ground_truth(item["solution"])
    if gt is None:
        return None
    prompt = (item["problem"].strip() + " " + item["question"].strip()).strip()
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": {
            "index": idx,
            "op": item.get("op"),
            "mode": item.get("mode"),
        },
    }


def collect_files(root, op_min, op_max, max_files_per_op):
    files = []
    for op in range(op_min, op_max + 1):
        op_files = sorted(glob.glob(os.path.join(root, str(op), "*.jsonl")))
        files.extend(op_files[:max_files_per_op])
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing op subdirs of JSONL files.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--op-min", type=int, required=True)
    p.add_argument("--op-max", type=int, required=True)
    p.add_argument("--max-files-per-op", type=int, default=1)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-source", default="reasoning-compression")
    args = p.parse_args()

    files = collect_files(
        args.input_dir, args.op_min, args.op_max, args.max_files_per_op
    )
    if not files:
        raise SystemExit(
            f"No JSONL files found under {args.input_dir} for op {args.op_min}-{args.op_max}"
        )
    print(f"[prepare] {len(files)} JSONL files")

    records = []
    for path in files:
        for item in iter_jsonl(path):
            rec = to_record(item, idx=len(records), data_source=args.data_source)
            if rec is not None:
                records.append(rec)
    print(f"[prepare] {len(records)} records")

    random.Random(args.seed).shuffle(records)
    n_val = max(1, int(len(records) * args.val_frac))
    val, train = records[:n_val], records[n_val:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    pd.DataFrame(train).to_parquet(train_path)
    pd.DataFrame(val).to_parquet(val_path)
    print(f"[prepare] train={len(train)} -> {train_path}")
    print(f"[prepare] val  ={len(val)} -> {val_path}")


if __name__ == "__main__":
    main()
