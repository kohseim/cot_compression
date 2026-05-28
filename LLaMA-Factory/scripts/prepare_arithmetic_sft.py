"""Prepare alpaca-format SFT data for LLaMA-Factory from datagenerationworker.py output.

The worker emits jsonl with the schema::

    {"problem", "question", "solution", "op", "id",
     "template", "mode", "length", "d"}

This script:
  1. Walks `{data_root}/<op>/*.jsonl` (also tolerates the nested
     `{data_root}/zero_context/<op>/*.jsonl` layout).
  2. Filters records by op range and (optionally) mode.
  3. Writes alpaca-format jsonl per mode under `LLaMA-Factory/data/arithmetic_sft/`:
         {"instruction": "<problem> <question>", "output": "<solution>"}
  4. With ``--register``, updates ``LLaMA-Factory/data/dataset_info.json`` with
     one entry per produced mode, named ``<prefix>_<mode>``.

Example::

    python LLaMA-Factory/scripts/prepare_arithmetic_sft.py \
        --data_root /path/to/datagen_output \
        --op_min 10 --op_max 12 \
        --modes explicit composed-nest2 \
        --register
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LF_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = LF_DIR / "data" / "arithmetic_sft"
DEFAULT_DATASET_INFO = LF_DIR / "data" / "dataset_info.json"
DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant that performs sequential arithmetic tasks, where all calculations must be done modulo 23."
)


def find_input_files(data_root: Path, op_min: int, op_max: int):
    files = []
    for op in range(op_min, op_max + 1):
        for base in (data_root / str(op), data_root / "zero_context" / str(op)):
            if base.is_dir():
                for p in sorted(base.glob("*.jsonl")):
                    files.append((op, p))
    return files


def safe_mode_slug(mode: str) -> str:
    # dataset_info keys must be filename-friendly; keep mode mostly verbatim
    # but normalise the hyphen variant separator just in case.
    return mode.replace("/", "_")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--op_min", type=int, required=True)
    parser.add_argument("--op_max", type=int, required=True)
    parser.add_argument("--modes", nargs="+", default=None, help="Filter to these modes (default: all encountered)")
    parser.add_argument("--max_samples_per_mode", type=int, default=None)
    parser.add_argument("--register", action="store_true", help="Write entries into dataset_info.json")
    parser.add_argument("--dataset_info_path", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--dataset_prefix", default="arithmetic_sft")
    parser.add_argument("--overwrite", action="store_true", help="Truncate per-mode output files before writing")
    parser.add_argument(
        "--system_prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt attached to every record"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode_filter = set(args.modes) if args.modes else None

    counts: dict[str, int] = defaultdict(int)
    skipped = 0
    writers: dict[str, "object"] = {}

    def get_writer(mode: str):
        if mode not in writers:
            path = args.output_dir / f"{safe_mode_slug(mode)}.jsonl"
            writers[mode] = open(path, "w" if args.overwrite else "a")
        return writers[mode]

    files = find_input_files(args.data_root, args.op_min, args.op_max)
    if not files:
        raise SystemExit(f"No input files under {args.data_root} for op {args.op_min}-{args.op_max}")
    print(f"Scanning {len(files)} input files...")

    for op, path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                mode = obj.get("mode", "unknown")
                if mode_filter and mode not in mode_filter:
                    continue
                if args.max_samples_per_mode is not None and counts[mode] >= args.max_samples_per_mode:
                    continue
                problem = (obj.get("problem") or "").strip()
                question = (obj.get("question") or "").strip()
                solution = (obj.get("solution") or "").strip()
                if not (problem and question and solution):
                    skipped += 1
                    continue
                rec = {
                    "instruction": f"{problem} {question}",
                    "output": solution,
                    "system": args.system_prompt,
                }
                get_writer(mode).write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[mode] += 1

    for w in writers.values():
        w.close()

    print("Samples written per mode:")
    for mode, n in sorted(counts.items()):
        print(f"  {mode}: {n}")
    if skipped:
        print(f"Skipped {skipped} malformed/empty records")

    if args.register and counts:
        info = {}
        if args.dataset_info_path.exists():
            info = json.loads(args.dataset_info_path.read_text() or "{}")
        info_dir = args.dataset_info_path.parent.resolve()
        for mode in counts:
            data_path = args.output_dir.resolve() / f"{safe_mode_slug(mode)}.jsonl"
            try:
                rel = data_path.relative_to(info_dir)
                file_name = str(rel)
            except ValueError:
                file_name = str(data_path)
            entry_name = f"{args.dataset_prefix}_{safe_mode_slug(mode)}"
            info[entry_name] = {
                "file_name": file_name,
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "response": "output",
                    "system": "system",
                },
            }
        args.dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
        args.dataset_info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
        print(f"Updated {args.dataset_info_path} ({len(counts)} entries)")


if __name__ == "__main__":
    main()
