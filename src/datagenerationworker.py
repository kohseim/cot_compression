"""Multiprocessing worker for generating arithmetic problems.

# CoT Compression (explicit + composed-nest/implicit-nest 2,3,4,5) from same chains:
    python datagenerationworker.py --numprocs 16 --total 40000 --mod 23 --number_range 5 \
        --target_length zero_context --d 2 --force \
        --listoperations 10 11 12 --output_dir /path/to/output --flat_output \
        --all_nests 2 3 4 5 --variant_types composed implicit

# CoT Order
    python datagenerationworker.py --numprocs 16 --opmax 3 --total 4000000 \
        --mod 23 --number_range 5 --target_length zero_context --d 2 --force \
        --listoperations 2 --output_dir /path/to/output --flat_output \
        --modes forward backward hierarchical
"""

import json
import os
import random
import argparse
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

from generator import generate_sample, generate_all_variants, string_to_number


def work_function(
    force,
    mod,
    number_range,
    listoperations,
    identifier,
    output_dir,
    modes,
    num_per_worker,
    flat_output,
    max_nest=1,
    all_nests=None,
    variant_types=("composed", "implicit"),
    include_rightforward=False,
):
    if all_nests is not None:
        _work_all_variants(
            force,
            mod,
            number_range,
            listoperations,
            identifier,
            output_dir,
            num_per_worker,
            flat_output,
            all_nests,
            variant_types,
            include_rightforward=include_rightforward,
        )
        return

    for mode in modes:
        # Prepare output file paths
        files = []
        for op in listoperations:
            if flat_output:
                dirname = "{}/{}/".format(output_dir, op)
            else:
                dirname = "{}/zero_context/{}/".format(output_dir, op)
            filename = dirname + "simple_op{}_force_{}_{}.jsonl".format(
                op, force, identifier
            )
            files.append(filename)

        items = [[] for _ in range(len(listoperations))]
        lines = 0

        np.random.seed(identifier)
        random.seed(identifier)

        pbar = tqdm(
            total=num_per_worker,
            desc="[proc{}] {}/op{}".format(identifier, mode, listoperations),
            position=identifier,
            leave=True,
        )

        while True:
            # Pick the op with fewest samples so far
            min_idx = min(range(len(listoperations)), key=lambda i: len(items[i]))
            target_op = listoperations[min_idx]

            try:
                problem, question, solution, op, id_val = generate_sample(
                    target_op, number_range, mod, mode, max_nest=max_nest
                )
            except Exception:
                continue

            for idx, ask_op in enumerate(listoperations):
                if op == ask_op:
                    item = {
                        "problem": problem,
                        "question": question,
                        "solution": solution,
                        "op": op,
                        "id": id_val,
                        "template": "simple_chain",
                        "mode": mode,
                        "length": "zero_context",
                        "d": 1,
                    }
                    items[idx].append(item)
                    break

            new_lines = min(len(items[i]) for i in range(len(listoperations)))
            if new_lines > lines:
                pbar.update(new_lines - lines)
                lines = new_lines
            if lines >= num_per_worker:
                break

        pbar.close()

        # Write output
        for idx, op in enumerate(listoperations):
            filename = files[idx]
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "a") as f:
                f.write("\n".join(json.dumps(item) for item in items[idx]) + "\n")


def _work_all_variants(
    force,
    mod,
    number_range,
    listoperations,
    identifier,
    output_dir,
    num_per_worker,
    flat_output,
    all_nests,
    variant_types=("composed", "implicit"),
    include_rightforward=False,
):
    """Generate explicit + {variant}-nest-K for all K from the same chains."""

    mode_names = ["explicit"]
    for n in all_nests:
        for vt in variant_types:
            mode_names.append("{}-nest{}".format(vt, n))
    if include_rightforward:
        mode_names.append("rightforward")

    # Prepare output file paths
    files = []
    for op in listoperations:
        if flat_output:
            dirname = "{}/{}/".format(output_dir, op)
        else:
            dirname = "{}/zero_context/{}/".format(output_dir, op)
        filename = dirname + "simple_op{}_force_{}_{}.jsonl".format(
            op, force, identifier
        )
        files.append(filename)

    # items[op_idx] = list of items (all modes interleaved)
    items = [[] for _ in range(len(listoperations))]
    # Track count per op (each chain produces one count regardless of num variants)
    counts = [0] * len(listoperations)
    lines = 0

    np.random.seed(identifier)
    random.seed(identifier)

    pbar = tqdm(
        total=num_per_worker,
        desc="[proc{}] all_variants/op{}".format(identifier, listoperations),
        position=identifier,
        leave=True,
    )

    while True:
        min_idx = min(range(len(listoperations)), key=lambda i: counts[i])
        target_op = listoperations[min_idx]

        try:
            problem, question, op, results = generate_all_variants(
                target_op,
                number_range,
                mod,
                max_nests=tuple(all_nests),
                variant_types=tuple(variant_types),
                include_rightforward=include_rightforward,
            )
        except Exception:
            continue

        for idx, ask_op in enumerate(listoperations):
            if op == ask_op:
                for mode_name in mode_names:
                    sol, id_val = results[mode_name]
                    item = {
                        "problem": problem,
                        "question": question,
                        "solution": sol,
                        "op": op,
                        "id": id_val,
                        "template": "simple_chain",
                        "mode": mode_name,
                        "length": "zero_context",
                        "d": 1,
                    }
                    items[idx].append(item)
                counts[idx] += 1
                break

        new_lines = min(counts)
        if new_lines > lines:
            pbar.update(new_lines - lines)
            lines = new_lines
        if lines >= num_per_worker:
            break

    pbar.close()

    # Write output
    for idx, op in enumerate(listoperations):
        filename = files[idx]
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "a") as f:
            f.write("\n".join(json.dumps(item) for item in items[idx]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate simple chain DAG arithmetic problems"
    )
    parser.add_argument("--numprocs", type=int, default=1)
    parser.add_argument(
        "--opmax",
        type=int,
        default=15,
        help="Accepted for CLI compatibility (not used)",
    )
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--mod", type=int, default=23)
    parser.add_argument("--number_range", type=int, default=5)
    parser.add_argument(
        "--target_length",
        type=str,
        default="zero_context",
        help="Accepted for CLI compatibility (only zero_context supported)",
    )
    parser.add_argument(
        "--d", type=int, default=2, help="Accepted for CLI compatibility (not used)"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--listoperations", nargs="+", type=int, default=[4])
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--flat_output", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        type=str,
        default=["forward"],
        choices=["forward", "backward", "hierarchical"],
    )
    parser.add_argument(
        "--max_nest",
        type=int,
        default=1,
        help="Max operations inlined per composed step (default: 1)",
    )
    parser.add_argument(
        "--all_nests",
        nargs="+",
        type=int,
        default=None,
        help="Generate all variants (explicit + {variant}-nest-K "
        "for each K) from the same chains. Overrides --modes.",
    )
    parser.add_argument(
        "--variant_types",
        nargs="+",
        type=str,
        default=["composed", "implicit"],
        choices=["composed", "implicit"],
        help="Variant types to generate with --all_nests.",
    )
    parser.add_argument(
        "--include_rightforward",
        action="store_true",
        help="With --all_nests, also emit rightforward "
        "(right-associative) solutions alongside "
        "explicit and the nested variants.",
    )
    args = parser.parse_args()
    print(args)

    num_per_worker = (args.total + args.numprocs - 1) // args.numprocs

    processes = []
    for i in range(args.numprocs):
        p = mp.Process(
            target=work_function,
            args=(
                args.force,
                args.mod,
                args.number_range,
                args.listoperations,
                i,
                args.output_dir,
                args.modes,
                num_per_worker,
                args.flat_output,
                args.max_nest,
                args.all_nests,
                tuple(args.variant_types),
                args.include_rightforward,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("processes joined")
