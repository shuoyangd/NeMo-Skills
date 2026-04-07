# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Remix target and non-target jsonl files at one or more target ratios.

For each requested ratio r (fraction of target instances in the final mix):
  - Computes the largest feasible total that can be satisfied from the available
    data without replacement: total = min(len(target)/r, len(non_target)/(1-r))
  - Samples target_count = round(total * r) from target
  - Samples non_target_count = total - target_count from non-target
  - Writes output to <output_dir>/remix_r<ratio>.jsonl

Memory-efficient: only the sampled lines are held in memory, not the full files.
Multiple ratio outputs are written in parallel via -j/--jobs.

NOTE on output ordering: the output is NOT a uniform shuffle of the combined lines.
Within each source (target / non-target), lines appear in their original file order.
What is randomized is (a) which lines are selected and (b) how the two streams are
interleaved. This is sufficient for training data purposes but callers that require
a true uniform shuffle should post-process the output.
"""

import argparse
import math
import multiprocessing
import random
from pathlib import Path


def count_lines(path):
    """Count non-empty lines in a file without loading it into memory."""
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def stream_sample(path, indices):
    """Generator: stream through file, yielding only lines at the given sorted indices."""
    idx_iter = iter(indices)
    next_idx = next(idx_iter, None)
    if next_idx is None:
        return

    line_num = 0
    with open(path) as f:
        for raw in f:
            if not raw.strip():
                continue
            if line_num == next_idx:
                yield raw.strip()
                next_idx = next(idx_iter, None)
                if next_idx is None:
                    break
            line_num += 1


def write_remix(args):
    """Worker: sample and write one remixed output file for a given ratio.

    Memory-efficient: builds a shuffled interleaving plan (list of booleans) and
    streams target/non-target lines one at a time following that plan, so only one
    line is held in memory at a time.
    """
    target_file, non_target_file, n_target, n_non_target, ratio, output_path, seed = args

    if not (0.0 < ratio < 1.0):
        raise ValueError(f"ratio must be strictly between 0 and 1, got {ratio}")

    # Largest total achievable without replacement from both files
    max_total = min(n_target / ratio, n_non_target / (1.0 - ratio))
    total = math.floor(max_total)

    target_count = round(total * ratio)
    non_target_count = total - target_count

    rng = random.Random(seed)
    target_indices = sorted(rng.sample(range(n_target), target_count))
    non_target_indices = sorted(rng.sample(range(n_non_target), non_target_count))

    # Shuffled interleaving plan: True = take next from target, False = from non-target.
    # This is just a list of booleans — negligible memory.
    plan = [True] * target_count + [False] * non_target_count
    rng.shuffle(plan)

    target_stream = stream_sample(target_file, target_indices)
    non_target_stream = stream_sample(non_target_file, non_target_indices)

    with open(output_path, "w") as f:
        for take_target in plan:
            line = next(target_stream) if take_target else next(non_target_stream)
            f.write(line + "\n")

    return ratio, target_count, non_target_count, total


def main():
    parser = argparse.ArgumentParser(description="Remix target and non-target datasets at given ratio(s).")
    parser.add_argument("--target_file", required=True, help="Path to target.jsonl")
    parser.add_argument("--non_target_file", required=True, help="Path to non_target.jsonl")
    parser.add_argument("--output_dir", required=True, help="Directory to write remixed output files")
    parser.add_argument(
        "--target_ratio",
        required=True,
        help="Target fraction(s) in the final mix. Single float or comma-separated list, e.g. '0.3' or '0.2,0.3,0.5'.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel workers for writing multiple ratio files (default: cpu count)",
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=500, help="Unused; kept for interface consistency with other scripts"
    )
    args = parser.parse_args()

    ratios = [float(r.strip()) for r in args.target_ratio.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Counting lines in target      : {args.target_file}")
    n_target = count_lines(args.target_file)
    print(f"Counting lines in non-target  : {args.non_target_file}")
    n_non_target = count_lines(args.non_target_file)
    print(f"Target size         : {n_target}")
    print(f"Non-target size     : {n_non_target}")

    worker_args = []
    for ratio in ratios:
        ratio_str = f"{ratio:.4f}".rstrip("0").rstrip(".")
        out_path = output_dir / f"remix_r{ratio_str}.jsonl"
        worker_args.append(
            (args.target_file, args.non_target_file, n_target, n_non_target, ratio, out_path, args.seed)
        )

    n_workers = min(args.jobs, len(ratios))
    pool = multiprocessing.Pool(processes=n_workers)
    try:
        async_results = [pool.apply_async(write_remix, (wa,)) for wa in worker_args]
        for res in async_results:
            ratio, target_count, non_target_count, total = res.get()
            print(f"ratio={ratio:.4f} -> target={target_count}, non_target={non_target_count}, total={total}")
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
