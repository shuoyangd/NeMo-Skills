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

"""Separate a jsonl file into target and non-target subsets based on metadata.domain regex match.

Supports parallel processing via -j/--jobs to handle large files efficiently.
Lines are dispatched to a worker pool in batches for JSON parsing and regex matching.
"""

import argparse
import json
import multiprocessing
import re
import sys
from pathlib import Path


def process_batch(args):
    lines, pattern_str = args
    pattern = re.compile(pattern_str)
    target = []
    non_target = []
    n_malformed = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            instance = json.loads(line)
        except json.JSONDecodeError:
            n_malformed += 1
            continue
        domain = instance.get("metadata", {}).get("domain", None)
        if domain is not None and bool(pattern.search(domain)):
            target.append(line)
        else:
            non_target.append(line)
    return target, non_target, n_malformed


def main():
    parser = argparse.ArgumentParser(description="Split a jsonl file into target/non-target by metadata.domain regex.")
    parser.add_argument("--input_file", required=True, help="Path to input .jsonl file")
    parser.add_argument("--output_dir", required=True, help="Directory to write target.jsonl and non_target.jsonl")
    parser.add_argument(
        "--domain_regex",
        required=True,
        help="Regex pattern matched against metadata.domain. Instances that match go to target.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel workers (default: cpu count)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=500,
        help="Number of lines per batch dispatched to each worker (default: 500)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_path = output_dir / "target.jsonl"
    non_target_path = output_dir / "non_target.jsonl"

    n_target = 0
    n_non_target = 0
    n_malformed = 0

    def batch_iter():
        with open(args.input_file) as fin:
            batch = []
            for line in fin:
                batch.append(line)
                if len(batch) >= args.batch:
                    yield (batch, args.domain_regex)
                    batch = []
            if batch:
                yield (batch, args.domain_regex)

    pool = multiprocessing.Pool(processes=args.jobs)
    try:
        with (
            open(target_path, "w") as f_target,
            open(non_target_path, "w") as f_non_target,
        ):
            for target_lines, non_target_lines, n_bad in pool.imap(process_batch, batch_iter()):
                for line in target_lines:
                    f_target.write(line + "\n")
                for line in non_target_lines:
                    f_non_target.write(line + "\n")
                n_target += len(target_lines)
                n_non_target += len(non_target_lines)
                n_malformed += n_bad
    finally:
        pool.close()
        pool.join()

    total = n_target + n_non_target
    print(f"Total instances : {total}")
    print(f"Target          : {n_target} ({100 * n_target / total:.1f}%)" if total else "Target: 0")
    print(f"Non-target      : {n_non_target} ({100 * n_non_target / total:.1f}%)" if total else "Non-target: 0")
    if n_malformed:
        print(f"Malformed lines : {n_malformed}", file=sys.stderr)
    print(f"Written to      : {output_dir}")


if __name__ == "__main__":
    main()
