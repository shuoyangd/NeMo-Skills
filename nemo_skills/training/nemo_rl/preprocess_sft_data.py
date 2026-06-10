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

import argparse
import glob
import os

from nemo_skills.training.nemo_rl.sft_data import (
    get_default_cache_dir,
    load_or_process_prompt_response_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess SFT JSONL into a Hugging Face dataset directory.")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Raw SFT JSONL file with either input/output or messages fields. Can be repeated.",
    )
    parser.add_argument(
        "--input-glob",
        action="append",
        default=[],
        help="Glob for raw SFT JSONL shards. Quote the glob so it expands on the run host.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output dataset directory. Defaults to <input parent>/.cache/<split-name>_<input stem>.",
    )
    parser.add_argument("--split-name", default="train", help="Name used for the default cache path.")
    parser.add_argument("--input-key", default="input", help="Input field name for input/output formatted data.")
    parser.add_argument("--output-key", default="output", help="Output field name for input/output formatted data.")
    parser.add_argument(
        "--num-proc",
        type=int,
        default=os.cpu_count() or 8,
        help="Number of processes for JSON parsing, mapping, and saving.",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Ignore an existing matching signature and rebuild the output dataset.",
    )
    parser.add_argument(
        "--max-shard-size",
        default=None,
        help='Optional shard size passed to save_to_disk, e.g. "1GB" or "500MB".',
    )
    return parser.parse_args()


def resolve_inputs(inputs: list[str], input_globs: list[str]) -> list[str]:
    resolved = list(inputs)
    for pattern in input_globs:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f"No files matched --input-glob pattern: {pattern}")
        resolved.extend(matches)
    if not resolved:
        raise ValueError("Provide at least one --input or --input-glob")
    return resolved


def main() -> None:
    args = parse_args()
    inputs = resolve_inputs(args.input, args.input_glob)
    output = args.output or str(get_default_cache_dir(inputs[0], args.split_name))

    dataset = load_or_process_prompt_response_split(
        inputs,
        args.split_name,
        input_key=args.input_key,
        output_key=args.output_key,
        num_proc=args.num_proc,
        force_reprocess=args.force_reprocess,
        cache_dir=output,
        max_shard_size=args.max_shard_size,
    )
    print(f"Preprocessed {len(dataset)} records into: {output}")


if __name__ == "__main__":
    main()
