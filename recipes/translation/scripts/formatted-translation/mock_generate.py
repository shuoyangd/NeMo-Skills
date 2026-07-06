#!/usr/bin/env python3
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

"""
Mock generation step for testing the formatted translation pipeline locally.

Reads a wrapped JSONL file (produced by wrap_sft_data.py) and produces a
fake generation output by prepending '[TRANSLATED] ' to every string value
in the src JSON.  Output mimics the real generate step: one line per input,
with a `generation` field containing the translated JSON wrapped in ```json```.

Usage:
    python mock_generate.py --input wrapped.jsonl --output output-rs0.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def translate_value(v, target_lang):
    """Recursively prepend '[Translateed to target_lang] ' to all string leaf values."""
    if isinstance(v, str):
        return f"[Translated to {target_lang}] " + v
    if isinstance(v, list):
        return [translate_value(item, target_lang) for item in v]
    if isinstance(v, dict):
        return {k: translate_value(val, target_lang) for k, val in v.items()}
    return v


def mock_generate_file(input_file: str, output_file: str):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            src = json.loads(row["src"])
            translated = translate_value(src, row["target_lang"])
            generation = "```json\n" + json.dumps(translated, ensure_ascii=False) + "\n```"
            fout.write(json.dumps({**row, "generation": generation}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Mock LLM generation for pipeline testing.")
    parser.add_argument("--input", required=True, help="Wrapped JSONL input file")
    parser.add_argument("--output", required=True, help="Output JSONL file (mimics generate output)")
    args = parser.parse_args()

    mock_generate_file(args.input, args.output)


if __name__ == "__main__":
    sys.exit(main())
