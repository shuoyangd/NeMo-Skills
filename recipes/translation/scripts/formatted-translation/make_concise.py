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
Produce a concise version of an input JSONL by keeping only the specified fields,
stamp each output record with a _translation_src_id derived from the original,
and metadata.dataset as _translation_dataset_id when present.

Fields prefixed with _translation_ are pipeline metadata. Downstream stages
carry them outside the model-visible source payload.

The ID is a SHA-256 hash (first 16 hex chars) of the canonically serialized
original record (sorted keys, no extra whitespace).  Two records with identical
content always get the same ID, which is fine — their translations will be
identical too.  The ID is used by downstream steps (filter_and_merge) to join
translations back to original records without relying on line alignment.

Usage:
    python make_concise.py --input data.jsonl --output concise.jsonl --fields problem generation
    python make_concise.py --input data.jsonl --output concise.jsonl --fields problem generation --from-messages
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def compute_src_id(record: dict) -> str:
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def extract_from_messages(record: dict, fields: list[str]) -> dict:
    """Extract fields by position from the messages array.

    fields[0] <- messages[0]["content"], fields[1] <- messages[1]["content"], etc.
    """
    messages = record.get("messages", [])
    concise = {}
    for i, field in enumerate(fields):
        if i < len(messages):
            concise[field] = messages[i].get("content", "")
        else:
            concise[field] = ""
    return concise


def make_concise_file(
    input_file: str,
    output_file: str,
    fields: list[str],
    from_messages: bool = False,
    add_src_id: bool = True,
    verbose: bool = False,
):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    total = skipped = 0
    with open(input_file, encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line_num, raw_line in enumerate(fin, 1):
            line = raw_line.strip()
            if not line:
                continue

            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                if verbose:
                    print(f"Line {line_num}: skipped (invalid JSON)", file=sys.stderr)
                continue

            if from_messages:
                concise = extract_from_messages(record, fields)
            else:
                missing = [k for k in fields if k not in record]
                if missing and verbose:
                    print(f"Line {line_num}: missing fields {missing}", file=sys.stderr)
                concise = {k: record[k] for k in fields if k in record}

            if add_src_id:
                concise["_translation_src_id"] = compute_src_id(record)

            metadata = record.get("metadata")
            if isinstance(metadata, dict) and "dataset" in metadata:
                concise["_translation_dataset_id"] = metadata["dataset"]

            fout.write(json.dumps(concise, ensure_ascii=False) + "\n")

    if skipped:
        print(f"Warning: skipped {skipped}/{total} lines (invalid JSON).", file=sys.stderr)
    print(f"Wrote {total - skipped} concise records to '{output_file}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Keep only specified fields from each JSON object in a JSONL file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument(
        "--fields",
        nargs="+",
        required=True,
        metavar="FIELD",
        help="Top-level fields to keep (e.g. --fields problem generation)",
    )
    parser.add_argument(
        "--from-messages",
        action="store_true",
        help="Extract fields by position from the messages array instead of top-level keys",
    )
    parser.add_argument(
        "--no-src-id",
        action="store_true",
        help="Do not add _translation_src_id to output records",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-line warnings to stderr")
    args = parser.parse_args()

    make_concise_file(
        args.input,
        args.output,
        args.fields,
        from_messages=args.from_messages,
        add_src_id=not args.no_src_id,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
