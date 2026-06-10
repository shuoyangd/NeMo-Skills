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

import json
import os
from pathlib import Path
from typing import Any, Sequence

from datasets import Dataset, load_dataset, load_from_disk


def is_hf_dataset_dir(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() and (path / "dataset_info.json").exists() and (path / "state.json").exists()


def get_default_cache_dir(data_path: str | Path, split_name: str) -> Path:
    data_path = Path(data_path)
    return data_path.parent / ".cache" / f"{split_name}_{data_path.stem}"


def get_default_num_proc(num_proc: int | None = None) -> int:
    if num_proc is not None:
        return num_proc
    cpu_count = os.cpu_count() or 2
    return min(8, cpu_count)


def get_source_signature(data_paths: str | Path | Sequence[str | Path]) -> dict[str, Any]:
    paths = [Path(data_paths)] if isinstance(data_paths, (str, Path)) else [Path(path) for path in data_paths]
    files = []
    total_size = 0
    for path in paths:
        stat = path.stat()
        total_size += stat.st_size
        files.append(
            {
                "source_path": str(path),
                "size": str(stat.st_size),
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "source_files": files,
        "size": str(total_size),
    }


def source_signature_matches(old_sig: dict[str, Any], new_sig: dict[str, Any]) -> bool:
    if old_sig.get("size") != new_sig["size"]:
        return False
    if "source_files" in old_sig or "source_files" in new_sig:
        return old_sig.get("source_files") == new_sig.get("source_files")
    # Keep compatibility with older signature files that only recorded size.
    if "mtime_ns" in old_sig and old_sig["mtime_ns"] != new_sig["mtime_ns"]:
        return False
    return True


def _get_first_record_from_jsonl(data_path: str | Path) -> dict[str, Any]:
    with open(data_path, "r") as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError(f"Dataset at {data_path} is empty")
        return json.loads(first_line)


def _get_first_record_from_hf_dataset(data_path: str | Path) -> dict[str, Any]:
    dataset = load_from_disk(str(data_path))
    if len(dataset) == 0:
        raise ValueError(f"Dataset at {data_path} is empty")
    return dataset[0]


def detect_data_format(data_path: str | Sequence[str]) -> str:
    """Detect SFT dataset format from either raw JSONL or save_to_disk output."""
    try:
        if isinstance(data_path, Sequence) and not isinstance(data_path, str):
            if not data_path:
                raise ValueError("No dataset files provided")
            data_path = data_path[0]
        path = Path(data_path)
        sig_file = path / "signature.json"
        if is_hf_dataset_dir(path) and sig_file.exists():
            with open(sig_file) as f:
                data_format = json.load(f).get("data_format")
            if data_format in {"input_output", "messages", "mixed"}:
                return data_format

        sample = _get_first_record_from_hf_dataset(path) if is_hf_dataset_dir(path) else _get_first_record_from_jsonl(path)

        has_input_output = "input" in sample and "output" in sample
        has_messages = "messages" in sample

        if has_input_output and has_messages:
            return "mixed"
        if has_input_output:
            return "input_output"
        if has_messages:
            return "messages"

        raise ValueError(
            f"Dataset at {data_path} has neither 'input'/'output' keys nor 'messages' key. "
            f"Available keys: {list(sample.keys())}"
        )
    except FileNotFoundError:
        raise ValueError(f"Dataset file not found: {data_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in dataset file {data_path}: {e}")


def add_messages_key(
    examples: dict[str, list[Any]],
    input_key: str,
    output_key: str,
) -> dict[str, list[list[dict[str, Any]]]]:
    return {
        "messages": [
            [
                {"role": "user", "content": input_},
                {"role": "assistant", "content": output},
            ]
            for input_, output in zip(examples[input_key], examples[output_key])
        ]
    }


def load_or_process_prompt_response_split(
    path: str | Sequence[str],
    split_name: str,
    input_key: str = "input",
    output_key: str = "output",
    num_proc: int | None = None,
    force_reprocess: bool = False,
    cache_dir: str | Path | None = None,
    max_shard_size: str | int | None = None,
) -> Dataset:
    """Load raw SFT JSONL as a Dataset, caching the Arrow result for reuse."""
    paths = [Path(path)] if isinstance(path, str) else [Path(item) for item in path]
    if not paths:
        raise ValueError("No dataset files provided")
    data_path = paths[0]
    num_proc = get_default_num_proc(num_proc)

    if len(paths) == 1 and is_hf_dataset_dir(data_path):
        print(f"[Cache] Loading preprocessed {split_name} dataset from: {data_path}")
        return load_from_disk(str(data_path))

    for data_path in paths:
        if not data_path.exists():
            raise ValueError(f"Dataset file not found: {data_path}")
        if not data_path.is_file():
            raise ValueError(f"Expected JSONL file, got: {data_path}")

    cache_dir = Path(cache_dir) if cache_dir is not None else get_default_cache_dir(paths[0], split_name)
    sig_file = cache_dir / "signature.json"
    source_sig = get_source_signature(paths)

    if cache_dir.exists() and sig_file.exists() and not force_reprocess:
        with open(sig_file) as f:
            old_sig = json.load(f)
        if source_signature_matches(old_sig, source_sig):
            print(f"[Cache] Loading {split_name} dataset from: {cache_dir}")
            return load_from_disk(str(cache_dir))
        print(f"[Cache] Invalidated (source signature changed): {path}")

    data_format = detect_data_format([str(item) for item in paths])

    print(f"[Map] Processing {split_name} dataset from: {path}")
    data_files = str(paths[0]) if len(paths) == 1 else [str(item) for item in paths]
    dataset = load_dataset("json", data_files=data_files, split="train", num_proc=num_proc)

    if "messages" not in dataset.column_names:
        dataset = dataset.map(
            add_messages_key,
            batched=True,
            fn_kwargs={"input_key": input_key, "output_key": output_key},
            num_proc=num_proc,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_dir), max_shard_size=max_shard_size, num_proc=num_proc)
    source_sig["data_format"] = data_format
    with open(sig_file, "w") as f:
        json.dump(source_sig, f)

    print(f"[Cache] Saved {split_name} dataset to: {cache_dir}")
    return dataset
