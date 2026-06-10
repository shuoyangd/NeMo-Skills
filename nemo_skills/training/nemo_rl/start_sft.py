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

# copied from https://github.com/NVIDIA/NeMo-RL/blob/main/examples/run_sft.py

import argparse
import os
import pprint
from functools import partial
from typing import Any, Dict, Optional

from datasets import Dataset
from nemo_rl.algorithms.sft import MasterConfig, setup, sft_train
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_skills.training.nemo_rl.sft_data import detect_data_format, load_or_process_prompt_response_split
from nemo_skills.utils import setup_make_sequence_length_divisible_by

TokenizerType = PreTrainedTokenizerBase
_call_counter = 0


class PromptResponseDataset:
    def __init__(
        self,
        train_ds_path: str,
        val_ds_path: str | None = None,
        input_key: str = "input",
        output_key: str = "output",
        num_proc: int | None = None,
        force_reprocess: bool = False,
        train_cache_path: str | None = None,
        val_cache_path: str | None = None,
        max_shard_size: str | int | None = None,
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.force_reprocess = force_reprocess

        # Auto-determine number of processes
        if num_proc is None:
            cpu_count = os.cpu_count() or 2
            self.num_proc = min(8, cpu_count)
        else:
            self.num_proc = num_proc

        # Train split
        self.formatted_ds = {
            "train": self.load_or_process_split(train_ds_path, "train", train_cache_path, max_shard_size),
        }
        # Validation split (optional)
        if val_ds_path:
            self.formatted_ds["validation"] = self.load_or_process_split(val_ds_path, "val", val_cache_path, max_shard_size)
        else:
            self.formatted_ds["validation"] = None

        self.task_spec = TaskDataSpec("json_dataset")

    def load_or_process_split(
        self,
        path: str,
        split_name: str,
        cache_path: str | None = None,
        max_shard_size: str | int | None = None,
    ) -> Dataset:
        return load_or_process_prompt_response_split(
            path,
            split_name,
            input_key=self.input_key,
            output_key=self.output_key,
            num_proc=self.num_proc,
            force_reprocess=self.force_reprocess,
            cache_dir=cache_path,
            max_shard_size=max_shard_size,
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SFT training with configuration")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


# =======================================================
# Data Processing
# =======================================================
def sft_preprocessor(
    datum_dict: Dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
    add_bos: bool = True,
    add_eos: bool = True,
    add_generation_prompt: bool = False,
) -> DatumSpec:
    """Process a datum dictionary for SFT training."""

    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_bos_token=add_bos,
        add_eos_token=add_eos,
        add_generation_prompt=add_generation_prompt,
    )

    # ==================== START: BLOCK FOR DEBUGGING ====================
    global _call_counter
    if _call_counter < 1:  # Only print for the first 3 samples
        print(f"\n--- 🐛 Debugging invocation #{_call_counter + 1} (for original sample idx: {idx}) ---")
        # Loop through up to the first 3 messages in the log
        for i, message in enumerate(message_log[:3]):
            print(f"  Message [{i}]:")
            print(f"    Role    : {message['role']}")
            print(f"    Content : {message['content']}")
            print(f"    token-ids : {message['token_ids'].tolist()}")
        print("----------------------------------------------------\n")
        _call_counter += 1
    # ===================== END: BLOCK FOR DEBUGGING =====================

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for message in message_log:
            message["token_ids"] = message["token_ids"][: min(4, max_seq_length // len(message_log))]
        loss_multiplier = 0.0

    output = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    return output


def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig):
    print("\n▶ Setting up data...")
    assert data_config["dataset_name"] == "prompt_response_dataset"
    data = PromptResponseDataset(
        data_config["train_data_path"],
        data_config.get("val_data_path"),
        data_config["input_key"],
        data_config["output_key"],
        num_proc=data_config.get("preprocessing_num_proc"),
        force_reprocess=data_config.get("force_reprocess", False),
        train_cache_path=data_config.get("train_cache_path"),
        val_cache_path=data_config.get("val_cache_path"),
        max_shard_size=data_config.get("preprocessing_max_shard_size"),
    )
    print(f"  ✓ Training dataset loaded with {len(data.formatted_ds['train'])} samples.")
    if data.formatted_ds["validation"] is not None:
        print(f"  ✓ Validation dataset loaded with {len(data.formatted_ds['validation'])} samples.")
    else:
        print("  ⚠ No validation dataset provided.")

    train_dataset = data.formatted_ds["train"]
    sft_task_spec = data.task_spec

    train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        sft_task_spec,
        partial(
            sft_preprocessor,
            add_bos=data_config["add_bos"],
            add_eos=data_config["add_eos"],
            add_generation_prompt=data_config["add_generation_prompt"],
        ),
        max_seq_length=data_config["max_input_seq_length"],
    )
    val_dataset: Optional[AllTaskProcessedDataset] = None
    if data.formatted_ds["validation"] is not None:
        val_dataset = AllTaskProcessedDataset(
            data.formatted_ds["validation"],
            tokenizer,
            sft_task_spec,
            partial(
                sft_preprocessor,
                add_bos=data_config["add_bos"],
                add_eos=data_config["add_eos"],
                add_generation_prompt=data_config["add_generation_prompt"],
            ),
            max_seq_length=data_config["max_input_seq_length"],
        )

    return train_dataset, val_dataset, sft_task_spec


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "sft.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
    if config["policy"]["make_sequence_length_divisible_by"] is None:
        tp = config["policy"]["tensor_model_parallel_size"]
        cp = config["policy"]["context_parallel_size"]
        config["policy"]["make_sequence_length_divisible_by"] = setup_make_sequence_length_divisible_by(tp, cp)
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Handle chat template inference from data format
    tokenizer_config = config["policy"]["tokenizer"]
    if tokenizer_config.get("chat_template") == "infer_from_data":
        print("Inferring chat template from data format...")

        # Detect data format from training data
        data_format = detect_data_format(config["data"]["train_data_path"])
        print(f"Detected data format: {data_format}")

        if data_format == "mixed":
            raise ValueError(
                "Dataset contains both 'input'/'output' and 'messages' keys. "
                "Please use a consistent data format or manually specify the chat_template."
            )
        elif data_format == "input_output":
            print("Setting chat_template to None (passthrough) for input/output format")
            tokenizer_config["chat_template"] = None
        elif data_format == "messages":
            print("Setting chat_template to 'default' for messages format")
            tokenizer_config["chat_template"] = "default"

        # Check validation data format if it exists
        if config["data"].get("val_data_path"):
            val_data_format = detect_data_format(config["data"]["val_data_path"])
            if val_data_format != data_format:
                raise ValueError(
                    f"Training data format ({data_format}) doesn't match validation data format ({val_data_format}). "
                    "Both datasets must use the same format."
                )
            print(f"Validation data format matches training data: {val_data_format}")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(f"Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")
    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    # setup data
    (
        dataset,
        val_dataset,
        sft_task_spec,
    ) = setup_data(tokenizer, config["data"])

    (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        sft_save_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)
    sft_train(
        policy,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        sft_task_spec,
        checkpointer,
        sft_save_state,
    )


if __name__ == "__main__":
    main()
