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
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig

# Add parent directory to sys.path to import nemo_skills
sys.path.append(str(Path(__file__).parent.parent.parent))

from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, nested_dataclass
from nemo_skills.utils import get_help_message, setup_logging


class ForcePrefixLogitsProcessor:
    """
    This processor forces the generation to start with a specific prefix.
    """

    def __init__(self, prefix_str: str, tokenizer):
        # Import vLLM LogitsProcessor at runtime (when actually needed)
        from vllm.logits_processor import LogitsProcessor

        # Make this class inherit from LogitsProcessor at runtime
        self.__class__.__bases__ = (LogitsProcessor,)
        super().__init__()
        # Tokenize the prefix and store the token IDs
        self.prefix_token_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
        self.prefix_len = len(self.prefix_token_ids)

    def __call__(self, token_ids: List[int], logits):
        import torch

        current_len = len(token_ids)

        # Check if the generation is still within the prefix length
        if current_len < self.prefix_len:
            # Get the next required token ID from our prefix
            next_token_id = self.prefix_token_ids[current_len]

            # Create a mask to suppress all tokens except the one we want
            # We set all logits to a very low number (-inf)
            mask = torch.full_like(logits, -float("inf"))

            # Set the logit for our desired next token to 0.0, making it
            # the only possible choice after softmax.
            mask[next_token_id] = 0.0

            # Apply the mask
            return logits + mask

        # If we are past the prefix, do nothing and return original logits
        return logits


@nested_dataclass(kw_only=True)
class ReasoningWithForcedLangConfig(GenerateSolutionsConfig):
    """Configuration for reasoning with forced language generation task."""

    # Language and prefix parameters
    language_code: str = "en"  # Language code to use from lang_libs.py
    enable_forced_prefix: bool = True  # Whether to enable forced prefix generation
    forced_prefix: str | None = None  # Override prefix (if None, will load from lang_libs.py)

    def _get_disallowed_params(self):
        """Override to specify parameters that should not be changed from defaults."""
        return super()._get_disallowed_params()


class ReasoningWithForcedLangTask(GenerationTask):
    """Custom generation task for reasoning with forced language."""

    @classmethod
    def get_generation_default_args(cls) -> str:
        """
        Returns the default arguments for the generation task.
        Override this method to customize the default arguments.
        """
        # Add any default command line arguments here
        return ""

    @classmethod
    def get_server_command_fn(cls) -> callable:
        """
        Returns the function to get the server command for the generation task.
        Override this method to customize the server command function.
        """
        # Use the default server command function or customize as needed
        return super().get_server_command_fn()

    def __init__(self, cfg: ReasoningWithForcedLangConfig):
        """Initialize the reasoning with forced language task."""
        super().__init__(cfg)

        # Load language configuration from lang_libs.py
        self.lang_config = self._load_language_config(cfg.language_code)

        # Determine the prefix to use
        if cfg.forced_prefix is not None:
            # Use explicitly provided prefix
            prefix_to_use = cfg.forced_prefix
        else:
            # Load from lang_libs.py
            prefix_to_use = f"<think> {self.lang_config['reasoning_prefix']}"

        # Initialize the forced prefix logits processor if enabled
        if cfg.enable_forced_prefix and self.tokenizer:
            from transformers import AutoTokenizer

            # Get the actual tokenizer object if we only have the name/path
            if isinstance(self.tokenizer, str):
                tokenizer_obj = AutoTokenizer.from_pretrained(self.tokenizer)
            else:
                tokenizer_obj = self.tokenizer
            self.force_prefix_processor = ForcePrefixLogitsProcessor(prefix_to_use, tokenizer_obj)
            self.actual_prefix = prefix_to_use
        else:
            self.force_prefix_processor = None
            self.actual_prefix = None

    def _load_language_config(self, language_code: str) -> dict:
        """Load language configuration from lang_libs.py"""
        lang_libs_path = Path(__file__).parent / "lang_libs.py"

        # Read the file as text and extract the JSON
        with open(lang_libs_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract the dictionary from the file (it's a Python dict, not JSON)
        # We'll use eval here since it's a controlled environment
        lang_configs = eval(content)

        if language_code not in lang_configs:
            raise ValueError(
                f"Language code '{language_code}' not found in lang_libs.py. Available: {list(lang_configs.keys())}"
            )

        return lang_configs[language_code]

    def postprocess(self):
        """
        Postprocess data after generation.
        Data is already saved to self.cfg.output_file.
        """
        # Call parent method first if needed
        super().postprocess()

        if self.cfg.enable_forced_prefix:
            with open(self.cfg.output_file, "r", encoding="utf-8") as f:
                results = [json.loads(line) for line in f]

            # Analyze how many generations successfully used the forced prefix
            prefix_usage_count = 0
            for result in results:
                if result.get("generation", "").startswith(self.actual_prefix):
                    prefix_usage_count += 1

            success_rate = prefix_usage_count / len(results) if results else 0
            print(f"Forced prefix usage: {prefix_usage_count}/{len(results)} ({success_rate:.2%})")
            print(f"Language: {self.cfg.language_code}")
            print(f"Forced prefix: {repr(self.actual_prefix)}")

            # You can add more analysis here:
            # - Language detection in the reasoning traces
            # - Quality metrics for reasoning
            # - Export statistics to a separate file

    def prefill_generation(self, data_point) -> dict | None:
        """
        Prefill generation in case LLM is not required.

        Args:
            data_point: Single data point to potentially prefill

        Returns:
            Prefilled output dict or None if LLM generation is needed
        """
        # Add logic to skip LLM generation for certain data points
        # Return None to use LLM generation, or return a dict with the output

        # Example:
        # if some_condition(data_point):
        #     return {"generation": "prefilled_response"}

        return super().prefill_generation(data_point)

    async def process_single_datapoint(self, data_point, all_data):
        """
        Process a single data point to generate output.

        Args:
            data_point: Single data point to process
            all_data: Full dataset

        Returns:
            Generated output dict
        """
        # Handle inference config - check if it's a dataclass or already a dict
        from dataclasses import asdict, is_dataclass

        if is_dataclass(self.cfg.inference):
            inference_params = asdict(self.cfg.inference)
        else:
            # Already a dict from Hydra
            inference_params = dict(self.cfg.inference)

        generation_params = {
            **inference_params,
            **self.extra_generate_params,
            "prompt": self.fill_prompt(data_point, all_data),
            "stop_phrases": [self.cfg.stop_phrase] if self.cfg.stop_phrase else None,
        }

        # Add forced prefix logits processor if enabled
        if self.force_prefix_processor is not None:
            # Add logits processors to the generation parameters
            logits_processors = generation_params.get("logits_processors", [])
            logits_processors.append(self.force_prefix_processor)
            generation_params["logits_processors"] = logits_processors

        if self.cfg.code_execution:
            if self.cfg.override_max_code_executions and self.cfg.total_code_executions_in_prompt is not None:
                generation_params["max_code_executions"] = data_point["total_code_executions"]

        output = await self.llm.generate_async(**generation_params)

        # Add the forced prefix to the output if it was used
        if self.force_prefix_processor is not None and self.cfg.enable_forced_prefix:
            # The forced prefix should already be in the generation, but we can add it explicitly
            # to make it clear in the output (this depends on how your model interface works)
            if "generation" in output and not output["generation"].startswith(self.actual_prefix):
                output["generation"] = self.actual_prefix + output["generation"]

        return output


GENERATION_TASK_CLASS = ReasoningWithForcedLangTask

# Register the custom config
cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="reasoning_with_forced_lang_config", node=ReasoningWithForcedLangConfig)


@hydra.main(version_base=None, config_name="reasoning_with_forced_lang_config")
def main(cfg: DictConfig):
    """Main function to run the reasoning with forced language task."""
    cfg = ReasoningWithForcedLangConfig(_init_nested=True, **cfg)

    task = ReasoningWithForcedLangTask(cfg)
    task.generate()


# Help message for this specific task
HELP_MESSAGE = get_help_message(
    ReasoningWithForcedLangConfig,
    # Add any additional help parameters here
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        main()
