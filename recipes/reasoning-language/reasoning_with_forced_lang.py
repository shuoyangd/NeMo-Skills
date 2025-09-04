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
Reasoning with Forced Language Generation Module

This module provides functionality to force language models to start their reasoning
with a specific prefix string. This is particularly useful for:

1. Multilingual reasoning: Force the model to think in a specific language
   Example: forced_prefix="<think> Ich muss auf Deutsch denken"

2. Structured reasoning: Ensure consistent reasoning patterns
   Example: forced_prefix="<reasoning> Let me think step by step:"

3. Output formatting: Control the beginning of the model's response
   Example: forced_prefix="I need to solve this problem by"

The implementation works by modifying the prompt to include the forced prefix
as the beginning of the assistant's response, then continuing generation from there.
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Add parent directory to sys.path to import nemo_skills
sys.path.append(str(Path(__file__).parent.parent.parent))

from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, nested_dataclass
from nemo_skills.utils import get_help_message, setup_logging


@nested_dataclass(kw_only=True)
class ReasoningWithForcedLangConfig(GenerateSolutionsConfig):
    """Configuration for reasoning with forced language generation task.

    This configuration allows you to force the model to start its reasoning
    with a specific prefix string. This is useful for:
    - Forcing the model to think in a specific language
    - Starting with specific reasoning patterns
    - Ensuring consistent output formatting
    """

    forced_prefix: str = ""  # Prefix to force at the beginning of reasoning chain
    """
    String to force at the beginning of the model's response.

    Examples:
    - "<think> Ich muss auf Deutsch denken" (force German thinking)
    - "<reasoning> Let me think step by step:" (force structured reasoning)
    - "I need to solve this problem by" (force specific approach)

    The prefix will be prepended to the assistant's response, and the model
    will continue generation from that point.
    """

    # Additional parameters you can add:
    # forced_language: str = "French"  # Language to force the model to use
    # reasoning_type: str = "step_by_step"  # Type of reasoning to apply

    def _get_disallowed_params(self):
        """Override to specify parameters that should not be changed from defaults."""
        return super()._get_disallowed_params()


class ReasoningWithForcedLangTask(GenerationTask):
    """Custom generation task for reasoning with forced language.

    This task extends the base GenerationTask to support forcing a specific
    prefix at the beginning of the model's response. This is particularly
    useful for multilingual reasoning tasks or when you want to ensure
    the model follows a specific reasoning pattern.

    Usage example:
        # Command line usage:
        python reasoning_with_forced_lang.py \
            input_file=data.jsonl \
            output_file=results.jsonl \
            forced_prefix="<think> Ich muss auf Deutsch denken" \
            server.model=llama-3.1-70b

        # Programmatic usage:
        cfg = ReasoningWithForcedLangConfig(
            input_file="data.jsonl",
            output_file="results.jsonl",
            forced_prefix="<think> Let me think step by step:",
            server={"model": "llama-3.1-70b"}
        )
        task = ReasoningWithForcedLangTask(cfg)
        task.generate()
    """

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
        # Add any custom initialization here

    def preprocess_data(self, data):
        """
        Preprocess data before generation.

        Args:
            data: List of data points to process

        Returns:
            Preprocessed data
        """
        # Add custom preprocessing logic here
        # Example: modify prompts to include language forcing instructions

        # Call parent method first if needed
        data = super().preprocess_data(data)

        # Your custom preprocessing logic here
        # for data_point in data:
        #     # Modify data_point as needed
        #     pass

        return data

    def postprocess(self):
        """
        Postprocess data after generation.
        Data is already saved to self.cfg.output_file.
        """
        # Add custom postprocessing logic here
        # Example: analyze language usage, filter results, etc.

        # Call parent method first if needed
        super().postprocess()

        # Your custom postprocessing logic here
        pass

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

    def fill_prompt(self, data_point, data):
        """
        Fill the prompt for a given data point.

        Args:
            data_point: Single data point to fill prompt for
            data: Full dataset (in case context is needed)

        Returns:
            Filled prompt (string or list of messages)
        """
        # Get the base prompt first
        filled_prompt = super().fill_prompt(data_point, data)

        # Add custom prompt modifications here
        # Example: add language forcing instructions
        # if isinstance(filled_prompt, list):
        #     # OpenAI format (list of messages)
        #     filled_prompt[-1]["content"] += f"\n\nPlease respond in {self.cfg.forced_language}."
        # else:
        #     # String format
        #     filled_prompt += f"\n\nPlease respond in {self.cfg.forced_language}."

        return filled_prompt

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

        # Add forced prefix if specified
        if self.cfg.forced_prefix:
            # For forced prefix, we need to modify the prompt to include the prefix
            # as the beginning of the assistant's response
            prompt = generation_params["prompt"]

            if isinstance(prompt, list):
                # OpenAI chat format - add assistant message with the forced prefix
                prompt.append({"role": "assistant", "content": self.cfg.forced_prefix})
            else:
                # String format - append the forced prefix
                prompt += f"\n\nAssistant: {self.cfg.forced_prefix}"

            generation_params["prompt"] = prompt

        if self.cfg.code_execution:
            if self.cfg.override_max_code_executions and self.cfg.total_code_executions_in_prompt is not None:
                generation_params["max_code_executions"] = data_point["total_code_executions"]

        output = await self.llm.generate_async(**generation_params)

        # If we used a forced prefix, we need to prepend it to the generation
        if self.cfg.forced_prefix and "generation" in output:
            output["generation"] = self.cfg.forced_prefix + output["generation"]

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
