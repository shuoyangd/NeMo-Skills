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
Formatted translation pipeline.

Stages:
  1. make_concise              – keep only fields_to_consider from each JSON object
  2. wrap                      – add source_lang / target_lang, serialize concise JSON as `src`
  3. escape_special_tokens     – (optional) escape model special tokens to <<ESC:...>> placeholders
  4. generate                  – run LLM to produce a formatted translation
  5. unescape_special_tokens   – (optional) restore <<ESC:...>> placeholders to original tokens
  6. unwrap                    – extract the translated JSON from the model's generation field
  7. format_check              – (optional) run only the format checker for pass/no-pass stats
  8. filter_and_merge          – validate with format checker; on pass, replace fields_to_translate
                                  in the original object with translated values
  9. curate                    – placeholder for downstream data curation steps
 10. convert_to_sft            – (optional) materialize curated translated data for SFT

Most standalone translation scripts live under:
  recipes/translation/scripts/formatted-translation/

Required top-level config keys:
  base_output_dir:      root directory for all stage outputs
  cluster:              cluster name (passed to pipeline utilities)
  expname:              base experiment name
  fields_to_translate:  list of top-level fields that must be translated
  fields_to_consider:   (optional) fields shown to the model; defaults to fields_to_translate
  input_file:           path to the original JSONL
  target_lang:          one target language name (e.g. "German"), or
  target_langs:         weighted list of {language, weight} objects
"""

import argparse
import json
import shlex
from pathlib import Path

from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import generate as generate_fn
from nemo_skills.pipeline.cli import run_cmd, wrap_arguments
from nemo_skills.utils import get_chunked_filename

# Absolute path to the scripts directory (as seen inside the container/job)
_SCRIPTS_DIR = "/nemo_run/code/recipes/translation/scripts/formatted-translation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _done_check(file_path: str) -> str:
    """Shell command that fails with a clear message if {file_path}.done is missing."""
    return (
        f'test -f "{file_path}.done" '
        f'|| {{ echo "ERROR: {file_path}.done not found — generate stage incomplete. '
        f'Re-run the pipeline to resume." >&2; exit 1; }}'
    )


def _stage_expname(base_expname: str, stage_name: str, suffix: str) -> str:
    return f"{base_expname}-{stage_name.replace('_', '-')}-{suffix}"


def _stage_dir(base_output_dir: str, stage_name: str, stage_config: dict) -> str:
    return stage_config.get("output_dir", f"{base_output_dir}/{stage_name}")


# ---------------------------------------------------------------------------
# Stage: make_concise
# ---------------------------------------------------------------------------


def make_concise(cluster, expname, run_after, stage_config, **kwargs):
    """Keep only fields_to_consider from each JSON object."""
    base_output_dir = kwargs["base_output_dir"]
    fields_to_consider = kwargs["fields_to_consider"]
    from_messages = kwargs.get("from_messages", False)

    input_file = stage_config.get("input_file", kwargs.get("input_file"))
    output_dir = _stage_dir(base_output_dir, "make_concise", stage_config)
    output_file = stage_config.get("output_file", f"{output_dir}/concise.jsonl")

    fields_arg = " ".join(fields_to_consider)
    from_messages_flag = " --from-messages" if from_messages else ""
    cmd = (
        f"python {_SCRIPTS_DIR}/make_concise.py "
        f"    --input {input_file} "
        f"    --output {output_file} "
        f"    --fields {fields_arg} "
        f"    {from_messages_flag} "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        num_gpus=0,
        **stage_config.get("stage_kwargs", {}),
    )


# ---------------------------------------------------------------------------
# Stage: wrap
# ---------------------------------------------------------------------------


def wrap(cluster, expname, run_after, stage_config, **kwargs):
    """Wrap each concise JSON object as {source_lang, target_lang, src: <json_string>}."""
    base_output_dir = kwargs["base_output_dir"]

    input_file = stage_config.get("input_file", f"{base_output_dir}/make_concise/concise.jsonl")
    if "target_lang" in stage_config:
        target_lang = stage_config["target_lang"]
        target_langs = None
    elif "target_langs" in stage_config:
        target_lang = None
        target_langs = stage_config["target_langs"]
    else:
        target_lang = kwargs.get("target_lang")
        target_langs = kwargs.get("target_langs")
    if (target_lang is None) == (target_langs is None):
        raise ValueError("wrap requires exactly one of target_lang or target_langs")
    seed = int(stage_config.get("language_seed", kwargs.get("language_seed", 0)))
    output_dir = _stage_dir(base_output_dir, "wrap", stage_config)
    output_file = stage_config.get("output_file", f"{output_dir}/wrapped.jsonl")

    language_args = (
        f"--target-lang {shlex.quote(str(target_lang))}"
        if target_lang is not None
        else f"--target-langs-json {shlex.quote(json.dumps(target_langs))} --seed {seed}"
    )
    cmd = (
        f"python {_SCRIPTS_DIR}/wrap_sft_data.py "
        f"    --input {input_file} "
        f"    --output {output_file} "
        f"    {language_args} "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        num_gpus=0,
        **stage_config.get("stage_kwargs", {}),
    )


# ---------------------------------------------------------------------------
# Stage: escape_special_tokens
# ---------------------------------------------------------------------------


def escape_special_tokens(cluster, expname, run_after, stage_config, **kwargs):
    """Escape model special tokens so they don't interfere with generation."""
    base_output_dir = kwargs["base_output_dir"]

    input_file = stage_config.get("input_file", f"{base_output_dir}/wrap/wrapped.jsonl")
    output_dir = _stage_dir(base_output_dir, "escape_special_tokens", stage_config)
    output_file = stage_config.get("output_file", f"{output_dir}/escaped.jsonl")
    model = stage_config["model"]

    cmd = (
        f"python {_SCRIPTS_DIR}/escape_special_tokens.py "
        f"    --input {input_file} "
        f"    --output {output_file} "
        f"    --model {model} "
        f"    --mode escape "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        num_gpus=0,
        **stage_config.get("stage_kwargs", {}),
    )


# ---------------------------------------------------------------------------
# Stage: unescape_special_tokens
# ---------------------------------------------------------------------------


def unescape_special_tokens(cluster, expname, run_after, stage_config, **kwargs):
    """Restore escaped special tokens in generation output."""
    base_output_dir = kwargs["base_output_dir"]

    input_file = stage_config.get("input_file", f"{base_output_dir}/generate/output.jsonl")
    output_dir = _stage_dir(base_output_dir, "unescape_special_tokens", stage_config)
    output_file = stage_config.get("output_file", f"{output_dir}/unescaped.jsonl")
    model = stage_config["model"]

    cmd = (
        f"{_done_check(input_file)} && "
        f"python {_SCRIPTS_DIR}/escape_special_tokens.py "
        f"    --input {input_file} "
        f"    --output {output_file} "
        f"    --model {model} "
        f"    --mode unescape "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        num_gpus=0,
        **stage_config.get("stage_kwargs", {}),
    )


# ---------------------------------------------------------------------------
# Stage: generate
# ---------------------------------------------------------------------------


def generate(cluster, expname, run_after, stage_config, **kwargs):
    """Run LLM translation with a format-specific prompt config."""
    base_output_dir = kwargs["base_output_dir"]
    pipeline_stages = kwargs.get("pipeline_stages", [])

    if "escape_special_tokens" in pipeline_stages:
        default_input = f"{base_output_dir}/escape_special_tokens/escaped.jsonl"
    else:
        default_input = f"{base_output_dir}/wrap/wrapped.jsonl"
    input_file = stage_config.get("input_file", default_input)
    output_dir = _stage_dir(base_output_dir, "generate", stage_config)
    format_config = stage_config["format_config"]

    stage_kwargs = stage_config.get("stage_kwargs", {})
    stage_kwargs.setdefault("sbatch_kwargs", {})
    if isinstance(stage_kwargs["sbatch_kwargs"], dict):
        stage_kwargs["sbatch_kwargs"].setdefault("poll_estimated_start_time", False)

    generate_fn(
        ctx=wrap_arguments(f"++prompt_config={format_config} {stage_config.get('inline_args', '')} "),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        expname=expname,
        run_after=run_after,
        **stage_kwargs,
    )


# ---------------------------------------------------------------------------
# Stage: mock_generate  (testing only — replaces generate for local runs)
# ---------------------------------------------------------------------------


def mock_generate(cluster, expname, run_after, stage_config, **kwargs):
    """Mock generation for local testing. Writes to the same dir as generate."""
    base_output_dir = kwargs["base_output_dir"]

    input_file = stage_config.get("input_file", f"{base_output_dir}/wrap/wrapped.jsonl")
    # Write to generate/ so downstream stages need no config changes.
    output_dir = stage_config.get("output_dir", f"{base_output_dir}/generate")
    output_file = stage_config.get("output_file", f"{output_dir}/output.jsonl")

    cmd = f"python {_SCRIPTS_DIR}/mock_generate.py     --input {input_file}     --output {output_file} "
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        num_gpus=0,
        **stage_config.get("stage_kwargs", {}),
    )


# ---------------------------------------------------------------------------
# Stage: unwrap
# ---------------------------------------------------------------------------


def unwrap(cluster, expname, run_after, stage_config, **kwargs):
    """Extract the translated JSON object from the model's generation field."""
    base_output_dir = kwargs["base_output_dir"]

    input_file = stage_config.get("input_file", f"{base_output_dir}/generate/output.jsonl")
    output_dir = _stage_dir(base_output_dir, "unwrap", stage_config)
    output_file = stage_config.get("output_file", f"{output_dir}/unwrapped.jsonl")

    verbose_flag = "--verbose" if stage_config.get("verbose", False) else ""
    cmd = f"python {_SCRIPTS_DIR}/unwrap.py     --input {input_file}     --output {output_file}     {verbose_flag} "
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        num_gpus=0,
        **stage_config.get("stage_kwargs", {}),
    )


# ---------------------------------------------------------------------------
# Stage: format_check
# ---------------------------------------------------------------------------


def format_check(cluster, expname, run_after, stage_config, **kwargs):
    """Run only the configured format checker and print pass/no-pass stats."""
    base_output_dir = kwargs["base_output_dir"]
    pipeline_stages = kwargs.get("pipeline_stages", [])

    if "unescape_special_tokens" in pipeline_stages:
        default_input = f"{base_output_dir}/unescape_special_tokens/unescaped.jsonl"
    else:
        default_input = f"{base_output_dir}/generate/output.jsonl"
    input_file = stage_config.get("input_file", stage_config.get("generation_file", default_input))
    checker_name = stage_config.get("checker")
    if checker_name is None:
        raise ValueError("format_check requires a checker, either in stages.format_check or stages.filter_and_merge.")
    output_dir = _stage_dir(base_output_dir, "format_check", stage_config)

    verbose_flag = "--verbose" if stage_config.get("verbose", False) else ""
    fail_flag = "--fail-on-nopass" if stage_config.get("fail_on_nopass", False) else ""
    max_failures = stage_config.get("max_failures", 10)

    cmd = (
        f"mkdir -p {output_dir} && "
        f"python {_SCRIPTS_DIR}/filter_by_format.py "
        f"    --stats-only "
        f"    --checker {checker_name} "
        f"    --input {input_file} "
        f"    --max-failures {max_failures} "
        f"    {verbose_flag} "
        f"    {fail_flag} "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        num_gpus=0,
        **stage_config.get("stage_kwargs", {}),
    )


# ---------------------------------------------------------------------------
# Stage: filter_and_merge
# ---------------------------------------------------------------------------


def filter_and_merge(cluster, expname, run_after, stage_config, **kwargs):
    """Validate each translated output; on pass, merge translated fields into the original."""
    base_output_dir = kwargs["base_output_dir"]
    fields_to_translate = kwargs["fields_to_translate"]
    pipeline_stages = kwargs.get("pipeline_stages", [])
    from_messages = kwargs.get("from_messages", False)

    original_file = stage_config.get("original_file", kwargs.get("input_file"))
    if "unescape_special_tokens" in pipeline_stages:
        default_gen = f"{base_output_dir}/unescape_special_tokens/unescaped.jsonl"
    else:
        default_gen = f"{base_output_dir}/generate/output.jsonl"
    generation_file = stage_config.get("generation_file", default_gen)
    checker_name = stage_config["checker"]
    output_dir = _stage_dir(base_output_dir, "filter_and_merge", stage_config)
    output_file = stage_config.get("output_file", f"{output_dir}/translated.jsonl")

    fields_arg = " ".join(fields_to_translate)
    on_fail = stage_config.get("on_fail", "keep")
    to_messages_flag = "--to-messages" if from_messages else ""
    verbose_flag = "--verbose" if stage_config.get("verbose", False) else ""

    cmd = (
        f"python {_SCRIPTS_DIR}/filter_and_merge.py "
        f"    --checker {checker_name} "
        f"    --original-file {original_file} "
        f"    --generation-file {generation_file} "
        f"    --fields-to-translate {fields_arg} "
        f"    --output {output_file} "
        f"    --on-fail {on_fail} "
        f"    {to_messages_flag} "
        f"    {verbose_flag} "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        num_gpus=0,
        **stage_config.get("stage_kwargs", {}),
    )


# ---------------------------------------------------------------------------
# Stage: convert_to_sft
# ---------------------------------------------------------------------------


def convert_to_sft(cluster, expname, run_after, stage_config, **kwargs):
    """Convert curated/filter_and_merge output to materialized SFT message chunks."""
    base_output_dir = kwargs["base_output_dir"]
    pipeline_stages = kwargs.get("pipeline_stages", [])

    default_input = f"{base_output_dir}/filter_and_merge/translated.jsonl"
    if (
        "curate" in pipeline_stages
        and "convert_to_sft" in pipeline_stages
        and pipeline_stages.index("curate") < pipeline_stages.index("convert_to_sft")
    ):
        default_input = f"{base_output_dir}/curate/curated.jsonl"
    input_file = stage_config.get("input_file", default_input)
    output_dir = _stage_dir(base_output_dir, "convert_to_sft", stage_config)
    output_file = stage_config.get("output_file", f"{output_dir}/translated_sft.jsonl")
    model = stage_config["model"]
    num_shards = stage_config.get("num_shards", 1)
    dependent_jobs = stage_config.get("dependent_jobs", 0)
    rerun_done = stage_config.get("rerun_done", False)

    jobs = stage_config.get("jobs", None)
    batch = stage_config.get("batch", None)
    inline_args = stage_config.get("inline_args", "")
    chat_template = stage_config.get("chat_template", None)
    materialize_command = stage_config.get("command")
    if materialize_command is None:
        materialize_script = stage_config.get("script")
        materialize_command = (
            f"python {materialize_script}" if materialize_script else f"python {_SCRIPTS_DIR}/materialize_sft.py"
        )

    materialize_cmd = (
        materialize_command
        + (f" -j {jobs}" if jobs is not None else "")
        + (f" -b {batch}" if batch is not None else "")
        + f" -m {model}"
        + (f" --chat_template {chat_template}" if chat_template else "")
        + (f" {inline_args}" if inline_args else "")
    )

    if num_shards <= 1:
        done_guard = (
            "" if rerun_done else f'if [ -f "{output_file}.done" ]; then echo "Already done, skipping"; exit 0; fi && '
        )
        cmd = (
            f"mkdir -p {output_dir} && "
            f"{done_guard}"
            f"{materialize_cmd}"
            f" --input_file {input_file}"
            f" --output_file {output_file}"
            f" && touch {output_file}.done"
        )
        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            log_dir=f"{output_dir}/logs",
            expname=expname,
            run_after=run_after,
            num_gpus=0,
            dependent_jobs=dependent_jobs,
            **stage_config.get("stage_kwargs", {}),
        )
        return

    shard_expnames = []
    final_tokens = f"{output_file}.tokens.jsonl"
    for shard_id in range(num_shards):
        chunk_out = get_chunked_filename(shard_id, output_file)
        chunk_tok = get_chunked_filename(shard_id, final_tokens)
        shard_input = f"{output_dir}/translated_shard{shard_id}_input.jsonl"

        done_guard = (
            ""
            if rerun_done
            else f'if [ -f "{chunk_out}.done" ]; then echo "Shard {shard_id} already done, skipping"; exit 0; fi && '
        )
        shard_cmd = (
            f"{done_guard}"
            f"mkdir -p {output_dir} && "
            f"total=$(wc -l < {input_file}) && "
            f'eval $(python -c "'
            f"from nemo_skills.file_utils import calculate_chunk_indices; "
            f"s, e = calculate_chunk_indices(int($total), {num_shards}, {shard_id}); "
            f"print(f'start={{s+1}} end={{e}}')"
            f'") && '
            f'if [ ! -f "{shard_input}" ]; then sed -n "${{start}},${{end}}p" {input_file} > "{shard_input}"; fi && '
            f"{materialize_cmd}"
            f" --input_file {shard_input}"
            f" --output_file {chunk_out}"
            f" --tokens_file {chunk_tok}"
            f" && touch {chunk_out}.done {chunk_tok}.done"
        )
        shard_expname = f"{expname}-shard-{shard_id}"
        shard_expnames.append(shard_expname)
        run_cmd(
            ctx=wrap_arguments(shard_cmd),
            cluster=cluster,
            log_dir=f"{output_dir}/logs",
            expname=shard_expname,
            run_after=run_after,
            num_gpus=0,
            dependent_jobs=dependent_jobs,
            **stage_config.get("stage_kwargs", {}),
        )

    chunk_outs = " ".join(get_chunked_filename(i, output_file) for i in range(num_shards))
    chunk_toks = " ".join(get_chunked_filename(i, final_tokens) for i in range(num_shards))
    shard_inputs = " ".join(f"{output_dir}/translated_shard{i}_input.jsonl" for i in range(num_shards))
    merge_cmd = (
        f"python -m nemo_skills.inference.merge_chunks {output_file} {chunk_outs} && "
        f"python -m nemo_skills.inference.merge_chunks {final_tokens} {chunk_toks} && "
        f"rm -f {shard_inputs} && "
        f"touch {output_file}.done {final_tokens}.done"
    )
    run_cmd(
        ctx=wrap_arguments(merge_cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=shard_expnames,
        num_gpus=0,
        **stage_config.get("stage_kwargs", {}),
    )


# ---------------------------------------------------------------------------
# Stage: curate
# ---------------------------------------------------------------------------


def curate(cluster, expname, run_after, stage_config, **kwargs):
    """Placeholder for downstream data curation steps."""
    raise NotImplementedError(
        "The 'curate' stage is not yet implemented. "
        "Remove it from 'pipeline_stages' in your config to run the pipeline without it."
    )


# ---------------------------------------------------------------------------
# Stage registry & entrypoint
# ---------------------------------------------------------------------------

stages_map = {
    "make_concise": make_concise,
    "wrap": wrap,
    "escape_special_tokens": escape_special_tokens,
    "generate": generate,
    "mock_generate": mock_generate,
    "unescape_special_tokens": unescape_special_tokens,
    "unwrap": unwrap,
    "format_check": format_check,
    "filter_and_merge": filter_and_merge,
    "curate": curate,
    "convert_to_sft": convert_to_sft,
}


def get_available_configs(config_dir):
    config_dir = Path(config_dir)
    if not config_dir.exists() or not config_dir.is_dir():
        return []
    return [f.stem for f in config_dir.glob("*.yaml")]


if __name__ == "__main__":
    config_dir = Path(__file__).parents[1] / "config" / "pipeline"
    available_configs = get_available_configs(config_dir)

    parser = argparse.ArgumentParser(description="Formatted translation pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=available_configs,
        help="Name of a YAML config file under config/pipeline/",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated list of stages to run. Defaults to all stages in config.",
    )
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default=None,
        help="Override base_output_dir from the config file.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Override input_file from the config file.",
    )
    args = parser.parse_args()

    config_path = config_dir / f"{args.config}.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    if "pipeline_stages" not in config or not config["pipeline_stages"]:
        raise ValueError(f"{config_path} must define a non-empty 'pipeline_stages' list.")
    full_stage_sequence = config["pipeline_stages"]

    stages_to_run = args.stages.split(",") if args.stages else full_stage_sequence
    print(f"Running stages: {stages_to_run}")

    for stage in stages_to_run:
        if stage not in stages_map:
            raise ValueError(f"Unknown stage '{stage}'. Available: {list(stages_map.keys())}")
        if args.stages is None and stage not in full_stage_sequence:
            raise ValueError(f"Stage '{stage}' is not in the config's pipeline_stages.")

    base_output_dir = args.base_output_dir or config["base_output_dir"]
    if args.input_file:
        config["input_file"] = args.input_file
    suffix = config.get("suffix", args.config)
    cluster = config["cluster"]
    expname_base = config["expname"]
    fields_to_translate = config["fields_to_translate"]
    fields_to_consider = config.get("fields_to_consider", fields_to_translate)
    from_messages = config.get("from_messages", False)
    input_file = config.get("input_file")
    target_lang = config.get("target_lang")
    target_langs = config.get("target_langs")
    language_seed = config.get("language_seed", 0)
    if (target_lang is None) == (target_langs is None):
        raise ValueError(f"{config_path} must define exactly one of 'target_lang' or 'target_langs'.")

    prev_expname = None
    for stage in stages_to_run:
        print(f"\n--- Running stage: {stage} ---")
        stage_configs = config.get("stages", {})
        stage_config = stage_configs.get(stage, {})
        if stage == "format_check":
            # Let `--stages format_check` work with existing configs that only
            # define checker/generation_file under filter_and_merge.
            merge_config = stage_configs.get("filter_and_merge", {})
            if "checker" not in stage_config and "checker" in merge_config:
                stage_config = {**stage_config, "checker": merge_config["checker"]}
            if "generation_file" not in stage_config and "generation_file" in merge_config:
                stage_config = {**stage_config, "generation_file": merge_config["generation_file"]}
        current_expname = _stage_expname(expname_base, stage, suffix)

        dep_stages = stage_config.get("dependencies")
        if dep_stages is not None:
            dependencies = [_stage_expname(expname_base, d, suffix) for d in dep_stages]
        elif prev_expname is not None:
            dependencies = [prev_expname]
        else:
            dependencies = config.get("initial_dependency", None)

        stages_map[stage](
            cluster=cluster,
            expname=current_expname,
            run_after=dependencies,
            stage_config=stage_config,
            base_output_dir=base_output_dir,
            fields_to_translate=fields_to_translate,
            fields_to_consider=fields_to_consider,
            from_messages=from_messages,
            input_file=input_file,
            target_lang=target_lang,
            target_langs=target_langs,
            language_seed=language_seed,
            pipeline_stages=full_stage_sequence,
        )
        prev_expname = current_expname

    print("\n--- Pipeline finished. ---")
