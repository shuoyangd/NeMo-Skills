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
from pathlib import Path

from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import run_cmd, wrap_arguments
from nemo_skills.utils import get_chunked_filename


def get_stage_expname(base_expname, stage_name, suffix):
    return f"{base_expname}-{stage_name.replace('_', '-')}-{suffix}"


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------


def separate_data(cluster, expname, run_after, stage_config, code_dir, **kwargs):
    """Split input jsonl into target/non_target by metadata.domain regex."""
    input_file = stage_config["input_file"]
    output_dir = stage_config["output_dir"]
    domain_regex = stage_config["domain_regex"]

    jobs = stage_config.get("jobs", None)
    batch = stage_config.get("batch", None)

    cmd = (
        f"python {code_dir}/recipes/data-remix/scripts/separate_by_domain.py "
        f"    --input_file {input_file} "
        f"    --output_dir {output_dir} "
        f"    --domain_regex '{domain_regex}' "
        + (f"    -j {jobs} " if jobs is not None else "")
        + (f"    -b {batch} " if batch is not None else "")
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


def remix_data(cluster, expname, run_after, stage_config, code_dir, **kwargs):
    """Remix target and non-target data at one or more ratios."""
    target_file = stage_config["target_file"]
    non_target_file = stage_config["non_target_file"]
    output_dir = stage_config["output_dir"]

    # target_ratio may be a single float or a list; pass as comma-separated string
    target_ratio = stage_config["target_ratio"]
    if isinstance(target_ratio, list):
        target_ratio_str = ",".join(str(r) for r in target_ratio)
    else:
        target_ratio_str = str(target_ratio)

    seed = stage_config.get("seed", 42)
    jobs = stage_config.get("jobs", None)
    batch = stage_config.get("batch", None)
    fixed_non_target_pool = stage_config.get("fixed_non_target_pool", True)

    cmd = (
        f"python {code_dir}/recipes/data-remix/scripts/remix_data.py "
        f"    --target_file {target_file} "
        f"    --non_target_file {non_target_file} "
        f"    --output_dir {output_dir} "
        f"    --target_ratio '{target_ratio_str}' "
        f"    --seed {seed} "
        f"    {'--fixed_non_target_pool' if fixed_non_target_pool else '--no_fixed_non_target_pool'} "
        + (f"    -j {jobs} " if jobs is not None else "")
        + (f"    -b {batch} " if batch is not None else "")
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


def convert_to_sft(cluster, expname, run_after, stage_config, code_dir, config=None, **kwargs):
    """Convert each remixed file to SFT format using materialize_fast.py.

    When num_shards > 1, submits parallel Slurm jobs: one per file × shard.
    Each shard job extracts its line range via sed, runs materialize_fast.py,
    then a merge job concatenates the shard outputs using merge_chunks.py.
    """
    input_dir = stage_config["input_dir"]
    output_dir = stage_config["output_dir"]
    model = stage_config["model"]
    num_shards = stage_config.get("num_shards", 1)
    rerun_done = stage_config.get("rerun_done", False)

    jobs = stage_config.get("jobs", None)
    batch = stage_config.get("batch", None)
    inline_args = stage_config.get("inline_args", "")

    # Derive input filenames from remix_data's target_ratio config
    remix_config = config["stages"]["remix_data"]
    target_ratio = remix_config["target_ratio"]
    if not isinstance(target_ratio, list):
        target_ratio = [target_ratio]
    input_files = [(f"remix_r{r}", f"{input_dir}/remix_r{r}.jsonl") for r in target_ratio]

    materialize_cmd = (
        f"python {code_dir}/recipes/data-remix/scripts/materialize_fast.py"
        + (f" -j {jobs}" if jobs is not None else "")
        + (f" -b {batch}" if batch is not None else "")
        + f" -m {model}"
        + (f" {inline_args}" if inline_args else "")
    )

    if num_shards <= 1:
        # No sharding — one job per file (still parallelizes across files vs old for-loop)
        for base, input_file in input_files:
            cmd = (
                f"mkdir -p {output_dir} && "
                f"{materialize_cmd}"
                f" --input_file {input_file}"
                f" --output_file {output_dir}/{base}_sft.jsonl"
            )
            run_cmd(
                ctx=wrap_arguments(cmd),
                cluster=cluster,
                log_dir=f"{output_dir}/logs",
                expname=f"{expname}-{base}",
                run_after=run_after,
                num_gpus=0,
                **stage_config.get("stage_kwargs", {}),
            )
    else:
        # Sharded — one job per file × shard, plus a merge job per file
        shard_expnames = []
        for base, input_file in input_files:
            final_out = f"{output_dir}/{base}_sft.jsonl"
            final_tokens = f"{final_out}.tokens.jsonl"
            chunk_out_files = []
            chunk_token_files = []

            for shard_id in range(num_shards):
                chunk_out = get_chunked_filename(shard_id, final_out)
                chunk_tok = get_chunked_filename(shard_id, final_tokens)
                chunk_out_files.append(chunk_out)
                chunk_token_files.append(chunk_tok)

                # Each shard job: count lines, compute range, extract via sed if needed,
                # then run materialize_fast.py. We intentionally reuse an existing
                # shard_input file to avoid re-running sed on retries / partial reruns.
                done_guard = (
                    f'if [ -f {chunk_out}.done ]; then echo "Shard {shard_id} already done, skipping"; exit 0; fi && '
                    if not rerun_done
                    else ""
                )
                shard_cmd = (
                    f"{done_guard}"
                    f"mkdir -p {output_dir} && "
                    f"total=$(wc -l < {input_file}) && "
                    f'eval $(python -c "'
                    f"from nemo_skills.file_utils import calculate_chunk_indices; "
                    f"s, e = calculate_chunk_indices(int($total), {num_shards}, {shard_id}); "
                    f"print(f'start={{s+1}} end={{e}}')"  # sed is 1-indexed
                    f'") && '
                    f"shard_input={output_dir}/{base}_shard{shard_id}_input.jsonl && "
                    f'if [ ! -f "$shard_input" ]; then sed -n "${{start}},${{end}}p" {input_file} > "$shard_input"; fi && '
                    f"{materialize_cmd}"
                    f" --input_file $shard_input"
                    f" --output_file {chunk_out}"
                    f" --tokens_file {chunk_tok}"
                    f" && touch {chunk_out}.done {chunk_tok}.done"
                )
                shard_exp = f"{expname}-{base}-shard-{shard_id}"
                shard_expnames.append(shard_exp)
                run_cmd(
                    ctx=wrap_arguments(shard_cmd),
                    cluster=cluster,
                    log_dir=f"{output_dir}/logs",
                    expname=shard_exp,
                    run_after=run_after,
                    num_gpus=0,
                    **stage_config.get("stage_kwargs", {}),
                )

        # Merge job — uses merge_chunks.py to concatenate and clean up
        merge_parts = []
        for base, _ in input_files:
            final_out = f"{output_dir}/{base}_sft.jsonl"
            final_tokens = f"{final_out}.tokens.jsonl"
            chunk_outs = " ".join(get_chunked_filename(i, final_out) for i in range(num_shards))
            chunk_toks = " ".join(get_chunked_filename(i, final_tokens) for i in range(num_shards))
            shard_inputs = " ".join(f"{output_dir}/{base}_shard{i}_input.jsonl" for i in range(num_shards))
            merge_parts.append(
                f"python -m nemo_skills.inference.merge_chunks {final_out} {chunk_outs} && "
                f"python -m nemo_skills.inference.merge_chunks {final_tokens} {chunk_toks} && "
                f"rm -f {shard_inputs}"
            )
        merge_cmd = " && ".join(merge_parts)
        run_cmd(
            ctx=wrap_arguments(merge_cmd),
            cluster=cluster,
            log_dir=f"{output_dir}/logs",
            expname=f"{expname}-merge",
            run_after=shard_expnames,
            num_gpus=0,
            **stage_config.get("stage_kwargs", {}),
        )


stages_map = {
    "separate_data": separate_data,
    "remix_data": remix_data,
    "convert_to_sft": convert_to_sft,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def get_available_configs(config_dir):
    config_dir = Path(config_dir)
    if not config_dir.exists() or not config_dir.is_dir():
        return []
    yaml_files = list(config_dir.glob("*.yaml"))
    return [f.stem for f in yaml_files if not f.name.startswith("template")]


if __name__ == "__main__":
    config_dir = Path(__file__).parents[1] / "configs"
    available_configs = get_available_configs(config_dir)

    parser = argparse.ArgumentParser(description="Data remix pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=available_configs,
        help="Will pick a corresponding config from the configs folder",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated list of stages to run. If not specified, runs all stages from the config.",
    )

    args = parser.parse_args()

    config_path = config_dir / f"{args.mode}.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    if "pipeline_stages" not in config or not config["pipeline_stages"]:
        raise ValueError(f"Config file {config_path} must define a non-empty 'pipeline_stages' list.")
    full_stage_sequence = config["pipeline_stages"]

    if args.stages:
        stages_to_run = args.stages.split(",")
        print(f"Running specified stages: {stages_to_run}")
    else:
        stages_to_run = full_stage_sequence
        print(f"Running all stages defined in config for mode '{args.mode}': {stages_to_run}")

    for stage in stages_to_run:
        if stage not in stages_map:
            raise ValueError(f"Unknown stage: '{stage}'. Available stages: {list(stages_map.keys())}")
        if stage not in full_stage_sequence:
            raise ValueError(
                f"Stage '{stage}' is not part of the sequence for mode '{args.mode}' in {config_path}. "
                f"Must be one of: {full_stage_sequence}"
            )

    cluster = config["cluster"]
    expname_base = config["expname"]
    suffix = config.get("suffix", args.mode)
    code_dir = config.get("code_dir", "/nemo_run/code")

    for stage in stages_to_run:
        print(f"\n--- Running stage: {stage} ---")
        stage_func = stages_map[stage]
        stage_config = config.get("stages", {}).get(stage, {})

        current_expname = get_stage_expname(expname_base, stage, suffix)

        dep_stages = stage_config.get("dependencies", None)
        if dep_stages is not None:
            dependencies = [get_stage_expname(expname_base, dep, suffix) for dep in dep_stages]
        else:
            dependencies = config.get("initial_dependency", None)

        print(f"Dependencies for '{stage}': {dependencies}")

        stage_func(
            cluster=cluster,
            expname=current_expname,
            run_after=dependencies,
            stage_config=stage_config,
            code_dir=code_dir,
            config=config,
        )

    print("\n--- Pipeline finished. ---")
