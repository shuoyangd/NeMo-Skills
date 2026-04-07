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

    cmd = (
        f"python {code_dir}/recipes/data-remix/scripts/remix_data.py "
        f"    --target_file {target_file} "
        f"    --non_target_file {non_target_file} "
        f"    --output_dir {output_dir} "
        f"    --target_ratio '{target_ratio_str}' "
        f"    --seed {seed} "
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


def convert_to_sft(cluster, expname, run_after, stage_config, code_dir, **kwargs):
    """Convert each remixed file from OAI format to SFT format."""
    input_dir = stage_config["input_dir"]
    output_dir = stage_config["output_dir"]

    # Glob pattern covers all remixed files produced by remix_data
    cmd = (
        f"mkdir -p {output_dir} && "
        f"for f in {input_dir}/remix_r*.jsonl; do "
        f"    base=$(basename $f .jsonl); "
        f"    python {code_dir}/recipes/data-remix/scripts/oai_to_sft.py "
        f"        --input_file $f "
        f"        --output_file {output_dir}/${{base}}_sft.jsonl "
        f"        {stage_config.get('inline_args', '')}; "
        f"done"
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
        )

    print("\n--- Pipeline finished. ---")
