# Translation Recipes

This directory contains two translation paths:

- `translate_jsonl.py`: a line-by-line translation task for preserving multiline text layout. This is currently documented here only as a stub.
- `pipeline/formatted_translation.py`: an end-to-end formatted translation pipeline for translating structured JSONL records, validating model outputs with format checkers, merging passing translations back into source records, and optionally materializing SFT data.

The formatted pipeline is the recommended path for translating structured datasets where the model must preserve a known JSON format.

## Line-By-Line Pipeline

TODO

## Formatted Pipeline

The formatted pipeline is configured by YAML files under:

```text
recipes/translation/config/pipeline/
```

The pipeline entry point is:

```bash
python recipes/translation/pipeline/formatted_translation.py --config <config_name>
```

`<config_name>` is the YAML filename without `.yaml`, for example:

```bash
python recipes/translation/pipeline/formatted_translation.py --config formatted_translation_example
```

You can override common paths at runtime:

```bash
python recipes/translation/pipeline/formatted_translation.py \
  --config formatted_translation_example \
  --base-output-dir /path/to/pipeline-output \
  --input-file /path/to/input.jsonl
```

You can run a subset of stages:

```bash
python recipes/translation/pipeline/formatted_translation.py \
  --config formatted_translation_example \
  --stages format_check
```

When `--stages` is supplied, the listed stages are run in that order. This is useful for resuming a partial pipeline or running inspection-only steps such as `format_check`.

## Config Structure

A pipeline config needs these top-level fields:

```yaml
base_output_dir: /path/to/output/root
expname: formatted-translation
cluster: cw-dfw-cpu

fields_to_translate:
  - problem
  - generation

fields_to_consider:
  - problem
  - generation

input_file: /path/to/input.jsonl
target_lang: Simplified Chinese

pipeline_stages:
  - make_concise
  - wrap
  - generate
  - filter_and_merge

stages:
  generate:
    format_config: recipes/translation/config/formatted-translation/format_nano1.yaml
    inline_args: >-
      ++inference.endpoint_type=chat
```

Important fields:

- `base_output_dir`: root directory where each stage writes a subdirectory.
- `cluster`: cluster config name passed to NeMo-Skills pipeline utilities.
- `expname`: base experiment name; each stage appends its own suffix.
- `suffix`: optional suffix used in experiment names. Defaults to the config name.
- `input_file`: original JSONL file.
- `target_lang`: a single target language name used by the prompt wrapper. Mutually exclusive with `target_langs`.
- `target_langs`: weighted list of `{language, weight}` objects. Weights must be positive and are normalized automatically. The wrapper uses largest-remainder allocation, so record counts match the requested proportions as closely as integer counts allow.
- `language_seed`: optional seed (default `0`) used to reproducibly shuffle weighted language assignments.
- `fields_to_translate`: fields that should be replaced in the final merged output.
- `fields_to_consider`: fields shown to the model. Defaults to `fields_to_translate`.
- `from_messages`: when true, fields are extracted from `messages[i].content` by position and merged back into the `messages` array.
- `pipeline_stages`: default ordered stage sequence.
- `stages.<stage_name>`: per-stage overrides.

Per-stage configs commonly support:

- `input_file`, `output_file`, or `output_dir` overrides.
- `dependencies`: explicit stage dependencies by stage name.
- `stage_kwargs`: extra arguments forwarded to `run_cmd` or `generate`.

## Stage Overview

### `make_concise`

Keeps only `fields_to_consider` from each input record and adds `_translation_src_id`, a stable hash of the original record. The ID lets downstream stages merge generation records back into the original file without relying only on line alignment. When the original record contains `metadata.dataset_id`, it is also propagated as `_translation_dataset_id`. Both `_translation_*` fields remain outside the model-visible `src` payload.

Default input:

```text
{input_file}
```

Default output:

```text
{base_output_dir}/make_concise/concise.jsonl
```

If `from_messages: true`, this stage maps configured fields to `messages[i].content` by position.

### `wrap`

Wraps each concise record as translation prompt input:

```json
{"source_lang": "English", "target_lang": "Simplified Chinese", "src": "{...}"}
```

With `target_langs`, every record still has exactly one `target_lang`, and the weighted split is applied independently to every unique `_translation_dataset_id`. For example, each dataset containing 100 rows and weights `0.4/0.3/0.2/0.1` produces 40 German, 30 French, 20 Russian, and 10 Japanese prompts in a seed-controlled shuffled order. Records without a dataset ID are treated as one additional group. For group sizes that do not divide cleanly, largest-remainder allocation gives the closest integer counts. The legacy single-language `target_lang` config remains supported.

Default output:

```text
{base_output_dir}/wrap/wrapped.jsonl
```

### `escape_special_tokens`

Optional stage that replaces model special tokens with placeholders before generation, so the model does not interpret source text as control tokens.

Default input:

```text
{base_output_dir}/wrap/wrapped.jsonl
```

Default output:

```text
{base_output_dir}/escape_special_tokens/escaped.jsonl
```

Requires:

```yaml
stages:
  escape_special_tokens:
    model: /path/to/tokenizer-or-model
```

### `generate`

Runs model generation using `nemo_skills.pipeline.cli.generate`.

Default input is:

```text
{base_output_dir}/escape_special_tokens/escaped.jsonl
```

if `escape_special_tokens` is in `pipeline_stages`, otherwise:

```text
{base_output_dir}/wrap/wrapped.jsonl
```

The stage requires a format prompt config:

```yaml
stages:
  generate:
    format_config: recipes/translation/config/formatted-translation/format_nano1_escaped.yaml
    inline_args: >-
      ++inference.endpoint_type=chat
      ++inference.temperature=0.6
    stage_kwargs:
      server_type: vllm
      model: /path/to/model
      server_gpus: 8
      num_chunks: 8
```

Generation output normally lands under:

```text
{base_output_dir}/generate/output.jsonl
```

or chunked output files, depending on `generate` stage options.

### `mock_generate`

Testing-only replacement for `generate`. It writes mock translations into the same `generate/` directory shape expected by downstream stages.

This is used by:

```bash
python recipes/translation/pipeline/test_pipeline.py
```

### `unescape_special_tokens`

Optional stage that restores placeholders from `escape_special_tokens` back to original special tokens in generated output.

Default input:

```text
{base_output_dir}/generate/output.jsonl
```

Default output:

```text
{base_output_dir}/unescape_special_tokens/unescaped.jsonl
```

### `unwrap`

Extracts the first valid JSON object from each raw model generation and writes it as standalone JSONL for inspection.

Default input:

```text
{base_output_dir}/generate/output.jsonl
```

Default output:

```text
{base_output_dir}/unwrap/unwrapped.jsonl
```

This stage is useful for manual inspection, but `filter_and_merge` performs its own extraction and validation.

### `format_check`

Runs only the format checker and prints pass/no-pass stats. It does not merge or write translated records.

Default input is:

```text
{base_output_dir}/unescape_special_tokens/unescaped.jsonl
```

if `unescape_special_tokens` is in `pipeline_stages`, otherwise:

```text
{base_output_dir}/generate/output.jsonl
```

Example:

```bash
python recipes/translation/pipeline/formatted_translation.py \
  --config nano-v3-zh-dfw \
  --stages format_check
```

Config:

```yaml
stages:
  format_check:
    checker: format_nano1
    input_file: /optional/custom/generation.jsonl
    max_failures: 20
    fail_on_nopass: false
    verbose: true
```

If `stages.format_check.checker` is omitted, the pipeline reuses `stages.filter_and_merge.checker` when available.

Internally, this stage calls:

```bash
python recipes/translation/scripts/formatted-translation/filter_by_format.py \
  --stats-only \
  --checker <checker> \
  --input <generation_file>
```

### `filter_and_merge`

Validates generated translations with a format checker. Passing translations are extracted and merged back into the original record. Failing rows are either kept unchanged or dropped based on `on_fail`.

Default original file:

```text
{input_file}
```

Default generation file is:

```text
{base_output_dir}/unescape_special_tokens/unescaped.jsonl
```

if `unescape_special_tokens` is in `pipeline_stages`, otherwise:

```text
{base_output_dir}/generate/output.jsonl
```

Default output:

```text
{base_output_dir}/filter_and_merge/translated.jsonl
```

Config:

```yaml
stages:
  filter_and_merge:
    checker: format_nano1
    on_fail: drop  # or keep
    generation_file: /optional/custom/generation.jsonl
```

If `from_messages: true`, translated fields are written back into `messages[i].content` by position. Otherwise top-level fields listed in `fields_to_translate` are updated.

### `curate`

Reserved for downstream curation. It is currently not implemented.

Do not include this stage in runnable configs yet. If included, it raises `NotImplementedError`.

The intended future ordering is:

```text
filter_and_merge -> curate -> convert_to_sft
```

### `convert_to_sft`

Converts translated message-style records into materialized SFT message chunks using a tokenizer chat template.

Default input is:

```text
{base_output_dir}/filter_and_merge/translated.jsonl
```

If `curate` appears before `convert_to_sft` in `pipeline_stages`, the default input becomes:

```text
{base_output_dir}/curate/curated.jsonl
```

Default output:

```text
{base_output_dir}/convert_to_sft/translated_sft.jsonl
```

Config:

```yaml
stages:
  convert_to_sft:
    model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
    jobs: 16
    batch: 100
    num_shards: 10
    dependent_jobs: 3
    inline_args: --extra_info
```

Notes:

- `model` is used to load the Hugging Face tokenizer and chat template.
- `chat_template` can point to a custom Jinja template file.
- `num_shards > 1` launches shard jobs plus a dependent merge job.
- `inline_args: --extra_info` preserves non-`messages` fields in the SFT output.
- Token counts are written next to the output as `translated_sft.jsonl.tokens.jsonl`.

## Format Checkers

Format checkers live under:

```text
recipes/translation/config/formatted-translation/
```

Each checker is a Python class that inherits from `FormatChecker` and implements:

```python
def check(self, input_text: str, output_text: str) -> bool:
    ...
```

Checker naming conventions:

- `format1` maps to `format1.py::Format1Checker`.
- `format_nano1` maps to `format_nano1.py::FormatNano1Checker`.
- `module.ClassName` can be used for custom imports.

Prompt configs for formatted translation are YAML files in the same directory. The `generate.format_config` field points to one of those prompt configs.

## Running Common Workflows

Run all configured stages:

```bash
python recipes/translation/pipeline/formatted_translation.py --config nano-v3-zh-dfw
```

Run only generation-time format stats:

```bash
python recipes/translation/pipeline/formatted_translation.py \
  --config nano-v3-zh-dfw \
  --stages format_check
```

Run only merge after generation is complete:

```bash
python recipes/translation/pipeline/formatted_translation.py \
  --config nano-v3-zh-dfw \
  --stages filter_and_merge
```

Run only SFT conversion after `filter_and_merge` is complete:

```bash
python recipes/translation/pipeline/formatted_translation.py \
  --config nano-v3-zh-dfw \
  --stages convert_to_sft
```

Override the output root for an existing generated run:

```bash
python recipes/translation/pipeline/formatted_translation.py \
  --config nano-v3-zh-dfw \
  --stages convert_to_sft \
  --base-output-dir /path/to/existing/pipeline-output
```

## Extending The Pipeline

To add a new stage:

1. Add a stage function in `pipeline/formatted_translation.py` with the same signature as existing stages.
2. Build the command using scripts under `scripts/formatted-translation/` or call an existing pipeline utility.
3. Add the function to `stages_map`.
4. Add the stage name to `pipeline_stages` in the config where it should run.
5. Add per-stage config under `stages.<stage_name>`.

Stage functions should prefer small, streamable scripts that read JSONL and write JSONL. Use `_stage_dir(...)` for standard output directory defaults and pass `stage_kwargs` through to pipeline utilities so clusters, mounts, resources, and dependencies remain configurable.

To add a new format:

1. Add a prompt YAML under `config/formatted-translation/`.
2. Add a checker Python file under `config/formatted-translation/`.
3. Point `stages.generate.format_config` to the prompt YAML.
4. Point `stages.filter_and_merge.checker` and optionally `stages.format_check.checker` to the checker name.
5. Run `format_check` on a small sample before launching large merge or SFT stages.

## Operational Notes

- Nemo Run packages committed code when launching remote jobs from this repository. Commit pipeline/script changes before submitting remote jobs.
- Avoid using `NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK` for normal runs, because uncommitted files may not be included in the packaged code.
- If a remote data path is not already mounted by the selected cluster config, pass `mount_paths` through `stage_kwargs`.
- Large generation outputs may be chunked. Ensure generation has completed and merged to the expected downstream input before running `format_check`, `filter_and_merge`, or `convert_to_sft`.
