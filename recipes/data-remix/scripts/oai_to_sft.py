import argparse
import functools
import json
import multiprocessing
import sys

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

_processor = None


class ThinkingStartError(ValueError):
    """Raised when --thinking-start is not found at the expected position in the output."""

    pass


def _worker_init(model_name_or_path):
    global _processor
    from transformers import AutoTokenizer

    _processor = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


def _prepare_messages(messages, reasoning_open, reasoning_close):
    """Normalize messages for apply_chat_template.

    - Always normalizes content=None to "" (common in tool-call assistant turns).
    - If reasoning_open is set, embeds reasoning_content into content as
      {open}{reasoning}{close}{content} and strips the field (for models without
      native reasoning_content support, e.g. Mistral Reasoning).
    - If reasoning_open is not set (default), passes reasoning_content through to
      the template as-is (for models with native support: Qwen3, Nemotron, etc.).
    """
    out = []
    for msg in messages:
        msg = dict(msg)
        if msg.get("content") is None:
            msg["content"] = ""
        if reasoning_open:
            reasoning = msg.pop("reasoning_content", None)
            if reasoning:
                msg["content"] = f"{reasoning_open}{reasoning}{reasoning_close}{msg['content']}"
        out.append(msg)
    return out


def _transform(line, reasoning_open, reasoning_close, thinking_start, default_system, placement):
    data = json.loads(line)
    messages = list(data["messages"])

    if default_system is not None and (not messages or messages[0]["role"] != "system"):
        messages = [{"role": "system", "content": default_system}] + messages

    messages = _prepare_messages(messages, reasoning_open, reasoning_close)
    tools = data.get("tools")

    last_asst = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "assistant"),
        None,
    )
    if last_asst is None:
        raise ValueError("no assistant message found")

    context = messages[:last_asst]
    full = messages[: last_asst + 1]

    input_str = _processor.apply_chat_template(context, tools=tools, tokenize=False, add_generation_prompt=True)
    full_str = _processor.apply_chat_template(full, tools=tools, tokenize=False, add_generation_prompt=False)

    # If the generation prompt already ends with thinking_start, strip it to get
    # the bare base for prefix matching against full_str.
    if thinking_start and input_str.endswith(thinking_start):
        base = input_str[: -len(thinking_start)]
    else:
        base = input_str

    # Some tokenizers add BOS only once; strip it from full_str if needed.
    actual_full = full_str
    if not actual_full.startswith(base):
        bos = getattr(_processor, "bos_token", "") or ""
        if bos and actual_full.startswith(bos) and not base.startswith(bos):
            actual_full = actual_full[len(bos) :]
    if not actual_full.startswith(base):
        raise ValueError("full template output does not start with context prefix")

    remainder = actual_full[len(base) :]

    if not thinking_start or placement == "output":
        # No thinking split: full remainder goes to output. Use base (not input_str)
        # so any thinking_start appended by the generation prompt is not leaked into
        # input when placement is "output".
        input_str = base
        output_str = remainder
    elif remainder.startswith(thinking_start):
        # thinking_start is at the start of the assistant output; move it to the
        # end of input (e.g. Qwen3, or Nemotron when reasoning_content is present).
        input_str = base + thinking_start
        output_str = remainder[len(thinking_start) :]
    elif base != input_str:
        # Generation prompt included thinking_start, but the completed turn renders
        # the opening tag differently (e.g. Nemotron with no reasoning_content
        # renders <think></think> while the gen prompt uses <think>\n).
        # If the completed turn opens with an empty think block, move it to input —
        # it acts as a prompt to skip reasoning. Otherwise fall through to full output.
        open_tag = thinking_start.rstrip()
        if open_tag.startswith("<") and open_tag.endswith(">"):
            close_tag = "</" + open_tag[1:]
        elif open_tag.startswith("[") and open_tag.endswith("]"):
            close_tag = "[/" + open_tag[1:]
        else:
            close_tag = None
        empty_think = open_tag + close_tag if close_tag else None
        if empty_think and remainder.startswith(empty_think):
            input_str = base + empty_think
            output_str = remainder[len(empty_think) :]
        else:
            input_str = base
            output_str = remainder
    elif placement == "best-effort-input":
        # thinking_start not found; fall back to full remainder in output.
        print(
            f"Warning: --thinking-start {thinking_start!r} not found; output begins: {remainder[:80]!r}",
            file=sys.stderr,
        )
        input_str = base
        output_str = remainder
    else:
        # placement == "input": thinking_start is required in every record.
        raise ThinkingStartError(
            f"--thinking-start {thinking_start!r} not found at end of generation "
            f"prompt or start of assistant output; "
            f"output begins: {remainder[:80]!r}"
        )

    return {"input": input_str, "output": output_str}


def worker(chunk, reasoning_open, reasoning_close, thinking_start, default_system, placement):
    results = []
    errors = 0
    for line in chunk:
        if not line.strip():
            continue
        try:
            record = _transform(line, reasoning_open, reasoning_close, thinking_start, default_system, placement)
            results.append(json.dumps(record, ensure_ascii=False))
        except ThinkingStartError:
            raise  # placement="input": always fatal, propagate to main
        except Exception as e:
            print(f"Warning: skipping line: {e}", file=sys.stderr)
            errors += 1
    return results, errors


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenAI-format JSONL to Megatron SFT format using a HuggingFace chat template."
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument(
        "-m", "--model", required=True, help="HuggingFace model name or local path whose chat template to apply"
    )
    parser.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")
    parser.add_argument(
        "--reasoning-open",
        default="",
        help="Opening delimiter for embedding reasoning_content into the message content "
        "(for models without native reasoning_content support). "
        "Must be paired with --reasoning-close. "
        "Default: '' (disabled; reasoning_content is passed to the template as-is)",
    )
    parser.add_argument(
        "--reasoning-close",
        default="",
        help="Closing delimiter for embedding reasoning_content into the message content. "
        "Must be paired with --reasoning-open.",
    )
    parser.add_argument(
        "--thinking-start",
        default="<think>\n",
        help="String that marks the start of the thinking block in the assistant turn. "
        "Interpretation depends on --thinking-placement. (default: '<think>\\n')",
    )
    parser.add_argument(
        "--thinking-placement",
        default="input",
        choices=["input", "output", "best-effort-input"],
        help="Where the thinking block lands in the input/output split. "
        "'input': thinking_start is moved to end of input; fatal if not found "
        "(use when every record is expected to have a thinking block). "
        "'output': thinking stays in output, nothing is moved (use for non-thinking "
        "models or when you want the full assistant turn in output). "
        "'best-effort-input': move to input when found, fall back to full output "
        "when not found — no error (use for mixed datasets where some records lack "
        "a thinking block). (default: input)",
    )
    parser.add_argument(
        "--default-system",
        default="",
        metavar="TEXT",
        help="Content of system message to inject when none is present in the data (default: empty string)",
    )
    parser.add_argument(
        "--no-default-system",
        dest="default_system",
        action="store_const",
        const=None,
        help="Disable system message injection even when none is present in the data",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print INPUT/OUTPUT for the first record and exit")
    parser.add_argument("-j", "--jobs", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("-b", "--batch", type=int, default=500)

    args = parser.parse_args()

    if bool(args.reasoning_open) != bool(args.reasoning_close):
        parser.error("--reasoning-open and --reasoning-close must be specified together")

    if args.dry_run:
        from transformers import AutoTokenizer

        global _processor
        _processor = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    result = _transform(
                        line,
                        args.reasoning_open,
                        args.reasoning_close,
                        args.thinking_start,
                        args.default_system,
                        args.thinking_placement,
                    )
                    print(f"INPUT:\n{result['input']}\nOUTPUT:\n{result['output']}")
                    return
        return

    out_file = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout

    try:
        pool = multiprocessing.Pool(
            processes=args.jobs,
            initializer=_worker_init,
            initargs=(args.model,),
        )

        worker_fn = functools.partial(
            worker,
            reasoning_open=args.reasoning_open,
            reasoning_close=args.reasoning_close,
            thinking_start=args.thinking_start,
            default_system=args.default_system,
            placement=args.thinking_placement,
        )

        def _batches(path, batch_size):
            with open(path, "r", encoding="utf-8") as f:
                batch = []
                for line in f:
                    batch.append(line)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch

        ok = 0
        skipped = 0
        bar = _tqdm(unit="lines", unit_scale=True, desc="Converting", file=sys.stderr) if _tqdm else None
        try:
            for results, errors in pool.imap(worker_fn, _batches(args.input, args.batch)):
                for line in results:
                    out_file.write(line + "\n")
                    ok += 1
                skipped += errors
                if bar is not None:
                    bar.update(len(results) + errors)
        except ThinkingStartError as e:
            print(f"Fatal (--thinking-placement input): {e}", file=sys.stderr)
            pool.terminate()
            sys.exit(1)
        finally:
            if bar is not None:
                bar.close()

        print(f"Converted: {ok}, Skipped: {skipped}", file=sys.stderr)

    except BrokenPipeError:
        sys.stderr.close()
    finally:
        pool.close()
        pool.join()
        if args.output and out_file:
            out_file.close()


if __name__ == "__main__":
    main()
