import argparse
import json
import multiprocessing
import sys

_tokenizer = None


def _worker_init(model_name_or_path):
    global _tokenizer
    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


def _embed_reasoning(messages, reasoning_tag):
    out = []
    for msg in messages:
        reasoning = msg.get("reasoning_content")
        # Pass all fields through; only modify content and drop reasoning_content.
        msg = {k: v for k, v in msg.items() if k != "reasoning_content"}
        if msg.get("content") is None:
            msg["content"] = ""
        if reasoning:
            msg["content"] = f"<{reasoning_tag}>\n{reasoning}\n</{reasoning_tag}>\n\n{msg['content']}"
        out.append(msg)
    return out


def _transform(line, reasoning_tag, thinking_start, default_system):
    data = json.loads(line)
    messages = list(data["messages"])

    if default_system is not None and (not messages or messages[0]["role"] != "system"):
        messages = [{"role": "system", "content": default_system}] + messages

    messages = _embed_reasoning(messages, reasoning_tag)
    tools = data.get("tools")

    last_asst = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "assistant"),
        None,
    )
    if last_asst is None:
        raise ValueError("no assistant message found")

    context = messages[:last_asst]
    full = messages[: last_asst + 1]

    input_str = _tokenizer.apply_chat_template(context, tools=tools, tokenize=False, add_generation_prompt=True)
    full_str = _tokenizer.apply_chat_template(full, tools=tools, tokenize=False, add_generation_prompt=False)

    # Some tokenizers add BOS only once; strip it from full_str if needed.
    if not full_str.startswith(input_str):
        bos = getattr(_tokenizer, "bos_token", "") or ""
        if bos and full_str.startswith(bos) and not input_str.startswith(bos):
            full_str = full_str[len(bos) :]
        if not full_str.startswith(input_str):
            raise ValueError("full template output does not start with context prefix")

    output_str = full_str[len(input_str) :]

    if thinking_start:
        if not output_str.startswith(thinking_start):
            print(
                f"Fatal: assistant output does not start with --thinking-start {thinking_start!r}.\n"
                f"First 200 chars of output: {output_str[:200]!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        input_str += thinking_start
        output_str = output_str[len(thinking_start) :]

    return {"input": input_str, "output": output_str}


def worker(chunk, reasoning_tag, thinking_start, default_system):
    results = []
    errors = 0
    for line in chunk:
        if not line.strip():
            continue
        try:
            record = _transform(line, reasoning_tag, thinking_start, default_system)
            results.append(json.dumps(record, ensure_ascii=False))
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
        "--reasoning-tag", default="think", help="XML tag name to wrap reasoning_content fields (default: think)"
    )
    parser.add_argument(
        "--thinking-start",
        default="<think>\n",
        help="String that must prefix the assistant output; moved to end of input "
        "to form the thinking split (default: '<think>\\n', pass '' to disable)",
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

    if args.dry_run:
        from transformers import AutoTokenizer

        global _tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    result = _transform(line, args.reasoning_tag, args.thinking_start, args.default_system)
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

        with open(args.input, "r", encoding="utf-8") as f:
            batch, async_results = [], []
            for line in f:
                batch.append(line)
                if len(batch) >= args.batch:
                    async_results.append(
                        pool.apply_async(worker, (batch, args.reasoning_tag, args.thinking_start, args.default_system))
                    )
                    batch = []
            if batch:
                async_results.append(
                    pool.apply_async(worker, (batch, args.reasoning_tag, args.thinking_start, args.default_system))
                )

        ok = 0
        skipped = 0
        for res in async_results:
            lines, errors = res.get()
            for line in lines:
                out_file.write(line + "\n")
                ok += 1
            skipped += errors

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
