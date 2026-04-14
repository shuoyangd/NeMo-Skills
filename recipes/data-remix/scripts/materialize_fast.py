#!/usr/bin/env python3

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import multiprocessing as mp
import time
from multiprocessing import Pool
from pathlib import Path

# Import the core functions from materialize.py
from materialize import create_masked_messages, replace_json_args
from tqdm import tqdm
from transformers import AutoTokenizer

_DEFAULT_CHAT_TEMPLATE = Path(__file__).parent / "chat_template_nemonext.jinja"


def encode_plain_text(tokenizer, text):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    return list(encoded["input_ids"])


def validate_processed_chunks_tokenization(tokenizer, processed_chunks):
    concatenated_processed = "".join(chunk["content"] for chunk in processed_chunks)
    conversation_tokens = encode_plain_text(tokenizer, concatenated_processed)
    part_tokens = []
    for chunk in processed_chunks:
        part_tokens.extend(encode_plain_text(tokenizer, chunk["content"]))
    return conversation_tokens == part_tokens, len(conversation_tokens)


def process_line(args):
    """Process a single line"""
    line_num, line, extra_info = args

    try:
        data = json.loads(line.strip())
        messages = data.get("messages", [])
        messages = replace_json_args(messages)

        # Simple check: if messages contain tool calls but top-level tools key is missing/null/empty, skip this line
        def _has_tool_calls(msgs):
            for _m in msgs:
                if not isinstance(_m, dict):
                    continue
                if _m.get("role") == "assistant" and _m.get("tool_calls"):
                    try:
                        if isinstance(_m["tool_calls"], list) and len(_m["tool_calls"]) > 0:
                            return True
                    except Exception:
                        pass
                for _k in ("content", "reasoning_content"):
                    _v = _m.get(_k)
                    if isinstance(_v, str) and "<tool_call>" in _v:
                        return True
            return False

        tools_missing = ("tools" not in data) or (data.get("tools") in (None, [], {}))
        if _has_tool_calls(messages) and tools_missing:
            return None, line_num, "Tool calls present but missing top-level tools key"

        # Validation: if any message has "<tool_call>" but no message has "# Tools", fail this line
        any_tool_call = any(
            isinstance(m, dict)
            and (
                (isinstance(m.get("content"), str) and "<tool_call>" in m.get("content"))
                or (isinstance(m.get("reasoning_content"), str) and "<tool_call>" in m.get("reasoning_content"))
            )
            for m in messages
        )
        any_tools_header = any(
            isinstance(m, dict)
            and (
                (isinstance(m.get("content"), str) and "# Tools" in m.get("content"))
                or (isinstance(m.get("reasoning_content"), str) and "# Tools" in m.get("reasoning_content"))
            )
            for m in messages
        )
        if any_tool_call and not any_tools_header:
            return None, line_num, "Some message has <tool_call> but no message has # Tools"

        # Extract tools
        tools = None
        conversation_tools = data.get("tools", tools)

        # Get masked messages
        masked_messages = create_masked_messages(messages, tokenizer, tools=conversation_tools)
        if len(masked_messages) == 0:
            return None, line_num, "No masked messages created"

        # Verify each group matches expected template
        all_passed = True
        error_message = None
        for i, (chunks, messages_i) in enumerate(masked_messages):
            # Determine if this group has thinking
            has_thinking = any("reasoning_content" in msg and msg["reasoning_content"] for msg in messages_i)

            # Additional sanity checks from materialize.py
            for j, obj in enumerate(chunks):
                if j > 0 and obj["role"] == "assistant" and chunks[j - 1]["role"] == "assistant":
                    all_passed = False
                    error_message = "Multiple assistant turns detected"
                    break
                content = obj.get("content", "")
                # New sanity check: some message has '<tool_call>' but the first message lacks '# Tools'
                if j == 0:
                    any_tool_call_in_chunks = any(
                        isinstance(c, dict) and ("<tool_call>" in c.get("content", "")) for c in chunks
                    )
                    if any_tool_call_in_chunks and ("# Tools" not in content):
                        all_passed = False
                        error_message = "Some message has <tool_call> but first message missing # Tools"
                        break
                if content.endswith("assistant\n<think><"):
                    all_passed = False
                    error_message = "Cutoff thinking detected"
                    break
            if not all_passed:
                break

            # Render the expected template for this group
            full_template_i = tokenizer.apply_chat_template(
                messages_i,
                tokenize=False,
                add_generation_prompt=False,
                tools=conversation_tools,
                chat_template_kwargs={"enable_thinking": has_thinking},
            )

            # Concatenate the chunks
            concatenated = "".join([chunk["content"] for chunk in chunks])

            # Verify they match
            if full_template_i != concatenated:
                all_passed = False
                error_message = "Template mismatch"
                break

        if not all_passed:
            return None, line_num, error_message or "Validation failed"

        # Process each chunk group separately
        results = []
        token_counts = []
        for chunks, _ in masked_messages:
            # Post-process: Split first chunk if it contains both system and user content
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                if (
                    i == 0
                    and chunk["role"] == "user"
                    and "<|im_start|>system" in chunk["content"]
                    and "<|im_end|>" in chunk["content"]
                    and "<|im_start|>user" in chunk["content"]
                ):
                    content = chunk["content"]
                    # Find the boundaries
                    system_start = content.find("<|im_start|>system")
                    system_end = content.find("<|im_end|>\n", system_start)
                    user_start = content.find("<|im_start|>user", system_end)

                    if system_start != -1 and system_end != -1 and user_start != -1:
                        # Extract system content (with tags)
                        system_content = content[system_start : system_end + len("<|im_end|>\n")]

                        # Extract user content (with tags)
                        user_content = content[user_start:]

                        # Create two separate chunks
                        processed_chunks.append({"role": "system", "content": system_content})
                        processed_chunks.append({"role": "user", "content": user_content})
                    else:
                        processed_chunks.append(chunk)
                else:
                    processed_chunks.append(chunk)

            tokens_match, num_tokens = validate_processed_chunks_tokenization(tokenizer, processed_chunks)
            if not tokens_match:
                return None, line_num, "Chunk tokenization mismatch"

            # Create output record for this chunk group
            if extra_info:
                output_data = {"messages": processed_chunks, **{k: v for k, v in data.items() if k != "messages"}}
            else:
                output_data = {
                    "messages": processed_chunks,
                }

            results.append(json.dumps(output_data))
            token_counts.append(num_tokens)

        return (results, token_counts), line_num, None

    except Exception as e:
        return None, line_num, str(e)


def main():
    parser = argparse.ArgumentParser(description="Simple parallel materialize processor")
    parser.add_argument("--input_file", required=True, help="Input JSONL file")
    parser.add_argument("--output_file", required=True, help="Output JSONL file")
    parser.add_argument("-m", "--model", required=True, help="HuggingFace model name or path for tokenizer")
    parser.add_argument(
        "--chat_template",
        type=str,
        default=str(_DEFAULT_CHAT_TEMPLATE),
        help="Path to Jinja2 chat template file (default: chat_template_nemonext.jinja next to this script)",
    )
    parser.add_argument(
        "--tokens_file",
        dest="tokens_file",
        type=str,
        default=None,
        help="Optional output file for token counts (one integer per line). Defaults to <output_file>.tokens.jsonl",
    )
    parser.add_argument("--extra_info", action="store_true", help="Include extra info in output")
    parser.add_argument("-j", "--workers", type=int, default=mp.cpu_count(), help="Number of worker processes")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size for processing")

    args = parser.parse_args()

    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Model: {args.model}")
    print(f"Chat template: {args.chat_template}")
    print(f"Workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")

    start_time = time.time()

    # Load tokenizer once in the main process. Workers inherit it via fork (copy-on-write),
    # so they never touch the filesystem for tokenizer initialization.
    print("Loading tokenizer...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with open(args.chat_template) as f:
        tokenizer.chat_template = f.read()
    print("Tokenizer loaded.")

    # Count lines without loading file into memory
    print("Counting lines...")
    with open(args.input_file, "rb") as f:
        total_lines = sum(1 for _ in f)
    print(f"Total lines: {total_lines:,}")

    def task_iter():
        with open(args.input_file) as f:
            for i, line in enumerate(f):
                yield (i + 1, line, args.extra_info)

    # Process in parallel
    print(f"Processing with {args.workers} workers...")
    passed_count = 0
    failed_count = 0

    tokens_output_path = args.tokens_file if args.tokens_file else f"{args.output_file}.tokens.jsonl"
    with open(args.output_file, "w") as outfile, open(tokens_output_path, "w") as tokenfile:
        with Pool(processes=args.workers) as pool:
            for result, line_num, error in tqdm(
                pool.imap(process_line, task_iter(), chunksize=args.batch_size),
                total=total_lines,
                desc="Processing",
            ):
                if result is not None:
                    output_lines, token_counts = result
                    for output_line, token_count in zip(output_lines, token_counts):
                        outfile.write(output_line + "\n")
                        tokenfile.write(f"{token_count}\n")
                    passed_count += 1
                else:
                    failed_count += 1
                    if error and failed_count <= 10:
                        tqdm.write(f"Line {line_num}: {error}")

    # Final summary
    total_time = time.time() - start_time
    final_rate = total_lines / total_time if total_time > 0 else 0
    success_rate = passed_count / total_lines * 100 if total_lines > 0 else 0

    print("\nProcessing complete!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Processed: {total_lines:,} lines")
    print(f"Passed: {passed_count:,} ({success_rate:.1f}%)")
    print(f"Failed: {failed_count:,}")
    print(f"Average rate: {final_rate:.1f} lines/sec")


if __name__ == "__main__":
    main()
