#!/usr/bin/env python3

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import multiprocessing as mp
import time
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

_DEFAULT_CHAT_TEMPLATE = Path(__file__).parent / "chat_template_nemonext.jinja"
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>\n"


def replace_json_args(messages):
    """Convert JSON string arguments to dict objects in tool calls."""
    for i in range(len(messages)):
        if messages[i]["role"] == "assistant":
            if messages[i].get("tool_calls"):
                for j in range(len(messages[i]["tool_calls"])):
                    if isinstance(messages[i]["tool_calls"][j]["function"]["arguments"], str):
                        messages[i]["tool_calls"][j]["function"]["arguments"] = json.loads(
                            messages[i]["tool_calls"][j]["function"]["arguments"]
                        )
    return messages


def _render_template(messages, tokenizer, enable_thinking=True, tools=None):
    """Render a conversation with the Nemotron chat template."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )


def _parse_rendered_blocks(rendered_template):
    """Parse top-level blocks from chat_template_nemonext.jinja output.

    This intentionally overfits to the Nemotron template structure, which emits
    top-level turns as `<|im_start|>role\\n...<|im_end|>\\n` blocks. We rely on
    those explicit markers instead of re-rendering conversation prefixes.
    """
    blocks = []
    cursor = 0

    while cursor < len(rendered_template):
        if not rendered_template.startswith(_IM_START, cursor):
            snippet = rendered_template[cursor : cursor + 80]
            raise ValueError(f"Unexpected content outside top-level block at offset {cursor}: {snippet!r}")

        role_start = cursor + len(_IM_START)
        role_end = rendered_template.find("\n", role_start)
        if role_end == -1:
            raise ValueError("Malformed rendered template: missing newline after <|im_start|> role header")

        block_end = rendered_template.find(_IM_END, role_end + 1)
        if block_end == -1:
            raise ValueError("Malformed rendered template: missing <|im_end|> terminator")

        block_end += len(_IM_END)
        blocks.append(
            {
                "role": rendered_template[role_start:role_end],
                "content": rendered_template[cursor:block_end],
            }
        )
        cursor = block_end

    if "".join(block["content"] for block in blocks) != rendered_template:
        raise ValueError("Parsed blocks do not reconstruct the rendered template")

    return blocks


def _logical_chunks_from_rendered(messages, rendered_blocks):
    """Map rendered top-level blocks back to the logical chunk roles expected downstream.

    This is template-specific:
    - the template always emits a leading system block
    - consecutive tool messages are rendered as a single synthetic user block
    """
    if not rendered_blocks:
        return []
    if rendered_blocks[0]["role"] != "system":
        raise ValueError("Expected leading system block in rendered template")

    chunks = [{"role": "system", "content": rendered_blocks[0]["content"]}]
    block_idx = 1
    msg_idx = 1 if messages and messages[0]["role"] == "system" else 0

    while msg_idx < len(messages):
        message = messages[msg_idx]

        if message["role"] == "tool":
            if block_idx >= len(rendered_blocks):
                raise ValueError("Missing rendered block for tool message group")
            if rendered_blocks[block_idx]["role"] != "user":
                raise ValueError("Expected synthetic user block for tool response group")

            while msg_idx < len(messages) and messages[msg_idx]["role"] == "tool":
                msg_idx += 1
            chunks.append({"role": "tool", "content": rendered_blocks[block_idx]["content"]})
            block_idx += 1
            continue

        if block_idx >= len(rendered_blocks):
            raise ValueError(f"Missing rendered block for message role {message['role']}")

        chunks.append({"role": message["role"], "content": rendered_blocks[block_idx]["content"]})
        block_idx += 1
        msg_idx += 1

    if block_idx != len(rendered_blocks):
        raise ValueError("Unused rendered blocks remain after reconstructing logical chunks")

    return chunks


def split_template_into_messages(messages, tokenizer, start_from_last_user=True, enable_thinking=True, tools=None):
    """Split rendered template back into logical message chunks.

    This implementation deliberately overfits to chat_template_nemonext.jinja.
    It trades generic prefix re-rendering for a single full render plus parsing
    of the template's explicit top-level markers.
    """
    full_template = _render_template(messages, tokenizer, enable_thinking=enable_thinking, tools=tools)
    rendered_blocks = _parse_rendered_blocks(full_template)
    logical_chunks = _logical_chunks_from_rendered(messages, rendered_blocks)

    if not start_from_last_user:
        return logical_chunks

    last_user_chunk_idx = max(i for i, chunk in enumerate(logical_chunks) if chunk["role"] == "user")
    return [
        logical_chunks[0],
        {
            "role": "user",
            "content": "".join(chunk["content"] for chunk in logical_chunks[1 : last_user_chunk_idx + 1]),
        },
        *logical_chunks[last_user_chunk_idx + 1 :],
    ]


def create_masked_messages(messages, tokenizer, tools=None):
    """Create message chunks, optionally starting from last user message"""

    # Check if conversation has thinking (determines splitting strategy)
    has_thinking = any("reasoning_content" in msg and msg["reasoning_content"] for msg in messages)

    if has_thinking:
        # Split based on user messages - create chunks up to each user message
        user_idxs = [i for i, msg in enumerate(messages) if msg["role"] == "user"]
        result = []
        for i in range(len(user_idxs)):
            if i == len(user_idxs) - 1:
                # Last user message - include all remaining messages
                messages_i = messages
            else:
                # Include messages up to but not including the next user message
                messages_i = messages[: user_idxs[i + 1]]

            chunks = split_template_into_messages(
                messages_i, tokenizer, start_from_last_user=True, enable_thinking=has_thinking, tools=tools
            )

            result.append((chunks, messages_i))  # Return both chunks and original messages
        return result
    else:
        # Generate one sequence
        chunks = split_template_into_messages(
            messages, tokenizer, start_from_last_user=False, enable_thinking=has_thinking, tools=tools
        )
        return [(chunks, messages)]  # Return both chunks and original messages


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
        for chunks, _ in masked_messages:
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

            # split_template_into_messages now derives chunk boundaries from a
            # single parsed full render, so we keep the structural checks here
            # and avoid re-rendering the same conversation slice again.
            if "".join(chunk["content"] for chunk in chunks) == "":
                all_passed = False
                error_message = "Empty chunk group"
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

            if skip_token_validation:
                concatenated_processed = "".join(chunk["content"] for chunk in processed_chunks)
                num_tokens = len(encode_plain_text(tokenizer, concatenated_processed))
            else:
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


def process_batch(batch):
    """Process a batch of input lines in one worker call."""
    return [process_line(item) for item in batch]


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
    parser.add_argument(
        "-b", "--batch_size", type=int, default=4, help="Number of input lines processed per worker call"
    )
    parser.add_argument(
        "--skip-token-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip per-chunk tokenization validation for faster processing (default: skip)",
    )
    parser.add_argument(
        "--ordered",
        action="store_true",
        default=False,
        help="Preserve input line ordering in output (slower due to imap vs imap_unordered)",
    )

    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")

    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Model: {args.model}")
    print(f"Chat template: {args.chat_template}")
    print(f"Workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Token validation: {'enabled' if not args.skip_token_validation else 'skipped'}")
    print(f"Output ordering: {'ordered' if args.ordered else 'unordered'}")

    start_time = time.time()

    # Load tokenizer once in the main process. Workers inherit it via fork (copy-on-write),
    # so they never touch the filesystem for tokenizer initialization.
    print("Loading tokenizer...")
    global tokenizer, skip_token_validation
    skip_token_validation = args.skip_token_validation
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

    def batched_task_iter():
        batch = []
        for item in task_iter():
            batch.append(item)
            if len(batch) >= args.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # Process in parallel
    print(f"Processing with {args.workers} workers...")
    passed_count = 0
    failed_count = 0

    tokens_output_path = args.tokens_file if args.tokens_file else f"{args.output_file}.tokens.jsonl"
    with open(args.output_file, "w") as outfile, open(tokens_output_path, "w") as tokenfile:
        with Pool(processes=args.workers) as pool:
            map_fn = pool.imap if args.ordered else pool.imap_unordered
            for batch_results in tqdm(
                map_fn(process_batch, batched_task_iter(), chunksize=1),
                total=(total_lines + args.batch_size - 1) // args.batch_size,
                desc="Processing",
                unit="batch",
            ):
                for result, line_num, error in batch_results:
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
