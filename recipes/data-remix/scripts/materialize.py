#!/usr/bin/env python3

import argparse
import json
import time

from tqdm import tqdm
from transformers import AutoTokenizer


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


def find_last_user_message_end(messages, tokenizer, enable_thinking=True, tools=None):
    """Find where the last user message ends in the rendered template"""

    # Find the last user message index
    last_user_idx = max(i for i, msg in enumerate(messages) if msg["role"] == "user")

    # Render up to the last user message (inclusive)
    if enable_thinking and (
        "reasoning_content" not in messages[last_user_idx + 1]
        or messages[last_user_idx + 1]["reasoning_content"] == ""
    ):
        # Manual hack for empty reasoning content mismatch
        template_up_to_last_user = tokenizer.apply_chat_template(
            messages[: last_user_idx + 1],
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        template_up_to_last_user += "<|im_start|>assistant\n<think></think>"
    else:
        template_up_to_last_user = tokenizer.apply_chat_template(
            messages[: last_user_idx + 1],
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
            tools=tools,
        )

    return len(template_up_to_last_user)


def split_template_into_messages(messages, tokenizer, start_from_last_user=True, enable_thinking=True, tools=None):
    """Split rendered template back into individual message chunks"""

    # Render full template
    full_template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )

    # Get first "message": if starting from last user, this includes all prior assistant turns as well
    if start_from_last_user:
        system_end = full_template.find("<|im_end|>\n") + len("<|im_end|>\n")
        last_user_idx = max(i for i, msg in enumerate(messages) if msg["role"] == "user")
        last_user_pos = find_last_user_message_end(messages, tokenizer, enable_thinking=enable_thinking, tools=tools)
        previous_pos = last_user_pos
        # First chunk: everything up to last user message, split at system boundary
        result = [
            {"role": "system", "content": full_template[:system_end]},
            {"role": "user", "content": full_template[system_end:last_user_pos]},
        ]
        message_range = range(last_user_idx + 1, len(messages))
    else:
        previous_pos = 0
        result = []
        message_range = range(len(messages))

    for i in message_range:
        # Parallel tool calls
        if messages[i]["role"] == "tool" and messages[i + 1]["role"] == "tool":
            continue

        # Render up to this message
        if (
            enable_thinking
            and messages[i]["role"] != "assistant"
            and ("reasoning_content" not in messages[i + 1] or messages[i + 1]["reasoning_content"] == "")
        ):
            # Manual hack for empty reasoning content mismatch
            template_up_to_here = tokenizer.apply_chat_template(
                messages[: i + 1],
                tokenize=False,
                add_generation_prompt=False,
                tools=tools,
                chat_template_kwargs={"enable_thinking": enable_thinking},
            )
            template_up_to_here += "<|im_start|>assistant\n<think></think>"
        else:
            # Tool and usermessages need generation prompt, others don't
            add_gen_prompt = messages[i]["role"] == "tool" or messages[i]["role"] == "user"
            template_up_to_here = tokenizer.apply_chat_template(
                messages[: i + 1],
                tokenize=False,
                add_generation_prompt=add_gen_prompt,
                tools=tools,
                chat_template_kwargs={"enable_thinking": enable_thinking},
            )

        current_pos = len(template_up_to_here)
        chunk_text = full_template[previous_pos:current_pos]

        # Verify incremental rendering matches full template
        if template_up_to_here != full_template[:current_pos]:
            raise ValueError(f"Template mismatch at message {i}: incremental rendering doesn't match full template")

        result.append({"role": messages[i]["role"], "content": chunk_text})
        previous_pos = current_pos

    return result


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL conversations to create masked message chunks")
    parser.add_argument(
        "input_file",
        nargs="?",
        default="/lustre/fsw/portfolios/llmservice/users/abukharin/nano-next/data_creation/Science/raw/part_00.jsonl",
        help="Input JSONL file",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="/lustre/fsw/portfolios/llmservice/users/abukharin/nano-next/data_creation/Science/mat/part_00_test.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument("--extra_info", action="store_true", default=False, help="Include extra info in the output")
    parser.add_argument(
        "--debug_failures",
        type=str,
        default="/lustre/fsw/portfolios/llmservice/users/abukharin/nano-next/data_creation/Science/mat/part_00_test_err.jsonl",
        help="Append failing raw records as JSONL to this path",
    )
    parser.add_argument("--pause_on_error", action="store_true", help="Pause on validation errors for inspection")
    args = parser.parse_args()

    # Create tokenizer ONCE
    tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    with open("chat_template_nemonext.jinja", "r") as f:
        tokenizer.chat_template = f.read()

    # Count total lines for progress bar (faster binary mode)
    print("📊 Counting total lines...")
    with open(args.input_file, "rb") as f:
        total_lines = sum(1 for _ in f)
    print(f"📊 Total lines: {total_lines:,}")

    # Process each line in the input JSONL
    passed_count = 0
    failed_count = 0

    # Add timing and better progress tracking
    start_time = time.time()

    with open(args.input_file, "r", buffering=8192) as infile, open(args.output_file, "w", buffering=8192) as outfile:
        with tqdm(total=total_lines, desc="Processing", unit="lines", unit_scale=True) as pbar:
            for line_num, line in enumerate(infile, 1):
                data = json.loads(line.strip())
                messages = data.get("messages", [])
                messages = replace_json_args(messages)

                #########################################################
                # Validation: message-level — some message has "<tool_call>" but none have "# Tools"
                # In this case we skip the line. If we are confident in our data, we can remove this block
                any_tool_call = any(
                    isinstance(m, dict)
                    and (
                        (isinstance(m.get("content"), str) and "<tool_call>" in m.get("content"))
                        or (
                            isinstance(m.get("reasoning_content"), str) and "<tool_call>" in m.get("reasoning_content")
                        )
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
                    failed_count += 1
                    tqdm.write(f"❌ Line {line_num}: Some message has <tool_call> but no message has # Tools")
                    if args.debug_failures:
                        try:
                            with open(args.debug_failures, "a") as dbg:
                                dbg.write(
                                    json.dumps(
                                        {
                                            "line_num": line_num,
                                            "reason": "message-level: <tool_call> present but # Tools missing",
                                            "tools": data.get("tools", None),
                                            "messages": messages,
                                        }
                                    )
                                    + "\n"
                                )
                        except Exception:
                            pass
                    if args.pause_on_error:
                        print("\n🔍 ORIGINAL MESSAGES:")
                        print(json.dumps(messages, indent=2, ensure_ascii=False))
                        input()
                    continue
                #########################################################

                # Extract tools from conversation data if present
                conversation_tools = data.get("tools", None)

                # Get masked messages for this conversation
                try:
                    masked_messages = create_masked_messages(messages, tokenizer, tools=conversation_tools)
                    if len(masked_messages) == 0:
                        failed_count += 1
                        tqdm.write(f"❌ Line {line_num}: No masked messages created")
                        continue
                except Exception as e:
                    failed_count += 1
                    tqdm.write(f"❌ Line {line_num}: Error creating masked messages - {str(e)[:100]}")
                    continue

                # Some simple sanity checks
                all_passed = True
                for i, (chunks, messages_i) in enumerate(masked_messages):
                    # Determine if this group has thinking
                    has_thinking = any("reasoning_content" in msg and msg["reasoning_content"] for msg in messages_i)
                    for j, obj in enumerate(chunks):
                        if j > 0 and obj["role"] == "assistant" and chunks[j - 1]["role"] == "assistant":
                            tqdm.write(f"❌ Line {line_num}: Multiple assistant turns detected")
                            print("\n🔍 ORIGINAL MESSAGES:")
                            print(json.dumps(messages_i, indent=2, ensure_ascii=False))
                            print("\n🔍 PROCESSED CHUNKS:")
                            print(json.dumps(chunks, indent=2, ensure_ascii=False))
                            all_passed = False
                            break
                        content = obj.get("content", "")
                        if obj["role"] == "user" and "<think>" not in content:
                            tqdm.write(f"❌ Line {line_num}: User message does not contain <think>")
                            print("\n🔍 ORIGINAL MESSAGES:")
                            print(json.dumps(messages_i, indent=2, ensure_ascii=False))
                            print("\n🔍 PROCESSED CHUNKS:")
                            print(json.dumps(chunks, indent=2, ensure_ascii=False))
                            all_passed = False
                            input()
                            break
                        if content.endswith("assistant\n<think><"):
                            tqdm.write(f"❌ Line {line_num}: Cutoff thinking detected in {obj['role']} message")
                            print("\n🔍 ORIGINAL MESSAGES:")
                            print(json.dumps(messages_i, indent=2, ensure_ascii=False))
                            print("\n🔍 PROCESSED CHUNKS:")
                            print(json.dumps(chunks, indent=2, ensure_ascii=False))
                            all_passed = False
                            break

                    # Render the expected template for this group
                    full_template_i = tokenizer.apply_chat_template(
                        messages_i,
                        tokenize=False,
                        add_generation_prompt=False,
                        tools=conversation_tools,
                        chat_template_kwargs={"enable_thinking": has_thinking},
                    )
                    # Rendered-level validation: '<tool_call>' present but '# Tools' missing
                    if ("<tool_call>" in full_template_i) and ("# Tools" not in full_template_i):
                        all_passed = False
                        tqdm.write(
                            f"❌ Line {line_num}: Rendered contains <tool_call> but missing # Tools (group {i + 1})"
                        )
                        if args.debug_failures:
                            try:
                                with open(args.debug_failures, "a") as dbg:
                                    dbg.write(
                                        json.dumps(
                                            {
                                                "line_num": line_num,
                                                "reason": "rendered-level: <tool_call> present but # Tools missing",
                                                "tools": data.get("tools", None),
                                                "messages": messages,
                                            }
                                        )
                                        + "\n"
                                    )
                            except Exception:
                                pass
                        if args.pause_on_error:
                            print("\n🔍 ORIGINAL MESSAGES:")
                            print(json.dumps(messages, indent=2, ensure_ascii=False))
                            print("\n🔍 RENDERED TEMPLATE:")
                            print(full_template_i[:2000])
                            input()
                        break

                    # Concatenate the chunks
                    concatenated = "".join([chunk["content"] for chunk in chunks])

                    # Verify they match
                    if full_template_i != concatenated:
                        all_passed = False
                        tqdm.write(f"❌ Line {line_num}: Template mismatch in group {i + 1}")
                        break

                if all_passed:
                    for chunks, _ in masked_messages:
                        if args.extra_info:
                            output_data = {
                                "messages": chunks,
                                **{k: v for k, v in data.items() if k != "messages"},  # Keep other fields
                            }
                        else:
                            output_data = {
                                "messages": chunks,
                            }

                        outfile.write(json.dumps(output_data) + "\n")
                    passed_count += 1
                else:
                    failed_count += 1

                # Update progress bar with current stats and throughput
                elapsed = time.time() - start_time
                rate = line_num / elapsed if elapsed > 0 else 0
                pbar.set_postfix(passed=passed_count, failed=failed_count, rate=f"{rate:.1f}/s")
                pbar.update(1)

    # Final summary
    total_time = time.time() - start_time
    final_rate = total_lines / total_time if total_time > 0 else 0
    success_rate = (passed_count / total_lines * 100) if total_lines > 0 else 0

    print("\n🏁 Processing complete!")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"📊 Processed: {total_lines:,} lines")
    print(f"✅ Passed: {passed_count:,} ({success_rate:.1f}%)")
    print(f"❌ Failed: {failed_count:,}")
    print(f"🚀 Average rate: {final_rate:.1f} lines/sec")
