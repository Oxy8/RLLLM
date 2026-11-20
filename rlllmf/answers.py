"""Extract final answer text while keeping model reasoning out of judge requests."""

from __future__ import annotations

import re


def final_answer(text: str) -> str:
    # Discard both complete reasoning blocks and a truncated, unclosed block.
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Some templates prefill the opening tag, leaving only the close in generation.
    endings = list(re.finditer(r"</think>", text, flags=re.IGNORECASE))
    if endings:
        text = text[endings[-1].end() :]
    return text.strip()


def decode_final_answer(tokenizer, ids, *, thinking_prefilled=False) -> str:
    # Decode with delimiters intact: skip_special_tokens can erase reasoning
    # boundaries before we get a chance to remove the text between them.
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    # The opening tag is already in the prompt in thinking mode. Restore its
    # parsing context so unfinished reasoning can never be mistaken for an answer.
    text = final_answer(("<think>" if thinking_prefilled else "") + raw)
    for token in sorted(tokenizer.all_special_tokens, key=len, reverse=True):
        text = text.replace(token, "")
    return text.strip()


def thinking_is_prefilled(tokenizer, enabled: bool) -> bool:
    if not enabled or not hasattr(tokenizer, "apply_chat_template"):
        return False
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Task"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enabled,
    )
    return isinstance(prefix, str) and prefix.rstrip().endswith("<think>")


def limit_answer_tokens(tokenizer, text: str, limit: int) -> str:
    """Limit only final-answer text, in the local model tokenizer's units."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= limit:
        return text
    end = limit
    while end:
        clipped = tokenizer.decode(ids[:end], skip_special_tokens=True)
        # Decoding a cut byte sequence can introduce a replacement character that
        # re-encodes into extra tokens. Verify the actual string sent to the API.
        if len(tokenizer.encode(clipped, add_special_tokens=False)) <= limit:
            return clipped
        end -= 1
    return ""


def generation_diagnostics(
    tokenizer, ids, max_tokens: int, *, thinking_prefilled=False, max_answer_tokens=None
) -> dict:
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    unfinished = thinking_prefilled
    for tag in re.findall(r"</?think>", raw, flags=re.IGNORECASE):
        unfinished = tag.lower() == "<think>"
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = eos if isinstance(eos, list) else [eos]
    ended = bool(len(ids) and int(ids[-1]) in eos_ids)
    answer = decode_final_answer(tokenizer, ids, thinking_prefilled=thinking_prefilled)
    info = {
        "generated_tokens": len(ids),
        "hit_token_limit": len(ids) >= max_tokens and not ended,
        "thinking_detected": thinking_prefilled
        or bool(re.search(r"</?think>", raw, flags=re.IGNORECASE)),
        "unfinished_thinking": unfinished,
        "empty_final_answer": not bool(answer),
    }
    if max_answer_tokens is not None:
        capped = limit_answer_tokens(tokenizer, answer, max_answer_tokens)
        info.update(
            {
                "final_answer_tokens": len(tokenizer.encode(answer, add_special_tokens=False)),
                "judge_answer_tokens": len(tokenizer.encode(capped, add_special_tokens=False)),
                "answer_truncated_for_judge": capped != answer,
                "answer_tokenizer": "local_policy",
            }
        )
    return info
