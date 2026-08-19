from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class Prompt:
    text: str
    statement_spans: list[tuple[int, int]]
    query_span: tuple[int, int]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _render_demo(row: dict[str, Any]) -> str:
    answer = "True" if row["answer"] else "False"
    return " ".join([*row["statements"], row["question"], "True or False?", answer]) + "\n"


def build_prompt(row: dict[str, Any], demos: list[dict[str, Any]]) -> Prompt:
    pieces = [_render_demo(demo) for demo in demos]
    text = "".join(pieces)
    statement_spans: list[tuple[int, int]] = []
    for statement in row["statements"]:
        if text and not text.endswith((" ", "\n")):
            text += " "
        start = len(text)
        text += statement
        statement_spans.append((start, len(text)))
        text += " "
    query_start = len(text)
    text += row["question"] + " True or False?"
    return Prompt(text=text, statement_spans=statement_spans, query_span=(query_start, len(text)))


def build_chat_multiturn_prompt(tokenizer: Any, row: dict[str, Any], demos: list[dict[str, Any]]) -> Prompt:
    """Render the fixed ICL examples as native user/assistant turns.

    No explicit system message is supplied: Qwen2.5-Instruct's official
    template therefore inserts its own default system message.
    """
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("chat-multiturn requires a tokenizer with a chat template")
    messages: list[dict[str, str]] = []
    for demo in demos:
        messages.extend(
            [
                {"role": "user", "content": build_prompt(demo, []).text},
                {"role": "assistant", "content": "True" if demo["answer"] else "False"},
            ]
        )
    target = build_prompt(row, [])
    messages.append({"role": "user", "content": target.text})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str):
        raise TypeError("apply_chat_template(tokenize=False) did not return text")
    offset = rendered.rfind(target.text)
    if offset < 0:
        raise ValueError("Could not locate the final user content in the rendered chat prompt")
    return Prompt(
        text=rendered,
        statement_spans=[(start + offset, end + offset) for start, end in target.statement_spans],
        query_span=(target.query_span[0] + offset, target.query_span[1] + offset),
    )


def render_prompt(tokenizer: Any, row: dict[str, Any], demos: list[dict[str, Any]], prompt_format: str) -> Prompt:
    if prompt_format == "raw":
        return build_prompt(row, demos)
    if prompt_format == "chat-multiturn":
        return build_chat_multiturn_prompt(tokenizer, row, demos)
    raise ValueError(f"Unsupported prompt format: {prompt_format}")


def answer_candidates(prompt_format: str) -> tuple[str, str]:
    if prompt_format == "raw":
        return " True", " False"
    if prompt_format == "chat-multiturn":
        return "True", "False"
    raise ValueError(f"Unsupported prompt format: {prompt_format}")


def _indices_for_span(offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    start, end = span
    return [index for index, (token_start, token_end) in enumerate(offsets) if token_end > start and token_start < end]


def _candidate_logprob(model: Any, tokenizer: Any, prompt_ids: torch.Tensor, prompt_text: str, candidate: str) -> float:
    full = tokenizer(prompt_text + candidate, return_tensors="pt", add_special_tokens=False).input_ids.to(prompt_ids.device)
    prompt_len = prompt_ids.shape[1]
    if full.shape[1] <= prompt_len or not torch.equal(full[:, :prompt_len], prompt_ids):
        raise ValueError("Candidate tokenization did not retain the prompt prefix")
    candidate_ids = full[:, prompt_len:]
    with torch.inference_mode():
        logits = model(input_ids=full, use_cache=False).logits[:, prompt_len - 1 : -1]
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, candidate_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_logprobs.mean().item())


def _load_model(model_id: str, device: torch.device) -> tuple[Any, Any, str]:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # The random-weight control must not need a network lookup: on restricted
    # compute nodes it reuses the tokenizer and architecture from a local
    # checkpoint selected by RANDOM_MODEL_REFERENCE.
    reference_id = os.environ.get("RANDOM_MODEL_REFERENCE") if model_id == "__random__" else model_id
    if not reference_id:
        raise ValueError("__random__ requires RANDOM_MODEL_REFERENCE to point to a local Qwen snapshot")
    tokenizer = AutoTokenizer.from_pretrained(reference_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model_id == "__random__":
        config = AutoConfig.from_pretrained(reference_id)
        # Initializing 7B bfloat16 parameters on CPU is disproportionately slow
        # on the shared server.  Allocate directly on the target GPU instead;
        # this remains an untrained random-weight control.
        with torch.device(device):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="eager", low_cpu_mem_usage=True
        )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, str(dtype).replace("torch.", "")


def _extract_row(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    demos: list[dict[str, Any]],
    max_length: int,
    prompt_format: str,
) -> list[dict[str, Any]]:
    prompt = render_prompt(tokenizer, row, demos, prompt_format)
    encoded = tokenizer(
        prompt.text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")[0].tolist()]
    input_ids = encoded["input_ids"]
    if input_ids.shape[1] > max_length:
        raise ValueError(f"prompt has {input_ids.shape[1]} tokens, exceeds max_length={max_length}")
    query_indices = _indices_for_span(offsets, prompt.query_span)
    statement_indices = [_indices_for_span(offsets, span) for span in prompt.statement_spans]
    if not query_indices or any(not indices for indices in statement_indices):
        raise ValueError("Could not align all prompt spans to tokenizer offsets")

    device = next(model.parameters()).device
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded, output_attentions=True, use_cache=False)
    attentions = output.attentions
    features: list[list[float]] = [[] for _ in statement_indices]
    for layer_attention in attentions:
        # [heads, query tokens, statement tokens] -> heads -> one scalar per statement.
        layer = layer_attention[0]
        for statement_number, token_indices in enumerate(statement_indices):
            value = layer[:, query_indices, :][:, :, token_indices].mean(dim=-1).max(dim=-1).values.mean()
            features[statement_number].append(float(value.cpu()))
    del output, attentions

    candidate_true, candidate_false = answer_candidates(prompt_format)
    score_true = _candidate_logprob(model, tokenizer, encoded["input_ids"], prompt.text, candidate_true)
    score_false = _candidate_logprob(model, tokenizer, encoded["input_ids"], prompt.text, candidate_false)
    predicted_true = score_true > score_false
    return [
        {
            "example_id": row["example_id"],
            "theory_group": row["theory_group"],
            "statement_index": index,
            "statement_count": row["statement_count"],
            "proof_depth": row["proof_depth"],
            "gold_useful": row["useful"][index],
            "gold_height": row["heights"][index],
            "answer": row["answer"],
            "prediction": predicted_true,
            "correct": predicted_true == row["answer"],
            "score_true": score_true,
            "score_false": score_false,
            "prompt_format": prompt_format,
            "features": features[index],
        }
        for index in range(len(row["statements"]))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract compact attention features from a frozen Qwen checkpoint")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-id", required=True, help="Hugging Face ID, local snapshot path, or __random__")
    parser.add_argument("--artifact-name", help="Stable output name; required for local snapshot paths")
    parser.add_argument("--sample-name", default="pilot", help="Frozen sample name created by prepare")
    parser.add_argument("--prompt-format", choices=("raw", "chat-multiturn"), default="raw")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    safe_name = args.artifact_name or args.model_id.replace("/", "--").replace("__", "") or "random"
    output_path = args.root / "artifacts" / f"features-{safe_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
    frozen = args.root / "data" / "frozen"
    sample_path = frozen / f"{args.sample_name}.jsonl"
    demo_path = frozen / ("icl_demos.json" if args.sample_name == "pilot" else f"{args.sample_name}-icl_demos.json")
    rows = _load_jsonl(sample_path)
    demos = json.loads(demo_path.read_text())
    device = torch.device("cuda")
    if args.model_id == "__random__":
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    model, tokenizer, dtype = _load_model(args.model_id, device)
    metadata_path = output_path.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(
            {
                "model": args.model_id,
                "dtype": dtype,
                "examples": len(rows),
                "sample_name": args.sample_name,
                "prompt_format": args.prompt_format,
                "candidate_strings": list(answer_candidates(args.prompt_format)),
                "uses_default_system_message": args.prompt_format == "chat-multiturn",
                "chat_template_sha256": (
                    hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
                    if args.prompt_format == "chat-multiturn"
                    else None
                ),
                "random_seed": args.random_seed if args.model_id == "__random__" else None,
            },
            indent=2,
        )
    )
    skipped: list[dict[str, str]] = []
    with output_path.open("w") as handle:
        for number, row in enumerate(rows, start=1):
            try:
                for feature_row in _extract_row(
                    model, tokenizer, row, demos, args.max_length, args.prompt_format
                ):
                    handle.write(json.dumps(feature_row) + "\n")
            except Exception as error:  # Record, don't silently alter the sample.
                skipped.append({"example_id": row["example_id"], "error": repr(error)})
            if number % 10 == 0:
                print(f"processed {number}/{len(rows)}")
                torch.cuda.empty_cache()
    (output_path.with_suffix(".skipped.json")).write_text(json.dumps(skipped, indent=2))
    del model
    torch.cuda.empty_cache()
    print(json.dumps({"output": str(output_path), "skipped": len(skipped)}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
