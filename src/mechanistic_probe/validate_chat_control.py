from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .extract import _indices_for_span, _load_jsonl, build_prompt, render_prompt


REFERENCE_ARTIFACTS = ("base-formal", "instruct-formal", "random-formal")
CHAT_ARTIFACT = "instruct-chat-formal"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_paths(root: Path) -> list[Path]:
    paths = [
        root / "data" / "frozen" / "formal.jsonl",
        root / "data" / "frozen" / "formal-icl_demos.json",
        root / "data" / "frozen" / "formal-manifest.json",
    ]
    for name in REFERENCE_ARTIFACTS:
        paths.extend(
            [
                root / "artifacts" / f"features-{name}.jsonl",
                root / "artifacts" / f"features-{name}.meta.json",
                root / "artifacts" / f"features-{name}.skipped.json",
            ]
        )
    return paths


def _signature(path: Path) -> list[tuple[Any, ...]]:
    signature: list[tuple[Any, ...]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            signature.append(
                (
                    row["example_id"],
                    int(row["statement_index"]),
                    int(row["gold_useful"]),
                    None if row["gold_height"] is None else int(row["gold_height"]),
                    bool(row["answer"]),
                )
            )
    return signature


def _baseline_path(root: Path) -> Path:
    return root / "results" / "chat-control-baseline.json"


def preflight(root: Path, model_id: str, max_length: int) -> dict[str, Any]:
    frozen = root / "data" / "frozen"
    rows = _load_jsonl(frozen / "formal.jsonl")
    demos = json.loads((frozen / "formal-icl_demos.json").read_text())
    if len(rows) != 6277:
        raise ValueError(f"Expected 6277 frozen examples, found {len(rows)}")
    if len(demos) != 4:
        raise ValueError(f"Expected four fixed ICL demonstrations, found {len(demos)}")
    statement_rows = sum(int(row["statement_count"]) for row in rows)
    if statement_rows != 85820:
        raise ValueError(f"Expected 85820 statement rows, found {statement_rows}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, use_fast=True)
    maximum_tokens = 0
    for row in rows:
        prompt = render_prompt(tokenizer, row, demos, "chat-multiturn")
        target = build_prompt(row, [])
        target_offset = prompt.text.rfind(target.text)
        if target_offset < 0:
            raise ValueError(f"Final user content missing for {row['example_id']}")
        for index, span in enumerate(prompt.statement_spans):
            if prompt.text[slice(*span)] != row["statements"][index]:
                raise ValueError(f"Statement span mismatch for {row['example_id']} statement {index}")
        if prompt.text[slice(*prompt.query_span)] != target.text[slice(*target.query_span)]:
            raise ValueError(f"Query span mismatch for {row['example_id']}")
        encoded = tokenizer(
            prompt.text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        if not _indices_for_span(offsets, prompt.query_span):
            raise ValueError(f"Query tokens missing for {row['example_id']}")
        if any(not _indices_for_span(offsets, span) for span in prompt.statement_spans):
            raise ValueError(f"Statement tokens missing for {row['example_id']}")
        maximum_tokens = max(maximum_tokens, len(encoded["input_ids"]))
    if maximum_tokens > max_length:
        raise ValueError(f"Maximum chat prompt length {maximum_tokens} exceeds {max_length}")

    protected = _protected_paths(root)
    missing = [str(path) for path in protected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected inputs are missing: {missing}")
    reference_signature = _signature(root / "artifacts" / "features-base-formal.jsonl")
    for name in REFERENCE_ARTIFACTS[1:]:
        candidate = _signature(root / "artifacts" / f"features-{name}.jsonl")
        if candidate != reference_signature:
            raise ValueError(f"Reference artifact {name} is not row-aligned with base-formal")

    result = {
        "phase": "preflight",
        "examples": len(rows),
        "statement_rows": statement_rows,
        "demos": len(demos),
        "maximum_chat_tokens": maximum_tokens,
        "max_length": max_length,
        "prompt_format": "chat-multiturn",
        "protected_sha256": {str(path.relative_to(root)): _sha256(path) for path in protected},
    }
    path = _baseline_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def postflight(root: Path) -> dict[str, Any]:
    baseline = json.loads(_baseline_path(root).read_text())
    protected = _protected_paths(root)
    current_hashes = {str(path.relative_to(root)): _sha256(path) for path in protected}
    if current_hashes != baseline["protected_sha256"]:
        changed = sorted(
            name
            for name, digest in current_hashes.items()
            if baseline["protected_sha256"].get(name) != digest
        )
        raise ValueError(f"Protected formal inputs or artifacts changed: {changed}")

    artifact = root / "artifacts" / f"features-{CHAT_ARTIFACT}.jsonl"
    metadata = json.loads(
        (root / "artifacts" / f"features-{CHAT_ARTIFACT}.meta.json").read_text()
    )
    skipped = json.loads(
        (root / "artifacts" / f"features-{CHAT_ARTIFACT}.skipped.json").read_text()
    )
    if skipped:
        raise ValueError(f"Chat extraction skipped {len(skipped)} examples")
    expected_metadata = {
        "examples": 6277,
        "sample_name": "formal",
        "prompt_format": "chat-multiturn",
        "candidate_strings": ["True", "False"],
        "uses_default_system_message": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Chat metadata mismatch: {mismatches}")

    reference_signature = _signature(root / "artifacts" / "features-base-formal.jsonl")
    chat_signature = _signature(artifact)
    if chat_signature != reference_signature:
        raise ValueError("Chat artifact is not row- and label-aligned with base-formal")
    if len(chat_signature) != 85820:
        raise ValueError(f"Expected 85820 chat rows, found {len(chat_signature)}")

    result = {
        "phase": "postflight",
        "examples": metadata["examples"],
        "statement_rows": len(chat_signature),
        "skipped_examples": len(skipped),
        "reference_artifacts_unchanged": True,
        "row_and_label_alignment": True,
        "chat_artifact_sha256": _sha256(artifact),
        "metadata": metadata,
    }
    output = root / "results" / "chat-control-validation.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the native Instruct chat-template control")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "postflight"), required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    if args.phase == "preflight":
        if not args.model_id:
            parser.error("--model-id is required for preflight")
        result = preflight(args.root, args.model_id, args.max_length)
    else:
        result = postflight(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
