from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from huggingface_hub import snapshot_download

from .proof import labels_for_statements


DATASET_ID = "yyyyifan/MechanisticProbe_ProofWriter_ARC"
BUCKETS = (1, 2, 4, 8, 12, 16, 20, 24)


def _answer_is_true(answer: Any) -> bool:
    if isinstance(answer, bool):
        return answer
    return str(answer).strip().lower() in {"true", "1", "yes"}


def _records_from_file(path: Path, split: str) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for theory_number, line in enumerate(handle):
            raw = json.loads(line)
            triples = {key: value["text"] for key, value in raw["triples"].items()}
            rules = {key: value["text"] for key, value in raw["rules"].items()}
            statement_keys = [*triples, *rules]
            statement_texts = [*triples.values(), *rules.values()]
            group_id = hashlib.sha256(raw["theory"].encode()).hexdigest()[:16]
            for question_id, question in raw["questions"].items():
                depth_text = str(question.get("QDep", ""))
                if not question.get("QLen") or "NAF" in question.get("proofs", ""):
                    continue
                if depth_text not in {"0", "1"}:
                    continue
                useful, heights = labels_for_statements(statement_keys, question["proofs"], int(depth_text))
                # MechanisticProbe's labels must map entirely to source statements.
                if not any(useful) or any(height not in {None, 0, 1} for height in heights):
                    continue
                yield {
                    "example_id": f"{split}:{path.parent.name}:{theory_number}:{question_id}",
                    "theory_group": group_id,
                    "split": split,
                    "context": raw["theory"],
                    "statement_keys": statement_keys,
                    "statements": statement_texts,
                    "question": question["question"],
                    "answer": _answer_is_true(question["answer"]),
                    "proof": question["proofs"],
                    "proof_depth": int(depth_text),
                    "useful": useful,
                    "heights": heights,
                    "statement_count": len(statement_keys),
                }


def _records_from_processed(path: Path, split: str) -> Iterator[dict[str, Any]]:
    """Load the authors' already processed CWA JSON release without regenerating it."""
    raw_by_depth = json.loads(path.read_text())
    for depth_text, examples in raw_by_depth.items():
        if str(depth_text) not in {"0", "1"}:
            continue
        for index, raw in enumerate(examples):
            statement_keys = [*raw["triples"], *raw["rules"]]
            statement_texts = [*raw["triples"].values(), *raw["rules"].values()]
            useful, heights = labels_for_statements(statement_keys, raw["proof"], int(depth_text))
            if not any(useful) or any(height not in {None, 0, 1} for height in heights):
                continue
            yield {
                "example_id": f"{split}:processed:{depth_text}:{index}",
                "theory_group": hashlib.sha256(raw["context"].encode()).hexdigest()[:16],
                "split": split,
                "context": raw["context"],
                "statement_keys": statement_keys,
                "statements": statement_texts,
                "question": raw["question"],
                "answer": _answer_is_true(raw["answer"]),
                "proof": raw["proof"],
                "proof_depth": int(depth_text),
                "useful": useful,
                "heights": heights,
                "statement_count": len(statement_keys),
            }


def _find_meta_files(source: Path, split: str) -> list[Path]:
    files = sorted(source.rglob(f"meta-{split}.jsonl"))
    if not files:
        raise FileNotFoundError(f"Could not find ProofWriter meta-{split}.jsonl below {source}")
    return files


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _select_demos(train_rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    # Keep demonstrations short so every test prompt has room for the full theory.
    eligible = [row for row in train_rows if row["statement_count"] <= 4]
    by_label: dict[bool, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_label[row["answer"]].append(row)
    rng = random.Random(seed)
    for values in by_label.values():
        rng.shuffle(values)
    if len(by_label[True]) < 2 or len(by_label[False]) < 2:
        raise RuntimeError("Insufficient short, balanced ICL examples")
    return [by_label[True][0], by_label[False][0], by_label[True][1], by_label[False][1]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and freeze a ProofWriter MechanisticProbe sample")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit-per-bucket", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-name", default="pilot", help="Name of the frozen sample without an extension")
    parser.add_argument("--source", type=Path, help="Directory containing CWA/{train,dev,test}.json")
    args = parser.parse_args()

    data_root = args.root / "data"
    source = args.source or data_root / "source"
    frozen = data_root / "frozen"
    source.mkdir(parents=True, exist_ok=True)
    frozen.mkdir(parents=True, exist_ok=True)
    processed_root = source / "CWA"
    if not all((processed_root / f"{split}.json").exists() for split in ("train", "dev", "test")):
        snapshot_download(repo_id=DATASET_ID, repo_type="dataset", local_dir=source)
        processed_root = source / "data" / "proofwriter" / "CWA"

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev", "test"):
        processed = processed_root / f"{split}.json"
        if processed.exists():
            rows = list(_records_from_processed(processed, split))
        else:
            rows = [row for path in _find_meta_files(source, split) for row in _records_from_file(path, split)]
        rows_by_split[split] = rows

    rng = random.Random(args.seed)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_by_split["test"]:
        if row["statement_count"] in BUCKETS:
            grouped[row["statement_count"]].append(row)
    selected: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"seed": args.seed, "limit_per_bucket": args.limit_per_bucket, "buckets": {}}
    for bucket in BUCKETS:
        values = grouped[bucket]
        rng.shuffle(values)
        chosen = values[: args.limit_per_bucket]
        selected.extend(chosen)
        manifest["buckets"][str(bucket)] = {"available": len(values), "selected": len(chosen)}

    demos = _select_demos(rows_by_split["train"], args.seed)
    if not args.sample_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("sample-name may contain only letters, numbers, dashes, and underscores")
    sample_path = frozen / f"{args.sample_name}.jsonl"
    demo_path = frozen / ("icl_demos.json" if args.sample_name == "pilot" else f"{args.sample_name}-icl_demos.json")
    manifest_path = frozen / ("manifest.json" if args.sample_name == "pilot" else f"{args.sample_name}-manifest.json")
    manifest["sample_name"] = args.sample_name
    _write_jsonl(sample_path, selected)
    demo_path.write_text(json.dumps(demos, indent=2, sort_keys=True))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
