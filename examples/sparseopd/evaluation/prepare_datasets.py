#!/usr/bin/env python3
"""Normalize the paper's math evaluation sets into dataset/eval/*.parquet."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUTS = {
    "aime24": REPO_ROOT / "dataset/aime_2024/data/train-00000-of-00001.parquet",
    "aime25": REPO_ROOT / "dataset/aime_2025/data/train-00000-of-00001.parquet",
    "hmmt25": REPO_ROOT / "dataset/hmmt_feb_2025/data/train-00000-of-00001.parquet",
}
OUTPUT_NAMES = {
    "aime24": "aime_2024.parquet",
    "aime25": "aime_2025.parquet",
    "hmmt25": "hmmt_feb_2025.parquet",
}
PROBLEM_FIELDS = ("problem", "Problem", "question", "Question")
ANSWER_FIELDS = ("answer", "Answer", "gold_answer", "ground_truth")
ID_FIELDS = ("problem_idx", "problem_id", "id", "ID")


def first_field(row: dict[str, Any], names: tuple[str, ...], field: str) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    raise ValueError(f"No {field} field found. Tried: {', '.join(names)}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    import datasets

    if not path.is_file():
        raise FileNotFoundError(f"Evaluation source dataset not found: {path}")
    suffix = path.suffix.lower()
    loader = {".parquet": "parquet", ".json": "json", ".jsonl": "json", ".csv": "csv"}.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported dataset format {suffix!r}: {path}")
    loaded = datasets.load_dataset(loader, data_files=str(path), split="train")
    return [dict(row) for row in loaded]


def normalize_rows(rows: list[dict[str, Any]], dataset_name: str) -> list[dict[str, str]]:
    normalized = []
    for index, row in enumerate(rows):
        problem = first_field(row, PROBLEM_FIELDS, "problem")
        answer = first_field(row, ANSWER_FIELDS, "answer")
        identity = next((row[name] for name in ID_FIELDS if row.get(name) is not None), index)
        normalized.append({"problem_id": str(identity), "problem": str(problem), "answer": str(answer)})
    if not normalized:
        raise ValueError(f"Dataset {dataset_name} is empty")
    return normalized


def prepare_one(name: str, source: Path, output_dir: Path, overwrite: bool) -> Path:
    import datasets

    destination = output_dir / OUTPUT_NAMES[name]
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination}; pass --overwrite to replace it")
    rows = normalize_rows(load_rows(source), name)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(rows).to_parquet(str(destination))
    print(f"{name}: wrote {len(rows)} rows to {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULT_INPUTS.items():
        parser.add_argument(f"--{name}", type=Path, default=default, help=f"Raw {name} file")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "dataset/eval")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name in DEFAULT_INPUTS:
        prepare_one(name, getattr(args, name), args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
