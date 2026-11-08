#!/usr/bin/env python3
"""Generate and merge benchmark responses; scoring is handled by evaluate.py."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
DATASET_LABELS = {"aime24": "AIME 2024", "aime25": "AIME 2025", "hmmt25": "HMMT-Feb-2025"}

MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("experiments"), dict):
        raise ValueError(f"Invalid evaluation config: {path}")
    return config


def find_checkpoint(config: dict[str, Any], experiment: str) -> Path:
    exp_cfg = config["experiments"][experiment] or {}
    configured = exp_cfg.get("model_path")
    if configured:
        path = Path(str(configured).format(experiment=experiment, prefix=config["experiment_prefix"]))
        return path if path.is_absolute() else REPO_ROOT / path
    checkpoint_root = Path(config["checkpoint_root"].format(experiment=experiment, prefix=config["experiment_prefix"]))
    if not checkpoint_root.is_absolute():
        checkpoint_root = REPO_ROOT / checkpoint_root
    latest_file = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not latest_file.is_file():
        raise FileNotFoundError(f"Checkpoint marker not found: {latest_file}")
    step = latest_file.read_text(encoding="utf-8").strip()
    model_path = checkpoint_root / f"hf_model_step_{step}"
    if not model_path.is_dir():
        raise FileNotFoundError(f"Merged Hugging Face checkpoint not found: {model_path}")
    return model_path


def evaluation_signature(experiment: str, config: dict[str, Any]) -> dict[str, Any]:
    datasets = {}
    for key, configured_path in config["data"].items():
        data_path = Path(configured_path)
        if not data_path.is_absolute():
            data_path = REPO_ROOT / data_path
        state: dict[str, Any] = {"path": str(data_path)}
        if data_path.is_file():
            stat = data_path.stat()
            state.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        datasets[key] = state
    return {
        "model_path": str(find_checkpoint(config, experiment)),
        "generation": config["generation"],
        "data": datasets,
    }


def load_eval_data(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared evaluation data not found: {path}. Run evaluation/prepare_datasets.py first."
        )
    import datasets

    rows = datasets.load_dataset("parquet", data_files=str(path), split="train")
    return [
        {"problem_id": str(row["problem_id"]), "problem": str(row["problem"]), "answer": str(row["answer"])}
        for row in rows
    ]


def generated_dataset_is_complete(path: Path, rows: list[dict[str, str]], sample_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as stream:
            generated_rows = [json.loads(line) for line in stream]
        return len(generated_rows) == len(rows) and all(
            generated["problem_id"] == expected["problem_id"]
            and generated["problem"] == expected["problem"]
            and generated["answer"] == expected["answer"]
            and len(generated["outputs"]) == sample_count
            for generated, expected in zip(generated_rows, rows, strict=True)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def experiment_generation_is_complete(experiment: str, config: dict[str, Any], output_dir: Path) -> bool:
    generation = config["generation"]
    raw_dir = output_dir / "generations" / experiment
    for dataset_key in DATASET_LABELS:
        data_path = Path(config["data"][dataset_key])
        if not data_path.is_absolute():
            data_path = REPO_ROOT / data_path
        rows = load_eval_data(data_path)
        count_key = "samples_for_avg_at_8" if dataset_key == "hmmt25" else "samples_for_pass_at_k"
        if not generated_dataset_is_complete(raw_dir / f"{dataset_key}.jsonl", rows, int(generation[count_key])):
            return False
    return True


def shard_directory(output_dir: Path, experiment: str, num_shards: int, shard_id: int) -> Path:
    return output_dir / "generations" / experiment / "shards" / str(num_shards) / str(shard_id)


def merge_generation_shards(
    experiment: str, config: dict[str, Any], output_dir: Path, num_shards: int, force: bool = False
) -> None:
    """Validate all shards before atomically publishing merged generation files."""
    signature = evaluation_signature(experiment, config)
    raw_dir = output_dir / "generations" / experiment
    for dataset_key in DATASET_LABELS:
        data_path = Path(config["data"][dataset_key])
        if not data_path.is_absolute():
            data_path = REPO_ROOT / data_path
        rows = load_eval_data(data_path)
        count_key = "samples_for_avg_at_8" if dataset_key == "hmmt25" else "samples_for_pass_at_k"
        sample_count = int(config["generation"][count_key])
        destination = raw_dir / f"{dataset_key}.jsonl"
        if not force and generated_dataset_is_complete(destination, rows, sample_count):
            continue
        merged: list[dict[str, Any] | None] = [None] * len(rows)
        for shard_id in range(num_shards):
            directory = shard_directory(output_dir, experiment, num_shards, shard_id)
            manifest = directory / "manifest.json"
            if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != signature:
                raise ValueError(f"Missing or mismatched generation manifest: {manifest}")
            source = directory / f"{dataset_key}.jsonl"
            if not generated_dataset_is_complete(source, rows[shard_id::num_shards], sample_count):
                raise ValueError(f"Incomplete generation shard: {source}")
            with source.open(encoding="utf-8") as stream:
                for index, line in zip(range(shard_id, len(rows), num_shards), stream, strict=True):
                    merged[index] = json.loads(line)
        if any(row is None for row in merged):
            raise ValueError(f"Missing rows when merging {experiment}/{dataset_key}")
        temporary = destination.with_suffix(".jsonl.tmp")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8") as stream:
            for row in merged:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, destination)
        print(f"Merged {num_shards} shards, {len(rows)} prompts: {destination}")


def generate_experiment(
    experiment: str,
    config: dict[str, Any],
    output_dir: Path,
    force: bool = False,
    num_shards: int = 1,
    shard_id: int = 0,
) -> None:
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_id < num_shards")
    model_path = find_checkpoint(config, experiment)
    generation = config["generation"]
    canonical_dir = output_dir / "generations" / experiment
    raw_dir = canonical_dir if num_shards == 1 else shard_directory(output_dir, experiment, num_shards, shard_id)
    signature = evaluation_signature(experiment, config)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = raw_dir / "manifest.json"
    matching_shards = False
    if num_shards > 1 and manifest.is_file():
        try:
            matching_shards = json.loads(manifest.read_text(encoding="utf-8")) == signature
        except (OSError, ValueError):
            pass
    if num_shards > 1 and (force or not matching_shards):
        for dataset_key in DATASET_LABELS:
            (raw_dir / f"{dataset_key}.jsonl").unlink(missing_ok=True)
        temporary_manifest = manifest.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(signature), encoding="utf-8")
        os.replace(temporary_manifest, manifest)

    pending = []
    for dataset_key in DATASET_LABELS:
        data_path = Path(config["data"][dataset_key])
        if not data_path.is_absolute():
            data_path = REPO_ROOT / data_path
        rows = load_eval_data(data_path)
        count_key = "samples_for_avg_at_8" if dataset_key == "hmmt25" else "samples_for_pass_at_k"
        sample_count = int(generation[count_key])
        if num_shards > 1:
            if not force and generated_dataset_is_complete(canonical_dir / f"{dataset_key}.jsonl", rows, sample_count):
                print(f"Already generated {experiment}/{dataset_key}; skipping inference.")
                continue
            rows = rows[shard_id::num_shards]
        raw_path = raw_dir / f"{dataset_key}.jsonl"
        if not force and generated_dataset_is_complete(raw_path, rows, sample_count):
            print(f"Already generated {experiment}/{dataset_key}; skipping inference.")
            continue
        if not rows:
            raw_path.write_text("", encoding="utf-8")
            continue
        pending.append((dataset_key, rows, sample_count, raw_path))
    if not pending:
        return

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("Inference requires transformers and vLLM in the configured environment") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype=generation.get("dtype", "bfloat16"),
        max_model_len=int(generation.get("max_model_len", 9217)),
        gpu_memory_utilization=float(generation.get("gpu_memory_utilization", 0.9)),
    )
    for dataset_key, rows, sample_count, raw_path in pending:
        prompts = []
        for row in rows:
            prompt = f"{row['problem']}\n{MATH_INSTRUCTION}"
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        sampling = SamplingParams(
            n=sample_count,
            temperature=float(generation.get("temperature", 1.0)),
            top_p=float(generation.get("top_p", 1.0)),
            max_tokens=int(generation.get("max_tokens", 8192)),
            seed=int(generation.get("seed", 42)) + shard_id,
        )
        generated = llm.generate(prompts, sampling, use_tqdm=True)
        temporary_raw_path = raw_path.with_suffix(".jsonl.tmp")
        with temporary_raw_path.open("w", encoding="utf-8") as stream:
            for row, request in zip(rows, generated, strict=True):
                stream.write(
                    json.dumps(
                        {
                            "problem_id": row["problem_id"],
                            "problem": row["problem"],
                            "answer": row["answer"],
                            "outputs": [candidate.text for candidate in request.outputs],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        os.replace(temporary_raw_path, raw_path)
        print(f"Generated {len(rows)} prompts x {sample_count} samples: {raw_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="*", help="Experiment names; defaults to every configured experiment")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/sparseopd/evaluation.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true", help="Regenerate responses even when outputs exist")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true", help="Merge completed inference shards")
    parser.add_argument("--list-experiments", action="store_true", help="Print configured experiment names")
    parser.add_argument("--list-needs-generation", action="store_true", help="Print experiments with missing responses")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or Path(config.get("output_dir", "eval"))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    experiments = args.experiments or list(config["experiments"])
    unknown = sorted(set(experiments) - set(config["experiments"]))
    if unknown:
        raise SystemExit(f"Unknown experiment(s): {', '.join(unknown)}")
    if args.list_experiments:
        for experiment in experiments:
            print(experiment)
        return
    if args.list_needs_generation:
        for experiment in experiments:
            if args.force or not experiment_generation_is_complete(experiment, config, output_dir):
                print(experiment)
        return
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error("Require num-shards >= 1 and 0 <= shard-id < num-shards")
    if args.merge_shards and (args.num_shards < 2 or args.shard_id != 0):
        parser.error("Shard merging requires --num-shards >= 2 and the default shard id")
    for experiment in experiments:
        if args.merge_shards:
            merge_generation_shards(experiment, config, output_dir, args.num_shards, force=args.force)
        else:
            generate_experiment(
                experiment,
                config,
                output_dir,
                force=args.force,
                num_shards=args.num_shards,
                shard_id=args.shard_id,
            )


if __name__ == "__main__":
    main()
