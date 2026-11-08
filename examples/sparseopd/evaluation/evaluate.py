#!/usr/bin/env python3
"""Score saved SparseOPD generations with Math-Verify and update the result table."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_LABELS = {"aime24": "AIME 2024", "aime25": "AIME 2025", "hmmt25": "HMMT-Feb-2025"}
PASS_AT_KS = (1, 2, 4, 8, 16, 32, 64, 128)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("experiments"), dict):
        raise ValueError(f"Invalid evaluation config: {path}")
    return config


@lru_cache(maxsize=1)
def scorer_metadata() -> dict:
    try:
        import math_verify  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('Install the evaluator: python -m pip install "math-verify[antlr4_9_3]==0.9.0"') from exc
    return {
        "name": "math-verify",
        "version": version("math-verify"),
        "antlr4_version": version("antlr4-python3-runtime"),
        "latex2sympy2_extended_version": version("latex2sympy2_extended"),
        "boxed": "last",
        "boxed_match_priority": 0,
        "extraction_mode": "first_match",
        "fallback_mode": "no_fallback",
        "timeout_seconds": 5,
    }


@lru_cache(maxsize=256)
def parse_gold(answer: str):
    from math_verify import LatexExtractionConfig, parse

    gold = parse(
        r"\boxed{" + answer + "}",
        extraction_config=(LatexExtractionConfig(),),
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        parsing_timeout=5,
        raise_on_error=True,
    )
    if not gold:
        raise ValueError(f"Math-Verify cannot parse the reference answer: {answer!r}")
    return gold


def grade_response(task: tuple[str, str]) -> dict:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    from math_verify.errors import TimeoutException

    output, answer = task
    latex = LatexExtractionConfig(boxed_match_priority=0)
    latex = replace(latex, normalization_config=replace(latex.normalization_config, boxed="last"))
    error = None
    gold, prediction = [], []
    correct = False
    try:
        gold = parse_gold(answer)
        prediction = parse(
            output,
            extraction_config=(latex, ExprExtractionConfig()),
            fallback_mode="no_fallback",
            extraction_mode="first_match",
            parsing_timeout=5,
            raise_on_error=True,
        )
        correct = bool(prediction and verify(gold, prediction, timeout_seconds=5, raise_on_error=True))
    except (Exception, TimeoutException) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "correct": correct,
        "extracted_prediction": [str(value) for value in prediction],
        "extracted_gold": [str(value) for value in gold],
        "error": error,
    }


def load_generations(path: Path, sample_count: int) -> list[dict]:
    records, identities = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            if not isinstance(record.get("answer"), str) or not record["answer"].strip():
                raise ValueError(f"Missing reference answer: {path}:{line_number}")
            outputs = record.get("outputs")
            if (
                not isinstance(outputs, list)
                or len(outputs) != sample_count
                or not all(isinstance(x, str) for x in outputs)
            ):
                raise ValueError(f"Expected {sample_count} responses per question: {path}:{line_number}")
            if record["problem_id"] in identities:
                raise ValueError(f"Duplicate problem_id: {path}:{line_number}")
            identities.add(record["problem_id"])
            records.append(record)
    if not records:
        raise ValueError(f"Empty generation file: {path}")
    return records


def generation_sources(output_dir: Path, experiment: str, dataset_keys) -> dict:
    sources = {}
    for key in dataset_keys:
        path = output_dir / "generations" / experiment / f"{key}.jsonl"
        stat = path.stat()
        sources[key] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return sources


def evaluation_is_complete(experiment: str, config: dict[str, Any], output_dir: Path) -> bool:
    result_path = output_dir / "results" / f"{experiment}.json"
    if not result_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("experiment") != experiment or result.get("scorer") != scorer_metadata():
            return False
        if result.get("sources") != generation_sources(output_dir, experiment, DATASET_LABELS):
            return False
        generation = result["generation"]
        configured_generation = config["generation"]
        if result.get("samples_for_pass_at_k") != int(configured_generation["samples_for_pass_at_k"]):
            return False
        if int(generation["samples_for_avg_at_8"]) != int(configured_generation["samples_for_avg_at_8"]):
            return False
        metrics = result["metrics"]
        if not all(key in metrics for key in ("aime24", "aime25", "hmmt25", "avg@8_mean")):
            return False
        if not all("avg@8" in metrics[key] for key in DATASET_LABELS):
            return False
        return all(set(map(str, PASS_AT_KS)).issubset(metrics[key].get("pass@k", {})) for key in ("aime24", "aime25"))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _binomial_probabilities(n: int, p: float) -> list[float]:
    if p <= 0.0:
        return [1.0] + [0.0] * n
    if p >= 1.0:
        return [0.0] * n + [1.0]
    log_p, log_q = math.log(p), math.log1p(-p)
    log_probabilities = [
        math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * log_p + (n - j) * log_q
        for j in range(n + 1)
    ]
    maximum = max(log_probabilities)
    probabilities = [math.exp(value - maximum) for value in log_probabilities]
    total = sum(probabilities)
    return [value / total for value in probabilities]


def pass_at_k(correct_count: int, sample_count: int, k: int) -> float:
    if not 0 <= correct_count <= sample_count or not 0 < k <= sample_count:
        raise ValueError("Require 0 <= correct_count <= sample_count and 0 < k <= sample_count")
    if sample_count - correct_count < k:
        return 1.0
    return 1.0 - math.comb(sample_count - correct_count, k) / math.comb(sample_count, k)


def pass_at_k_with_ci(correct_counts: list[int], sample_count: int, k: int) -> dict[str, float]:
    if not correct_counts:
        raise ValueError("At least one prompt is required")
    estimates = [pass_at_k(count, sample_count, k) for count in correct_counts]
    estimate = sum(estimates) / len(estimates)
    variance_sum = 0.0
    for count in correct_counts:
        probabilities = _binomial_probabilities(sample_count, count / sample_count)
        values = [pass_at_k(j, sample_count, k) for j in range(sample_count + 1)]
        mean = sum(p * value for p, value in zip(probabilities, values, strict=True))
        second = sum(p * value * value for p, value in zip(probabilities, values, strict=True))
        variance_sum += max(0.0, second - mean * mean)
    margin = 1.96 * math.sqrt(variance_sum) / len(correct_counts)
    return {"estimate": estimate, "ci95_low": max(0.0, estimate - margin), "ci95_high": min(1.0, estimate + margin)}


def evaluate_counts(
    correct_counts: list[int],
    sample_count: int,
    include_pass_at_k: bool,
    avg_at_8_correct_counts: list[int] | None = None,
) -> dict[str, Any]:
    avg8_counts = avg_at_8_correct_counts if avg_at_8_correct_counts is not None else correct_counts
    if (
        not avg8_counts
        or len(avg8_counts) != len(correct_counts)
        or any(count < 0 or count > 8 for count in avg8_counts)
    ):
        raise ValueError("avg@8 requires one correctness count in [0, 8] per prompt")
    metrics: dict[str, Any] = {"avg@8": sum(avg8_counts) / len(avg8_counts) / 8}
    if include_pass_at_k:
        metrics["pass@k"] = {str(k): pass_at_k_with_ci(correct_counts, sample_count, k) for k in PASS_AT_KS}
    return metrics


def table_rows(result_files: list[Path]) -> list[str]:
    rows = []
    for result_file in sorted(result_files):
        result = json.loads(result_file.read_text(encoding="utf-8"))
        experiment = result["experiment"]
        for dataset_key in DATASET_LABELS:
            metrics = result["metrics"][dataset_key]
            cells = [experiment, DATASET_LABELS[dataset_key]]
            for k in PASS_AT_KS:
                metric = metrics.get("pass@k", {}).get(str(k))
                cells.append(
                    ""
                    if metric is None
                    else (
                        f"{metric['estimate'] * 100:.2f} "
                        f"[{metric['ci95_low'] * 100:.2f}, {metric['ci95_high'] * 100:.2f}]"
                    )
                )
            cells.append(f"{metrics['avg@8'] * 100:.2f}")
            rows.append("| " + " | ".join(cells) + " |")
        mean_cells = [experiment, "Mean (3 benchmarks)"] + [""] * len(PASS_AT_KS)
        mean_cells.append(f"{result['metrics']['avg@8_mean'] * 100:.2f}")
        rows.append("| " + " | ".join(mean_cells) + " |")
    return rows


def render_results(output_dir: Path) -> Path:
    results_dir = output_dir / "results"
    result_files = (
        [
            path
            for path in results_dir.glob("*.json")
            if json.loads(path.read_text(encoding="utf-8")).get("scorer") == scorer_metadata()
        ]
        if results_dir.is_dir()
        else []
    )
    headers = ["Experiment", "Benchmark", *(f"pass@{k} % (95% CI)" for k in PASS_AT_KS), "avg@8 %"]
    lines = [
        "# SparseOPD Evaluation Results",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(table_rows(result_files))
    if not result_files:
        lines.append("| _No completed evaluations_ | " + " | ".join([""] * (len(headers) - 1)) + " |")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "results.md"
    with (output_dir / ".results.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fd, temporary = tempfile.mkstemp(prefix="results.", suffix=".md", dir=output_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write("\n".join(lines) + "\n")
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return destination


def score_experiment(experiment: str, config: dict[str, Any], output_dir: Path, workers: int | None = None) -> Path:
    scorer = scorer_metadata()
    workers = int(workers if workers is not None else config.get("scoring", {}).get("workers", 16))
    if workers < 1:
        raise ValueError("Scoring workers must be >= 1")
    generation = config["generation"]
    sources = generation_sources(output_dir, experiment, DATASET_LABELS)
    audit_dir = output_dir / "audit" / experiment
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "issues.jsonl"
    temporary_audit = audit_path.with_suffix(".jsonl.tmp")
    metrics, diagnostics = {}, {}
    pool = (
        ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        if workers > 1
        else nullcontext(None)
    )
    with pool as executor, temporary_audit.open("w", encoding="utf-8") as audit:
        for dataset_key in DATASET_LABELS:
            count_key = "samples_for_avg_at_8" if dataset_key == "hmmt25" else "samples_for_pass_at_k"
            sample_count = int(generation[count_key])
            if sample_count < (8 if dataset_key == "hmmt25" else max(PASS_AT_KS)):
                raise ValueError("Require at least 128 AIME samples and 8 HMMT samples per question")
            raw_path = output_dir / "generations" / experiment / f"{dataset_key}.jsonl"
            records = load_generations(raw_path, sample_count)
            correct_counts, avg8_counts = [], []
            stats = {"questions": len(records), "responses": 0, "unparsed": 0, "errors": 0}
            for row_index, row in enumerate(records):
                tasks = [(output, row["answer"]) for output in row["outputs"]]
                grades = (
                    list(executor.map(grade_response, tasks, chunksize=8))
                    if executor
                    else list(map(grade_response, tasks))
                )
                correct_counts.append(sum(grade["correct"] for grade in grades))
                avg8_counts.append(sum(grade["correct"] for grade in grades[:8]))
                for sample_index, (output, grade) in enumerate(zip(row["outputs"], grades, strict=True)):
                    stats["responses"] += 1
                    stats["unparsed"] += not bool(grade["extracted_prediction"])
                    stats["errors"] += bool(grade["error"])
                    if not grade["extracted_prediction"] or grade["error"]:
                        audit.write(
                            json.dumps(
                                {
                                    "dataset": dataset_key,
                                    "problem_id": row["problem_id"],
                                    "sample_index": sample_index,
                                    "answer": row["answer"],
                                    "output": output,
                                    **grade,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                print(f"{experiment}/{dataset_key}: scored {row_index + 1}/{len(records)} questions", flush=True)
            metrics[dataset_key] = evaluate_counts(correct_counts, sample_count, dataset_key != "hmmt25", avg8_counts)
            diagnostics[dataset_key] = stats
            print(f"{experiment}/{dataset_key}: {stats}", flush=True)
    os.replace(temporary_audit, audit_path)
    metrics["avg@8_mean"] = sum(metrics[key]["avg@8"] for key in DATASET_LABELS) / 3
    result_path = output_dir / "results" / f"{experiment}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment,
        "generation": generation,
        "samples_for_pass_at_k": int(generation["samples_for_pass_at_k"]),
        "scorer": scorer,
        "sources": sources,
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, result_path)
    print(f"Scored {experiment} with Math-Verify: {result_path}")
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="*", help="Experiment names; defaults to every configured experiment")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/sparseopd/evaluation.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, help="CPU scoring workers; defaults to scoring.workers in YAML")
    parser.add_argument("--force", action="store_true", help="Rescore even when matching results already exist")
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("workers must be >= 1")
    config = load_config(args.config)
    output_dir = args.output_dir or Path(config.get("output_dir", "eval"))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    experiments = args.experiments or list(config["experiments"])
    unknown = sorted(set(experiments) - set(config["experiments"]))
    if unknown:
        parser.error(f"Unknown experiment(s): {', '.join(unknown)}")
    for experiment in experiments:
        if not args.force and evaluation_is_complete(experiment, config, output_dir):
            print(f"Evaluation already complete for {experiment}; skipping.")
            continue
        missing = [
            key for key in DATASET_LABELS if not (output_dir / "generations" / experiment / f"{key}.jsonl").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing generations for {experiment}: {', '.join(missing)}. Run evaluation/infer.sh first."
            )
        score_experiment(experiment, config, output_dir, workers=args.workers)
    print(f"Wrote summary table: {render_results(output_dir)}")


if __name__ == "__main__":
    main()
