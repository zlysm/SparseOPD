#!/usr/bin/env python3
"""Audit and remove repeated prompts from DAPO-Math-17k."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor
import hashlib
import json
import os
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("path/to/dapo-math-17k.parquet")
DEFAULT_OUTPUT = REPO_ROOT / "dataset" / "dapo-math-17k-dedup.parquet"


def canonical_row_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def semantic_row(row: dict[str, Any]) -> dict[str, Any]:
    """Build a diagnostic key without the per-row UUID; never use it for deletion."""
    result = dict(row)
    extra_info = result.get("extra_info")
    if isinstance(extra_info, dict) and "index" in extra_info:
        result["extra_info"] = {key: value for key, value in extra_info.items() if key != "index"}
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    prompt_hashes = []
    semantic_hashes = []
    full_hashes = []
    for row in rows:
        if "prompt" not in row or row["prompt"] is None:
            raise ValueError("Every DAPO-Math row must contain a non-null prompt.")
        prompt_hashes.append(canonical_row_hash({"prompt": row["prompt"]}))
        semantic_hashes.append(canonical_row_hash(semantic_row(row)))
        full_hashes.append(canonical_row_hash(row))
    return prompt_hashes, semantic_hashes, full_hashes


def classify_order(row_hashes: list[str], unique_count: int, repeat_count: int) -> str:
    if all(row_hashes[index] == row_hashes[index % unique_count] for index in range(len(row_hashes))):
        return "whole_dataset_block_repeated"
    if all(row_hashes[index] == row_hashes[index // repeat_count] for index in range(len(row_hashes))):
        return "each_row_repeated_consecutively"
    return "repeated_rows_interleaved_or_shuffled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the documented block repetition, audit prompt/metadata collisions, "
            "then keep the first source row for each exact prompt."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source parquet path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Deduplicated parquet path")
    parser.add_argument("--expected-repeats", type=int, default=100)
    parser.add_argument(
        "--expected-prompts",
        type=int,
        default=17_398,
        help="Expected number of distinct prompt contents (0 disables the count check)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 2) - 2)),
        help="Hashing worker processes (default: up to 16, leaving two CPUs free)",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output and manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_repeats < 2:
        raise SystemExit("--expected-repeats must be at least 2")
    if args.expected_prompts < 0:
        raise SystemExit("--expected-prompts must be non-negative")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if not source.is_file():
        raise SystemExit(f"Source parquet not found: {source}")
    if source == output:
        raise SystemExit("Source and output paths must be different")
    if not args.force and (output.exists() or manifest_path.exists()):
        raise SystemExit(f"Output already exists; inspect it or pass --force: {output}")

    started_at = time.monotonic()
    parquet_file = pq.ParquetFile(source)
    total_rows = parquet_file.metadata.num_rows
    print(f"[1/4] Scanning {total_rows:,} source rows from {source}", flush=True)
    row_hashes: list[str] = []
    prompt_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    full_counts: Counter[str] = Counter()
    first_rows: dict[str, dict[str, Any]] = {}
    first_rows_by_prompt: dict[str, dict[str, Any]] = {}
    full_counts_by_semantic: dict[str, Counter[str]] = {}
    full_identities_by_prompt: dict[str, set[str]] = {}
    reward_models_by_prompt: dict[str, set[str]] = {}
    source_rows = 0
    next_progress = 0
    batch_iter = iter(parquet_file.iter_batches(batch_size=65_536, use_threads=True))
    pending: deque[tuple[list[dict[str, Any]], Future[tuple[list[str], list[str], list[str]]]]] = deque()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        while True:
            while len(pending) < args.workers:
                try:
                    rows = next(batch_iter).to_pylist()
                except StopIteration:
                    break
                pending.append((rows, executor.submit(hash_rows, rows)))
            if not pending:
                break

            rows, future = pending.popleft()
            prompt_hashes, semantic_hashes, full_hashes = future.result()
            for row, prompt_hash, semantic_hash, full_hash in zip(
                rows, prompt_hashes, semantic_hashes, full_hashes, strict=True
            ):
                row_hashes.append(full_hash)
                prompt_counts[prompt_hash] += 1
                semantic_counts[semantic_hash] += 1
                full_counts[full_hash] += 1
                first_rows.setdefault(full_hash, row)
                first_rows_by_prompt.setdefault(prompt_hash, row)
                full_counts_by_semantic.setdefault(semantic_hash, Counter())[full_hash] += 1
                full_identities_by_prompt.setdefault(prompt_hash, set()).add(full_hash)
                reward_models_by_prompt.setdefault(prompt_hash, set()).add(
                    canonical_row_hash(row.get("reward_model"))
                )
            source_rows += len(rows)

            if source_rows >= next_progress:
                elapsed = time.monotonic() - started_at
                rate = source_rows / elapsed if elapsed else 0.0
                print(
                    f"  scanned {source_rows:,}/{total_rows:,} rows "
                    f"({source_rows / total_rows:.1%}), {rate:,.0f} rows/s "
                    f"[{args.workers} workers]",
                    flush=True,
                )
                next_progress = source_rows + 65_536

    print(
        f"[2/4] Validating {len(full_counts):,} unique full rows, "
        f"{len(semantic_counts):,} semantic groups, and {len(prompt_counts):,} unique prompts",
        flush=True,
    )
    semantic_collisions = sum(
        len(identities) > 1 for identities in full_counts_by_semantic.values()
    )
    if semantic_collisions:
        print(
            f"  detected {semantic_collisions:,} semantic groups with multiple complete rows; "
            "these rows will be preserved separately",
            flush=True,
        )
    unique_full_hashes = list(first_rows)
    unique_full_count = len(unique_full_hashes)
    unique_prompt_hashes = list(first_rows_by_prompt)
    unique_prompt_count = len(unique_prompt_hashes)
    expected_unique_full_count = source_rows // args.expected_repeats

    if source_rows % args.expected_repeats:
        raise SystemExit(
            f"Refusing to deduplicate: {source_rows} rows is not divisible by "
            f"{args.expected_repeats}."
        )
    bad_counts = [(row_hash, count) for row_hash, count in full_counts.items() if count != args.expected_repeats]
    if bad_counts:
        semantic_outliers = [
            (semantic_hash, count)
            for semantic_hash, count in semantic_counts.items()
            if count != args.expected_repeats
        ]
        examples = []
        for semantic_hash, count in semantic_outliers[:5]:
            identity_counts = sorted(full_counts_by_semantic[semantic_hash].values(), reverse=True)
            examples.append(
                f"{semantic_hash[:12]}... semantic x{count}, "
                f"{len(identity_counts)} full-row identities with counts {identity_counts[:5]}"
            )
        raise SystemExit(
            f"Refusing to deduplicate: {len(bad_counts)} complete source rows do not each occur "
            f"exactly {args.expected_repeats} times. Semantic-only outliers: "
            f"{len(semantic_outliers)}; examples: {'; '.join(examples)}. "
            "A semantic group occurring 200 times is safe to preserve as two source rows only "
            "when it consists of distinct complete rows occurring 100 times each."
        )
    if unique_full_count != expected_unique_full_count:
        raise SystemExit(
            f"Refusing to deduplicate: found {unique_full_count} unique full rows; expected "
            f"{expected_unique_full_count} from the configured repeat factor."
        )
    if args.expected_prompts and unique_prompt_count != args.expected_prompts:
        raise SystemExit(
            f"Refusing to write output: found {unique_prompt_count:,} unique prompt contents; "
            f"expected {args.expected_prompts:,}. Inspect the source revision and prompt collisions, "
            "or pass --expected-prompts 0 to disable this check."
        )

    multirow_prompts = [key for key, identities in full_identities_by_prompt.items() if len(identities) > 1]
    multireward_prompts = [key for key, identities in reward_models_by_prompt.items() if len(identities) > 1]
    print(
        f"  {len(multirow_prompts):,} prompts have multiple complete row identities; "
        f"{len(multireward_prompts):,} have conflicting reward_model values",
        flush=True,
    )

    order_pattern = classify_order(row_hashes, unique_full_count, args.expected_repeats)
    print(
        f"  detected ordering: {order_pattern}; keeping the first source row for each of "
        f"{unique_prompt_count:,} distinct prompts",
        flush=True,
    )
    deduplicated = pa.Table.from_pylist(
        [first_rows_by_prompt[prompt_hash] for prompt_hash in unique_prompt_hashes],
        schema=parquet_file.schema_arrow,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + f".tmp.{os.getpid()}")
    try:
        print(f"[3/4] Writing deduplicated parquet to {output}", flush=True)
        pq.write_table(deduplicated, temporary_output, compression="zstd")
        # Re-read the artifact and validate row count and semantic uniqueness.
        print("[4/4] Re-reading output and validating uniqueness/checksums", flush=True)
        written = pq.read_table(temporary_output)
        written_rows = written.to_pylist()
        written_prompt_hashes = [canonical_row_hash({"prompt": row["prompt"]}) for row in written_rows]
        if len(written_prompt_hashes) != unique_prompt_count or len(set(written_prompt_hashes)) != unique_prompt_count:
            raise RuntimeError("Written parquet failed prompt-count or prompt-uniqueness validation")
        os.replace(temporary_output, output)
    finally:
        temporary_output.unlink(missing_ok=True)

    manifest = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_rows": source_rows,
        "unique_full_rows": unique_full_count,
        "unique_semantic_rows": len(semantic_counts),
        "unique_prompt_rows": unique_prompt_count,
        "semantic_groups_with_multiple_full_row_identities": semantic_collisions,
        "prompts_with_multiple_full_row_identities": len(multirow_prompts),
        "prompts_with_multiple_reward_model_values": len(multireward_prompts),
        "prompt_collision_audit": [
            {
                "prompt_sha256": prompt_hash,
                "kept_full_row_sha256": canonical_row_hash(first_rows_by_prompt[prompt_hash]),
                "row_variants": [
                    {
                        "full_row_sha256": full_hash,
                        "source_multiplicity": full_counts[full_hash],
                        "reward_model_sha256": canonical_row_hash(first_rows[full_hash].get("reward_model")),
                    }
                    for full_hash in sorted(full_identities_by_prompt[prompt_hash])
                ],
                "conflicting_reward_models": len(reward_models_by_prompt[prompt_hash]) > 1,
            }
            for prompt_hash in sorted(multirow_prompts)
        ],
        "expected_repeats_per_full_row": args.expected_repeats,
        "observed_order_pattern": order_pattern,
        "deduplication_rule": "keep first source occurrence per exact prompt content",
        "columns": parquet_file.schema_arrow.names,
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Completed in {time.monotonic() - started_at:.1f}s", flush=True)


if __name__ == "__main__":
    main()
