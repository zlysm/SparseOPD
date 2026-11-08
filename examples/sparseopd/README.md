# Sparse OPD commands

This directory contains the runnable entry points for the Family 5 reproduction with verl 0.8.0. The default setting is Qwen3-1.7B ← Qwen3-4B-Instruct-2507 on DAPO-Math-17K.

## Configure local paths

Replace the `path/to/...` placeholders in `config/sparseopd/qwen3_1.7b_common.yaml` and `config/sparseopd/evaluation.yaml` with your local model paths. The launcher also contains a `path/to/conda/env` placeholder for the training environment; update it if you want the script to activate Conda automatically.

The default configuration uses 8 NVIDIA A100 GPUs on one node: 4 GPUs for the student and 4 GPUs for the teacher. Adjust the YAML for another layout.

## Prepare DAPO-Math-17K

The paper uses 17,398 unique prompts from [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k). Download the source parquet and pass its path to the deduplication script:

```bash
python examples/sparseopd/prepare_dapo_math_17k.py \
  --input path/to/dapo-math-17k.parquet
```

The output is written to `dataset/dapo-math-17k-dedup.parquet`. Use `--output` to place it elsewhere.

## Train

The default experiment suite runs all configured methods sequentially. Submit it with:

```bash
bash examples/sparseopd/run_experiments.sh
```

To run only selected methods from the suite, pass their names:

```bash
bash examples/sparseopd/run_experiments.sh rand1tok minmaxtok dense_opd
```

To run a single experiment directly, use the single-experiment launcher:

```bash
bash examples/sparseopd/train_sparseopd.sh rand1tok
```

Available methods are `dense_opd`, `rand1tok`, `maxtok`, `mintok`, `minmaxtok`, `randmask_0p1pct`, `pctltail_0p05pct`, and the threshold ablations listed in `run_experiments.sh`.

Use another YAML with the existing `SPARSEOPD_CONFIG_PATH` override:

```bash
SPARSEOPD_CONFIG_PATH=path/to/config.yaml \
  bash examples/sparseopd/train_sparseopd.sh rand1tok
```

Checkpoints, resolved configs, logs, and TensorBoard files are written below the configured output directory.

## Evaluate

Download the evaluation datasets and place their parquet files at the paths expected by `examples/sparseopd/evaluation/prepare_datasets.py`:

| Benchmark | Hugging Face dataset |
| --- | --- |
| AIME 2024 | [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) |
| AIME 2025 | [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025) |
| HMMT February 2025 | [MathArena/hmmt_feb_2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025) |

Then run:

```bash
python3 examples/sparseopd/evaluation/prepare_datasets.py
EVAL_PYTHON=path/to/conda/env/bin/python bash examples/sparseopd/evaluation/infer.sh
EVAL_PYTHON=path/to/conda/env/bin/python bash examples/sparseopd/evaluation/score.sh
```

`EVAL_PYTHON`, `EVAL_CONFIG`, and `EVAL_OUTPUT_DIR` are existing script overrides. Pass experiment names to `infer.sh` or `score.sh` to process selected checkpoints; add `--force` to regenerate or rescore completed outputs.

The evaluation protocol follows the paper: 256 samples per prompt for pass@k, k in {1, 2, 4, 8, 16, 32, 64, 128}, 95% confidence intervals, and avg@8 on all three benchmarks.
