# XLINE Protocol — Universal Model Execution Framework

Built by **Vihaan (14 years old)**.

I created XLINE because I wanted to build a stronger PC, but RAM prices were too high. Instead of waiting for better hardware, I designed a method that makes existing AI models cheaper to adapt: **freeze the full model and only train a tiny external controller**.

## What this repo now contains

- A full XLINE benchmark pipeline with real pretrained models (5 different sizes).
- Automatic baseline vs XLINE computation comparison.
- Percent reduction reporting in terminal.
- Generated visual proof (PNG charts) saved as `artifacts/screenshots/`.
- A GitHub Action to run the entire benchmark and upload evidence artifacts.
- A research-style write-up in `docs/XLINE_RESEARCH.md`.

## XLINE in one line

**Base model = frozen brain** + **XLINE controller = lightweight steering layer**.

## Run locally

```bash
python -m pip install --upgrade pip
pip install -r scripts/xline/requirements.txt
python scripts/xline/benchmark.py --output-dir artifacts
```

## Expected outputs

After running, you get:

- `artifacts/xline_benchmark_results.csv`
- `artifacts/xline_benchmark_results.json`
- `artifacts/screenshots/time_comparison.png`
- `artifacts/screenshots/trainable_params.png`

The terminal prints per-model:

- baseline step time
- XLINE step time
- time reduction percentage
- trainable parameter reduction percentage

## GitHub Actions

Workflow file:

- `.github/workflows/xline-benchmark.yml`

It installs dependencies, runs the benchmark for 5 models, and uploads all proof artifacts.
