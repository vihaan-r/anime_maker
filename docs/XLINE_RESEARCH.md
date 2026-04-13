# XLINE Protocol Research Note

## Title

**XLINE: A Universal External-Controller Strategy for Efficient Adaptation of Frozen Pretrained Models**

## Author

**Vihaan** (Independent builder, age 14)

## Motivation

I wanted to build a high-performance PC for AI work, but memory and RAM costs were too high. Instead of scaling hardware cost, I focused on reducing the amount of computation needed to adapt large models. XLINE came from that pressure: keep the full model frozen and train only a tiny adaptive layer.

## Abstract

This project evaluates XLINE, a protocol that injects lightweight trainable controllers into intermediate activations of frozen pretrained models. We compare full-parameter adaptation (baseline) against XLINE on five transformer models. The benchmark records training-step time, trainable parameter count, and memory usage. Results consistently show major reductions in trainable parameters and competitive compute cost for adaptation workloads.

## Method

1. Load pretrained model (`AutoModel`) and tokenizer.
2. Baseline run: unfreeze all model parameters and benchmark short training steps.
3. XLINE run:
   - freeze all base parameters,
   - inject controller modules at early/mid/late encoder blocks via forward hooks,
   - train only XLINE controller parameters.
4. Collect metrics and compute percentage deltas.
5. Save CSV/JSON and chart images for reproducible proof.
6. If a model cannot be downloaded in the current environment, the error is recorded in `artifacts/xline_failures.json` without crashing the whole run.

## Model Set (5 sizes)

- `sshleifer/tiny-distilbert-base-cased`
- `prajjwal1/bert-tiny`
- `distilbert-base-uncased`
- `google/electra-small-discriminator`
- `bert-base-uncased`

## Reproducibility

```bash
pip install -r scripts/xline/requirements.txt
python scripts/xline/benchmark.py --output-dir artifacts
```

## Artifact Evidence

### Training Time Comparison

![Training time comparison](../artifacts/screenshots/time_comparison.png)

### Trainable Parameter Comparison

![Trainable parameters comparison](../artifacts/screenshots/trainable_params.png)

## Interpretation

XLINE does not retrain the entire model. It introduces a compact steering module that changes hidden activations while preserving the original pretrained knowledge. This gives a practical path for efficient adaptation when hardware resources are limited.

## Conclusion

XLINE treats large pretrained models as fixed intelligence cores and learns behavior with small, external controllers. This enables lower training overhead and much smaller adaptation checkpoints.
