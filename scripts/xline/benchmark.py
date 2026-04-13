#!/usr/bin/env python3
"""XLINE benchmark runner.

Loads five pretrained transformer models, benchmarks a baseline full fine-tuning
step against the XLINE controller strategy, and exports metrics + figures.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

REAL_TEXT_SAMPLES = [
    "Tokyo is one of the most densely populated cities on Earth, blending history and robotics.",
    "SpaceX successfully returned booster stages, changing launch economics for private spaceflight.",
    "Photosynthesis converts sunlight into chemical energy through chlorophyll-rich structures.",
    "Open-source software transformed collaborative engineering by making global contribution practical.",
    "Medical imaging systems rely on deep learning for faster diagnosis and enhanced pattern discovery.",
    "Climate models integrate atmospheric, oceanic, and land data to estimate long-term trends.",
    "A violin concerto combines mathematical structure and emotional storytelling through orchestration.",
    "Quantum error correction is required to stabilize noisy qubits for reliable computation.",
]

DEFAULT_MODELS = [
    "sshleifer/tiny-distilbert-base-cased",
    "prajjwal1/bert-tiny",
    "distilbert-base-uncased",
    "google/electra-small-discriminator",
    "bert-base-uncased",
]


@dataclass
class MetricRow:
    model: str
    scenario: str
    params_trainable: int
    params_total: int
    trainable_pct: float
    avg_step_ms: float
    peak_memory_mb: float


@dataclass
class FailureRow:
    model: str
    error: str


class XLineController(nn.Module):
    """Small external trainable module injected through forward hooks."""

    def __init__(self, hidden_size: int, bottleneck: int = 16, scale: float = 0.05) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.up(torch.tanh(self.down(x)))


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def get_encoder_layers(model: nn.Module) -> List[nn.Module]:
    # Works for most BERT-like encoders from AutoModel.
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return list(model.encoder.layer)
    if hasattr(model, "distilbert") and hasattr(model.distilbert, "transformer"):
        return list(model.distilbert.transformer.layer)
    if hasattr(model, "electra") and hasattr(model.electra, "encoder"):
        return list(model.electra.encoder.layer)
    raise RuntimeError("Unable to find encoder blocks for hook injection.")


def prepare_batch(tokenizer, device: torch.device, seq_len: int, batch_size: int) -> Dict[str, torch.Tensor]:
    samples = REAL_TEXT_SAMPLES[:batch_size]
    batch = tokenizer(
        samples,
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in batch.items()}


def maybe_cuda_peak_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device=device) / (1024**2)
    return 0.0


def benchmark_steps(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    steps: int,
    lr: float,
) -> tuple[float, float]:
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    times = []
    peak = 0.0

    model.train()
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        start = time.perf_counter()
        output = model(**batch)
        loss = output.last_hidden_state.pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

        if device.type == "cuda":
            peak = max(peak, maybe_cuda_peak_mb(device))

    return statistics.mean(times), peak


def install_xline_hooks(model: nn.Module, bottleneck: int) -> tuple[nn.ModuleList, list]:
    layers = get_encoder_layers(model)
    target_ids = sorted({0, max(0, len(layers) // 2), len(layers) - 1})
    hidden_size = model.config.hidden_size
    controllers = nn.ModuleList([XLineController(hidden_size, bottleneck=bottleneck) for _ in target_ids])
    handles = []

    for idx, ctrl in zip(target_ids, controllers):
        layer = layers[idx]

        def hook_fn(_module, _inputs, output, c=ctrl):
            if isinstance(output, tuple):
                modified = c(output[0])
                return (modified,) + output[1:]
            return c(output)

        handles.append(layer.register_forward_hook(hook_fn))

    return controllers, handles


def build_xline_model(model: nn.Module, bottleneck: int) -> nn.Module:
    for p in model.parameters():
        p.requires_grad = False
    controllers, handles = install_xline_hooks(model, bottleneck=bottleneck)
    model.add_module("xline_controllers", controllers)
    model._xline_handles = handles  # type: ignore[attr-defined]
    for p in model.xline_controllers.parameters():
        p.requires_grad = True
    return model


def teardown_xline(model: nn.Module) -> None:
    for handle in getattr(model, "_xline_handles", []):
        handle.remove()


def percentage_delta(before: float, after: float) -> float:
    if math.isclose(before, 0.0):
        return 0.0
    return ((before - after) / before) * 100.0


def render_figures(rows: List[MetricRow], output_dir: Path) -> None:
    if not rows:
        return
    models = sorted({r.model for r in rows})
    baseline_time = {r.model: r.avg_step_ms for r in rows if r.scenario == "baseline"}
    xline_time = {r.model: r.avg_step_ms for r in rows if r.scenario == "xline"}
    baseline_trainable = {r.model: r.params_trainable for r in rows if r.scenario == "baseline"}
    xline_trainable = {r.model: r.params_trainable for r in rows if r.scenario == "xline"}

    x = range(len(models))
    width = 0.36

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], [baseline_time[m] for m in models], width=width, label="Baseline")
    plt.bar([i + width / 2 for i in x], [xline_time[m] for m in models], width=width, label="XLINE")
    plt.xticks(list(x), models, rotation=20, ha="right")
    plt.ylabel("Avg training step (ms)")
    plt.title("Training Step Time: Baseline vs XLINE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "time_comparison.png", dpi=220)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], [baseline_trainable[m] for m in models], width=width, label="Baseline")
    plt.bar([i + width / 2 for i in x], [xline_trainable[m] for m in models], width=width, label="XLINE")
    plt.xticks(list(x), models, rotation=20, ha="right")
    plt.ylabel("Trainable params")
    plt.yscale("log")
    plt.title("Trainable Parameter Reduction with XLINE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "trainable_params.png", dpi=220)
    plt.close()


def write_outputs(rows: List[MetricRow], failures: List[FailureRow], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    screenshots = output_root / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "xline_benchmark_results.json"
    csv_path = output_root / "xline_benchmark_results.csv"
    failures_path = output_root / "xline_failures.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
        else:
            f.write("model,scenario,params_trainable,params_total,trainable_pct,avg_step_ms,peak_memory_mb\n")

    with failures_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in failures], f, indent=2)

    render_figures(rows, screenshots)


def print_report(rows: List[MetricRow]) -> None:
    print("\n=== XLINE BENCHMARK REPORT ===")
    models = sorted({r.model for r in rows})
    for model in models:
        b = next(r for r in rows if r.model == model and r.scenario == "baseline")
        x = next(r for r in rows if r.model == model and r.scenario == "xline")
        time_cut = percentage_delta(b.avg_step_ms, x.avg_step_ms)
        trainable_cut = percentage_delta(float(b.params_trainable), float(x.params_trainable))
        print(f"\nModel: {model}")
        print(f"  Baseline step: {b.avg_step_ms:.2f} ms | XLINE step: {x.avg_step_ms:.2f} ms")
        print(f"  Time reduction: {time_cut:.2f}%")
        print(f"  Trainable params reduction: {trainable_cut:.2f}%")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run XLINE protocol benchmark")
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="HF model ids")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bottleneck", type=int, default=16)
    p.add_argument("--output-dir", default="artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rows: List[MetricRow] = []
    failures: List[FailureRow] = []

    for model_id in args.models:
        print(f"\nLoading model: {model_id}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            baseline_model = AutoModel.from_pretrained(model_id).to(device)
            batch = prepare_batch(tokenizer, device, args.seq_len, args.batch_size)

            for p in baseline_model.parameters():
                p.requires_grad = True
            b_trainable, b_total = count_parameters(baseline_model)
            b_ms, b_peak = benchmark_steps(baseline_model, batch, device, args.steps, args.lr)
            rows.append(
                MetricRow(
                    model=model_id,
                    scenario="baseline",
                    params_trainable=b_trainable,
                    params_total=b_total,
                    trainable_pct=(b_trainable / b_total) * 100,
                    avg_step_ms=b_ms,
                    peak_memory_mb=b_peak,
                )
            )
            del baseline_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            xline_model = AutoModel.from_pretrained(model_id).to(device)
            xline_model = build_xline_model(xline_model, bottleneck=args.bottleneck).to(device)
            x_trainable, x_total = count_parameters(xline_model)
            x_ms, x_peak = benchmark_steps(xline_model, batch, device, args.steps, args.lr)
            teardown_xline(xline_model)
            rows.append(
                MetricRow(
                    model=model_id,
                    scenario="xline",
                    params_trainable=x_trainable,
                    params_total=x_total,
                    trainable_pct=(x_trainable / x_total) * 100,
                    avg_step_ms=x_ms,
                    peak_memory_mb=x_peak,
                )
            )
            del xline_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] Skipping model due to error: {exc}")
            failures.append(FailureRow(model=model_id, error=str(exc)))

    write_outputs(rows, failures, Path(args.output_dir))
    if rows:
        print_report(rows)
    else:
        print("\nNo models completed successfully. Check artifacts/xline_failures.json for details.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    main()
