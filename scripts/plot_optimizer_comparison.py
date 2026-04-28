from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RUN_DIRS = [
    PROJECT_ROOT / "outputs" / "deeplabv3_resnet50_voc_sgd_50ep",
    PROJECT_ROOT / "outputs" / "deeplabv3_resnet50_voc_adam_50ep",
    PROJECT_ROOT / "outputs" / "deeplabv3_resnet50_voc_adamw_50ep",
    PROJECT_ROOT / "outputs" / "deeplabv3_resnet50_voc_lion_50ep",
]

TRAIN_METRICS = [
    "mean_loss",
    "miou",
    "pixel_accuracy",
    "mean_step_time_sec",
    "mean_forward_time_sec",
    "mean_backward_time_sec",
    "mean_optimizer_time_sec",
    "examples_per_sec",
    "optimizer_state_bytes",
    "parameter_bytes",
]

VAL_METRICS = [
    "mean_loss",
    "miou",
    "pixel_accuracy",
    "mean_step_time_sec",
    "examples_per_sec",
]

METRIC_LABELS = {
    "mean_loss": "Loss",
    "miou": "mIoU",
    "pixel_accuracy": "Pixel Accuracy",
    "mean_step_time_sec": "Mean Step Time (s)",
    "mean_forward_time_sec": "Mean Forward Time (s)",
    "mean_backward_time_sec": "Mean Backward Time (s)",
    "mean_optimizer_time_sec": "Mean Optimizer Time (s)",
    "examples_per_sec": "Examples / s",
    "optimizer_state_bytes": "Optimizer State (bytes)",
    "parameter_bytes": "Parameter Memory (bytes)",
}

COLORS = {
    "sgd": "#1b9e77",
    "adam": "#d95f02",
    "adamw": "#7570b3",
    "lion": "#e7298a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot optimizer comparison graphs from history.jsonl files.")
    parser.add_argument(
        "--run-dirs",
        nargs="*",
        default=[str(path) for path in DEFAULT_RUN_DIRS],
        help="Run directories containing history.jsonl and config.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "comparison_plots_25ep"),
        help="Directory to write comparison plots into.",
    )
    parser.add_argument(
        "--primary-metric",
        default="miou",
        help="Primary validation metric for the shared four-optimizer comparison plot.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=25,
        help="Cap each run to this many epochs before plotting and summarizing.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_history(path: Path, max_epoch: int | None = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            epoch = int(record.get("val", {}).get("epoch", len(records) + 1))
            if max_epoch is not None and epoch > max_epoch:
                continue
            records.append(record)
    return records


def infer_optimizer_label(run_dir: Path) -> str:
    config = load_yaml(run_dir / "config.yaml")
    return str(config.get("optimizer", {}).get("name", run_dir.name)).lower()


def extract_series(records: List[Dict[str, Any]], split: str, metric: str) -> Tuple[List[int], List[float]]:
    epochs: List[int] = []
    values: List[float] = []
    for record in records:
        section = record.get(split, {})
        if metric not in section:
            continue
        value = section.get(metric)
        if value is None:
            continue
        epochs.append(int(section.get("epoch", len(epochs) + 1)))
        values.append(float(value))
    return epochs, values


def plot_line_metric(run_histories: Dict[str, List[Dict[str, Any]]], split: str, metric: str, output_dir: Path, max_epoch: int) -> None:
    plt.figure(figsize=(9, 5.5))
    for optimizer_name, records in run_histories.items():
        epochs, values = extract_series(records, split, metric)
        if not epochs:
            continue
        plt.plot(epochs, values, marker="o", linewidth=2, markersize=4, label=optimizer_name.upper(), color=COLORS.get(optimizer_name))

    plt.xlabel("Epoch")
    plt.ylabel(METRIC_LABELS.get(metric, metric))
    plt.title(f"{split.upper()} {METRIC_LABELS.get(metric, metric)} vs Epoch (First {max_epoch} Epochs)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{split}_{metric}.png", dpi=180)
    plt.close()


def plot_best_metric_bar(run_histories: Dict[str, List[Dict[str, Any]]], metric: str, output_dir: Path, max_epoch: int) -> None:
    labels: List[str] = []
    values: List[float] = []
    colors: List[str] = []
    for optimizer_name, records in run_histories.items():
        _, series = extract_series(records, "val", metric)
        if not series:
            continue
        labels.append(optimizer_name.upper())
        values.append(max(series))
        colors.append(COLORS.get(optimizer_name, "#4c78a8"))

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel(f"Best Validation {METRIC_LABELS.get(metric, metric)}")
    plt.title(f"Best Validation {METRIC_LABELS.get(metric, metric)} Within First {max_epoch} Epochs")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / f"best_val_{metric}_bar.png", dpi=180)
    plt.close()


def plot_final_metric_bar(run_histories: Dict[str, List[Dict[str, Any]]], split: str, metric: str, output_dir: Path, max_epoch: int) -> None:
    labels: List[str] = []
    values: List[float] = []
    colors: List[str] = []
    for optimizer_name, records in run_histories.items():
        _, series = extract_series(records, split, metric)
        if not series:
            continue
        labels.append(optimizer_name.upper())
        values.append(series[-1])
        colors.append(COLORS.get(optimizer_name, "#4c78a8"))

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel(f"Final {split.upper()} {METRIC_LABELS.get(metric, metric)}")
    plt.title(f"Final {split.upper()} {METRIC_LABELS.get(metric, metric)} Within First {max_epoch} Epochs")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / f"final_{split}_{metric}_bar.png", dpi=180)
    plt.close()


def plot_final_per_class_iou(run_histories: Dict[str, List[Dict[str, Any]]], output_dir: Path, max_epoch: int) -> None:
    class_names: List[str] | None = None
    data: Dict[str, List[float]] = {}
    for optimizer_name, records in run_histories.items():
        if not records:
            continue
        val = records[-1].get("val", {})
        named = val.get("per_class_iou_named")
        if not named:
            continue
        if class_names is None:
            class_names = list(named.keys())
        data[optimizer_name] = [float(named[name]) if named[name] == named[name] else 0.0 for name in class_names]

    if not data or class_names is None:
        return

    plt.figure(figsize=(14, 6))
    x = range(len(class_names))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for offset, (optimizer_name, values) in zip(offsets, data.items()):
        shifted = [idx + offset * width for idx in x]
        plt.bar(shifted, values, width=width, label=optimizer_name.upper(), color=COLORS.get(optimizer_name))

    plt.xticks(list(x), class_names, rotation=45, ha="right")
    plt.ylabel("Validation IoU")
    plt.title(f"Final Validation Per-Class IoU Within First {max_epoch} Epochs")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_dir / "final_val_per_class_iou.png", dpi=180)
    plt.close()


def write_metric_catalog(output_dir: Path, max_epoch: int) -> None:
    content = f"""Features we compare across optimizers (first {max_epoch} epochs)\n\nPrimary accuracy / quality metrics\n- val.miou\n- train.miou\n- val.pixel_accuracy\n- train.pixel_accuracy\n- val.mean_loss\n- train.mean_loss\n- val.mean_main_loss\n- train.mean_main_loss\n- val.mean_aux_loss\n- train.mean_aux_loss\n- val.per_class_iou_named\n\nEfficiency / systems metrics\n- train.mean_step_time_sec\n- train.mean_forward_time_sec\n- train.mean_backward_time_sec\n- train.mean_optimizer_time_sec\n- train.examples_per_sec\n- val.mean_step_time_sec\n- val.examples_per_sec\n- train.epoch_time_sec\n- val.epoch_time_sec\n\nMemory / optimizer-state metrics\n- train.optimizer_state_bytes\n- train.parameter_bytes\n"""
    (output_dir / "metric_catalog.txt").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dirs = [Path(path) for path in args.run_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_histories: Dict[str, List[Dict[str, Any]]] = {}
    for run_dir in run_dirs:
        history_path = run_dir / "history.jsonl"
        config_path = run_dir / "config.yaml"
        if not history_path.exists() or not config_path.exists():
            print(f"Skipping missing run artifacts in: {run_dir}")
            continue
        optimizer_name = infer_optimizer_label(run_dir)
        run_histories[optimizer_name] = load_history(history_path, max_epoch=args.max_epoch)

    if not run_histories:
        raise FileNotFoundError("No valid run directories with history.jsonl/config.yaml were found.")

    for metric in TRAIN_METRICS:
        plot_line_metric(run_histories, split="train", metric=metric, output_dir=output_dir, max_epoch=args.max_epoch)
    for metric in VAL_METRICS:
        plot_line_metric(run_histories, split="val", metric=metric, output_dir=output_dir, max_epoch=args.max_epoch)

    plot_best_metric_bar(run_histories, metric=args.primary_metric, output_dir=output_dir, max_epoch=args.max_epoch)
    plot_final_metric_bar(run_histories, split="val", metric=args.primary_metric, output_dir=output_dir, max_epoch=args.max_epoch)
    plot_final_metric_bar(run_histories, split="train", metric="mean_optimizer_time_sec", output_dir=output_dir, max_epoch=args.max_epoch)
    plot_final_metric_bar(run_histories, split="train", metric="optimizer_state_bytes", output_dir=output_dir, max_epoch=args.max_epoch)
    plot_final_per_class_iou(run_histories, output_dir=output_dir, max_epoch=args.max_epoch)
    write_metric_catalog(output_dir, max_epoch=args.max_epoch)

    print(f"Saved comparison plots to: {output_dir}")


if __name__ == "__main__":
    main()
