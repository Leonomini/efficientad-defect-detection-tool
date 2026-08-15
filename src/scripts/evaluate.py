import argparse
from datetime import datetime
import time

from src.utils.timing import compute_pipeline_timings, print_pipeline_timings
from src.inference.inference import load_efficientad
from src.utils.load_config import load_config
from src.evaluation.evaluation import evaluate, save_metrics_json
from src.utils.to_class_set import to_class_set
from src.utils.cli_utils import print_kv, resolve_checkpoint, resolve_dataset, resolve_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained EfficientAD model on a dataset.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/efficientad.yaml",
        help=(
            "Path to the configuration file (YAML). "
            "Default: configs/efficientad.yaml"
        ),
    )
    parser.add_argument(
        "-ch", "--checkpoint",
        type=str,
        help=(
            "Path to the trained EfficientAD checkpoint. "
            "Overrides the checkpoint path defined in the config file."
        ),
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        help=(
            "Path to the dataset root directory (e.g. MVTec test folder). "
            "Overrides the dataset path defined in the config file."
        ),
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        help=(
            "Custom root directory for outputs. "
            "Overrides the output path defined in the config file."
        ),
    )
    parser.add_argument(
        "--save-only-metrics",
        action="store_true",
        help=(
            "Disable saving heatmaps, region overlays and polygon annotations. "
            "Only evaluation metrics will be computed."
        ),
    )
    parser.add_argument(
        "--save-component-maps",
        action="store_true",
        help=(
            "Save individual EfficientAD anomaly maps "
            "(combined, student-teacher, autoencoder). "
            "Overrides the config setting."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.save_component_maps:
        cfg.inference.save_component_maps = True

    t_start = time.perf_counter()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    checkpoint = resolve_checkpoint(
        cli_checkpoint=args.checkpoint,
        cfg_checkpoint=getattr(cfg.paths, "checkpoint", None),
    )
    dataset = resolve_dataset(
        cli_dataset=args.dataset,
        cfg_dataset_root=cfg.dataset.root,
        cfg_dataset_split=cfg.dataset.splits.test,
    )
    output_dir = resolve_output_dir(
        cli_output_root=args.output_dir,
        cfg_output_root=cfg.paths.output_dir,
        dataset=dataset,
        subdir="evaluation",
        timestamp=timestamp,
    )

    print_kv("Config", args.config)
    print_kv("Checkpoint", checkpoint)
    print_kv("Dataset", dataset)
    print_kv("Output dir", output_dir)

    ctx = load_efficientad(cfg, checkpoint)

    normal_classes = to_class_set(cfg.dataset.normal_subdirs)

    metrics, per_sample_timings, num_processed_samples = evaluate(
        dataset=dataset,
        ctx=ctx,
        normal_classes=normal_classes,
        output_dir=None if args.save_only_metrics else output_dir,
    )
    c = metrics["classification"]
    l = metrics["localization"]

    t_end = time.perf_counter()
    pipeline_runtime = t_end - t_start
    pipeline_timings = compute_pipeline_timings(
        total_seconds=pipeline_runtime,
        num_samples=num_processed_samples, # total processed samples (incl. warmup)
    )

    metrics_json = {
        "dataset": str(dataset),
        "model": str(checkpoint),
        "timestamp": timestamp,
        "classification": {
            "image_auroc": round(c["image_auroc"], 4),
            "precision": round(c["precision"], 4),
            "recall": round(c["recall"], 4),
            "specificity": round(c["specificity"], 4),
            "f1_score": round(c["f1_score"], 4),
            "threshold": c["threshold"],
            "tp": c["tp"],
            "fp": c["fp"],
            "tn": c["tn"],
            "fn": c["fn"],
        },
        "localization": {
            "pixel_auroc": round(l["pixel_auroc"], 4),
            "pixel_aupro": round(l["pixel_aupro"], 4),
        },
        "pipeline_performance": {
            "run_total_samples": pipeline_timings["run_total_samples"],
            "run_total_seconds": round(pipeline_timings["run_total_seconds"], 2),
            "avg_ms_per_sample": round(pipeline_timings["avg_ms_per_sample"], 2),
            "avg_fps": round(pipeline_timings["avg_fps"], 2),
        },
    }
    
    if not args.save_only_metrics:
        metrics_json["per_sample_breakdown_in_ms"] = {
            "forward": round(per_sample_timings["forward"] * 1000, 2),
            "outputs": round(per_sample_timings["outputs"] * 1000, 2),
            "total_latency": round(per_sample_timings["total_latency"] * 1000, 2),
            "processed_samples": per_sample_timings["processed_samples"],
            "warmup_samples": cfg.inference.warmup,
        }

    save_metrics_json(output_dir, metrics_json)

    print_pipeline_timings(pipeline_timings)
    print("Done.")



if __name__ == "__main__":
    main()
