import argparse
from datetime import datetime
import time

from src.utils.timing import compute_pipeline_timings, print_pipeline_timings
from src.inference.inference import run_dataset_inference, load_efficientad
from src.utils.load_config import load_config
from src.utils.cli_utils import print_kv, resolve_checkpoint, resolve_dataset, resolve_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch inference with a trained EfficientAD model on a dataset.")
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
        cfg_dataset_split=cfg.dataset.splits.val,
    )
    output_dir = resolve_output_dir(
        cli_output_root=args.output_dir,
        cfg_output_root=cfg.paths.output_dir,
        dataset=dataset,
        subdir="inference",
        timestamp=timestamp,
    )

    print_kv("Config", args.config)
    print_kv("Checkpoint", checkpoint)
    print_kv("Dataset", dataset)
    print_kv("Output dir", output_dir)

    ctx = load_efficientad(cfg, checkpoint)

    num_samples = run_dataset_inference(
        dataset=dataset,
        ctx=ctx,
        output_dir=output_dir,
    )

    t_end = time.perf_counter()
    pipeline_runtime = t_end - t_start
    pipeline_timings = compute_pipeline_timings(
        total_seconds=pipeline_runtime,
        num_samples=num_samples, # total processed samples (incl. warmup)
    )

    print_pipeline_timings(pipeline_timings)
    print("Done.")


if __name__ == "__main__":
    main()
