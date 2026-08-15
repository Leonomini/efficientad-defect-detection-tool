from datetime import datetime
from pathlib import Path


def resolve_checkpoint(
    cli_checkpoint: str | None,
    cfg_checkpoint: str | None,
) -> Path:

    checkpoint = cli_checkpoint or cfg_checkpoint
    if checkpoint is None:
        raise ValueError(
            "Checkpoint must be specified either via --checkpoint "
            "or in cfg.paths.checkpoint."
        )

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path


def resolve_dataset(
    cli_dataset: str | None,
    cfg_dataset_root: str,
    cfg_dataset_split: str,
) -> Path:

    if cli_dataset is not None:
        dataset_path = Path(cli_dataset)
    else:
        dataset_path = Path(cfg_dataset_root) / cfg_dataset_split

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return dataset_path


def resolve_output_dir(
    cli_output_root: str | None,
    cfg_output_root: str,
    dataset: Path,
    subdir: str,
    timestamp: str | None = None,
) -> Path:

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    root = Path(cli_output_root) if cli_output_root else Path(cfg_output_root)

    output_dir = root / subdir / f"{timestamp}_{dataset.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def print_kv(key, value, width=20):
    print(f"{key:<{width}s}: {value}")
