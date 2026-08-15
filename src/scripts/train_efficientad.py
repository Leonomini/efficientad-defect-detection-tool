import argparse
from src.training.trainer import Trainer
from src.utils.load_config import load_config
from src.utils.cli_utils import print_kv


def parse_args():
    parser = argparse.ArgumentParser(description="Train an EfficientAD model on a dataset")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/efficientad.yaml",
        help=(
            "Path to the configuration file (YAML). "
            "Default: configs/efficientad.yaml"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    print_kv("Config", args.config)

    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
