import argparse
from src.utils.cli_utils import print_kv
from src.utils.load_config import load_config
from src.pretraining.teacher_trainer import TeacherTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain EfficientAD teacher")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/pretraining.yaml",
        help=(
            "Path to the configuration file (YAML). "
            "Default: configs/pretraining.yaml"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    print_kv("Config", args.config)

    pretrainer = TeacherTrainer(cfg)
    pretrainer.run()


if __name__ == "__main__":
    main()
