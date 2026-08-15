import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import Wide_ResNet101_2_Weights
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from src.utils.device import resolve_device
from src.utils.seed import set_seed
from src.utils.cli_utils import print_kv
from src.models.pdn import build_pdn_small, build_pdn_medium
from src.data.datasets import ImageFolderWithoutTarget, InfiniteDataloader
from src.pretraining.feature_extractor import FeatureExtractor
from src.pretraining.normalization import feature_normalization
from src.pretraining.transforms import train_transform


class TeacherTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = resolve_device(cfg.general.device)
        set_seed(cfg.general.seed)

        # --- Paths ---
        self.output_folder = Path(cfg.paths.output_dir) / "teacher"
        self.imagenet_train_path = Path(cfg.paths.imagenet_train)

        self.model_size = cfg.model.size
        self.steps = cfg.training.steps

        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self._build_models()
        self._build_dataloader()
        self._build_optimizer()


    def _build_models(self):
        out_channels = self.cfg.model.out_channels
        input_shape = tuple(self.cfg.data.input_shape)
        layers_to_extract = self.cfg.data.layers_to_extract_from

        backbone = torchvision.models.wide_resnet101_2(
            weights=Wide_ResNet101_2_Weights.IMAGENET1K_V1
        )

        self.extractor = FeatureExtractor(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract,
            device=self.device,
            input_shape=input_shape,
            out_channels=out_channels,
        )

        if self.model_size == "small":
            self.pdn = build_pdn_small(out_channels, padding=True)
        elif self.model_size == "medium":
            self.pdn = build_pdn_medium(out_channels, padding=True)
        else:
            raise ValueError(f"Unknown model size: {self.model_size}")

        self.pdn.to(self.device)
        self.pdn.train()


    def _build_dataloader(self):
        batch_size = self.cfg.training.batch_size
        num_workers = self.cfg.training.num_workers

        train_set = ImageFolderWithoutTarget(
            self.imagenet_train_path,
            transform=train_transform,
        )

        loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=("cuda" in str(self.device)),
        )
        self.train_loader = InfiniteDataloader(loader)


    def _build_optimizer(self):
        lr = float(self.cfg.training.learning_rate)
        weight_decay = float(self.cfg.training.weight_decay)

        self.optimizer = torch.optim.Adam(
            self.pdn.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )


    def run(self):
        print_kv("Device", self.device)
        print_kv("Model size", self.model_size)
        print_kv("Steps", self.steps)
        print_kv("Dataset", self.imagenet_train_path)
        print("\n==== Starting teacher pretraining ====")

        normalization_steps = self.cfg.training.normalization_steps
        log_interval = self.cfg.training.log_interval
        check_interval = self.cfg.training.check_interval

        # --- Feature normalization ---
        channel_mean, channel_std = feature_normalization(
            extractor=self.extractor,
            train_loader=self.train_loader,
            device=self.device,
            steps=normalization_steps,
        )

        channel_mean = channel_mean.to(self.device)
        channel_std = channel_std.to(self.device)

        tqdm_obj = tqdm(range(self.steps), desc="Teacher pretraining")

        for iteration, (image_fe, image_pdn) in zip(tqdm_obj, self.train_loader):
            image_fe = image_fe.to(self.device)
            image_pdn = image_pdn.to(self.device)

            # --- Extractor without gradients ---
            with torch.inference_mode():
                target = self.extractor.embed(image_fe)
                target = (target - channel_mean) / channel_std

            prediction = self.pdn(image_pdn)
            loss = torch.mean((target - prediction) ** 2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # --- Logging ---
            if iteration % log_interval == 0:
                tqdm_obj.set_postfix(loss=f"{loss.item():.4f}")

            # --- Save checkpoint ---
            if iteration > 0 and iteration % check_interval == 0:
                self._save_checkpoint(suffix="tmp")

        # --- Final checkpoint ---
        self._save_checkpoint(suffix="final")

        print("\n==== Teacher pretraining finished ====")
        print_kv("Final checkpoint", self.output_folder / f"teacher_{self.model_size}_final.pth")
        print_kv("Final state dict", self.output_folder / f"teacher_{self.model_size}_final_state.pth")


    def _save_checkpoint(self, suffix: str):
        torch.save(
            self.pdn,
            self.output_folder / f"teacher_{self.model_size}_{self.timestamp}_{suffix}.pth",
        )

        torch.save(
            self.pdn.state_dict(),
            self.output_folder / f"teacher_{self.model_size}_{self.timestamp}_{suffix}_state.pth",
        )