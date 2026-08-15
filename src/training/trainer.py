import itertools
import torch
import json
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime

from src.evaluation.evaluation import evaluate
from src.inference.inference import EfficientADContext
from src.models.factory import build_models
from src.data.datasets import ImageFolderWithGT, ImageFolderWithoutTarget, InfiniteDataloader
from src.data.transforms import build_transforms, build_penalty_transforms
from src.training.normalization import map_normalization, teacher_normalization
from src.utils.device import resolve_device
from src.utils.seed import set_seed
from src.utils.to_class_set import to_class_set
from src.utils.cli_utils import print_kv
from src.utils.load_config import namespace_to_dict


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = resolve_device(cfg.general.device)
        set_seed(cfg.general.seed)

        self.dataset = cfg.dataset.name
        self.subdataset = cfg.dataset.subdataset
        self.image_size = cfg.dataset.image_size
        self.model_size = cfg.model.size
        self.out_channels = cfg.model.out_channels
        self.steps = cfg.training.steps

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self._prepare_data()
        self._prepare_models()
        self._prepare_optimizer()

        self.output_dir = Path(cfg.paths.output_dir) / "training" / self.dataset
        if self.subdataset is not None:
            self.output_dir = self.output_dir / self.subdataset
        self.output_dir = self.output_dir / self.timestamp

        self.checkpoint_dir = self.output_dir / "checkpoints"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_score = None
        self.best_step = None


    def _prepare_data(self):
        cfg = self.cfg
        dataset_root = Path(cfg.dataset.root)
        train_dir = dataset_root / cfg.dataset.splits.train
        test_dir = dataset_root / cfg.dataset.splits.val
        
        # Transforms
        # ------------------------------------------------------------------
        self.default_tf, self.train_tf = build_transforms(self.image_size)

        # Datasets
        # ------------------------------------------------------------------
        self.test_set = ImageFolderWithGT(test_dir)

        full_train_set = ImageFolderWithoutTarget(
            train_dir,
            transform=transforms.Lambda(self.train_tf)
        )
        
        if cfg.dataset.splits.map_norm is not None:
            self.train_set = full_train_set
            val_dir = dataset_root / cfg.dataset.splits.map_norm
            self.val_set = ImageFolderWithoutTarget(
                val_dir,
                transform=transforms.Lambda(self.train_tf),
            )
        else:
            # mvtec dataset paper recommend 10% validation set
            train_size = int(0.9 * len(full_train_set))
            val_size = len(full_train_set) - train_size

            rng = torch.Generator().manual_seed(cfg.general.seed)
            self.train_set, self.val_set = torch.utils.data.random_split(
                full_train_set, 
                [train_size, val_size], 
                rng
            )
        
        print_kv("Dataset", self.dataset)
        if self.subdataset is not None:
            print_kv("Subdataset", self.subdataset)
        print_kv("Test images", len(self.test_set))
        print_kv("Training images", len(self.train_set))
        print_kv("Validation images", len(self.val_set))

        # Dataloaders
        # ------------------------------------------------------------------
        self.train_loader = InfiniteDataloader(self._build_dataloader(self.train_set))

        self.val_loader = self._build_dataloader(self.val_set, shuffle=False)

        # ImageNet penalty (optional)
        # ------------------------------------------------------------------        
        self.penalty_enabled = cfg.training.imagenet_penalty.enabled
        print_kv("ImageNet penalty enabled", self.penalty_enabled)

        if self.penalty_enabled and cfg.training.imagenet_penalty.path:
            penalty_transform = build_penalty_transforms(self.image_size)
            penalty_set = ImageFolderWithoutTarget(
                cfg.training.imagenet_penalty.path, transform=penalty_transform
            )
            self.penalty_loader = InfiniteDataloader(self._build_dataloader(penalty_set))
        else:
            self.penalty_loader = itertools.repeat(None)


    def _prepare_models(self):
        self.teacher, self.student, self.autoencoder = build_models(self.out_channels, self.model_size)

        weight_path = self.cfg.paths.teacher_weights
        weight_path = getattr(weight_path, self.model_size)
        self.teacher.load_state_dict(torch.load(weight_path, map_location="cpu"))
        self.teacher.requires_grad_(False)

        for model in (self.teacher, self.student, self.autoencoder):
            model.to(self.device)

        self.teacher.eval()
        self.student.train()
        self.autoencoder.train()

        # Teacher normalization
        # ------------------------------------------------------------------
        print("\n[Starting teacher normalization]")
        self.teacher_mean, self.teacher_std = teacher_normalization(
            self.teacher,
            self._build_dataloader(self.train_set),
            self.device,
        )


    def _prepare_optimizer(self):
        cfg = self.cfg.training
        self.optimizer = torch.optim.Adam(
            itertools.chain(
                self.student.parameters(), 
                self.autoencoder.parameters()),
                lr=float(cfg.learning_rate),
                weight_decay=float(cfg.weight_decay),
        )


    def _build_dataloader(self, dataset, shuffle=True):
        num_workers = self.cfg.training.dataloader.num_workers
        pin_memory = self.cfg.training.dataloader.pin_memory
        batch_size = self.cfg.training.batch_size

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=pin_memory,
        )
    

    def run(self):
        cfg = self.cfg.training
        q_hard = cfg.hard_mining_quantile
        lr_milestone = cfg.lr_decay.step
        lr_new = float(cfg.lr_decay.new_value)
        step_width = len(str(self.steps))

        log_interval = cfg.logging.log_interval
        eval_interval = cfg.logging.eval_interval

        steps_history = []
        loss_history = []
        loss_st_history = []
        loss_ae_history = []
        loss_stae_history = []

        self._save_run_config()

        print("\n[Starting EfficientAD training]")
        print_kv("Device", self.device)
        print_kv("Model size", self.model_size)
        print_kv("Steps", self.steps)
        print()

        # Training loop
        # ------------------------------------------------------------------
        pbar = tqdm(range(self.steps), desc="Training")
        
        for step, (img_st, img_ae), img_penalty in zip(
            pbar, self.train_loader, self.penalty_loader
        ):
            img_st = img_st.to(self.device)
            img_ae = img_ae.to(self.device)

            if img_penalty is not None:
                img_penalty = img_penalty.to(self.device)
            
            # --- Student-Teacher loss ---
            with torch.inference_mode():
                teacher_output_st = (self.teacher(img_st) - self.teacher_mean) / self.teacher_std

            student_output_st = self.student(img_st)[:, :self.out_channels]
            distance_st = (teacher_output_st - student_output_st) ** 2

            d_hard = torch.quantile(distance_st, q=q_hard)
            loss_st = torch.mean(distance_st[distance_st >= d_hard])

            if img_penalty is not None:
                student_output_penalty = self.student(img_penalty)[:, :self.out_channels]
                loss_penalty = torch.mean(student_output_penalty**2)
                loss_st += loss_penalty

            # --- Autoencoder loss ---
            ae_output = self.autoencoder(img_ae)
            with torch.inference_mode():
                teacher_output_ae = (self.teacher(img_ae) - self.teacher_mean) / self.teacher_std

            student_output_ae = self.student(img_ae)[:, self.out_channels:]
            distance_ae = (teacher_output_ae - ae_output)**2
            distance_stae = (ae_output - student_output_ae)**2

            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)
            loss = loss_st + loss_ae + loss_stae

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # --- Learning rate decay ---
            if step == lr_milestone:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr_new
                tqdm.write(f"[Step {step}] Learning rate decayed to {lr_new}")

            # --- Loss logging ---
            if step % log_interval == 0:
                total_loss = float(loss.item())
                st_loss = float(loss_st.item())
                ae_loss = float(loss_ae.item())
                stae_loss = float(loss_stae.item())

                pbar.set_postfix(loss=f"{total_loss:.4f}")

                steps_history.append(step)
                loss_history.append(total_loss)
                loss_st_history.append(st_loss)
                loss_ae_history.append(ae_loss)
                loss_stae_history.append(stae_loss)

            # --- Intermediate evaluation ---
            if step % eval_interval == 0 and step > 0:
                tqdm.write("===========================================================")
                tqdm.write(f"[Step {step}: Intermediate evaluation]")
                with tqdm.external_write_mode():
                    metrics = self._run_evaluation()
                    print()

                img_auroc, precision, recall, f1, px_auroc, px_aupro = self._extract_metrics(metrics)
                self._log_eval_results(step, img_auroc, precision, recall, f1, px_auroc, px_aupro)
                
                if self._is_best_model(img_auroc, px_auroc, px_aupro):
                    self.best_step = step
                    self._save_checkpoint(step=step, width=step_width)

                self.teacher.eval()
                self.student.train()
                self.autoencoder.train()

        # Final evaluation and save
        # ------------------------------------------------------------------
        print("\n===========================================================")
        print("[Final evaluation]")
        metrics = self._run_evaluation()

        img_auroc, precision, recall, f1, px_auroc, px_aupro = self._extract_metrics(metrics)
        self._log_eval_results(self.steps, img_auroc, precision, recall, f1, px_auroc, px_aupro)
        self._save_checkpoint(step=self.steps, width=step_width, is_final=True)
        self._save_loss_plot(
            steps_history,
            loss_history,
            loss_st_history,
            loss_ae_history,
            loss_stae_history,
        )
        if self._is_best_model(img_auroc, px_auroc, px_aupro):
            self.best_step = self.steps

        print("\nTraining finished successfully.")
        if self.best_step is not None:
            print_kv("Best checkpoint step", self.best_step)
        print_kv("Results saved to", self.output_dir)


    def _run_evaluation(self):
        self.teacher.eval()
        self.student.eval()
        self.autoencoder.eval()

        q_start = self.cfg.training.normalization.q_start
        q_end = self.cfg.training.normalization.q_end

        # qa = lower quantile, qb = upper quantile
        self.qa_st, self.qb_st, self.qa_ae, self.qb_ae = map_normalization(
            validation_loader=self.val_loader, 
            teacher=self.teacher,
            student=self.student, 
            autoencoder=self.autoencoder,
            teacher_mean=self.teacher_mean, 
            teacher_std=self.teacher_std,
            device=self.device,
            q_start=q_start,
            q_end=q_end,
            leave_pbar=False,
        )

        ctx = self._build_inference_context()
        normal_classes = to_class_set(self.cfg.dataset.normal_subdirs)

        metrics, _, _ = evaluate(
            dataset=self.test_set,
            ctx=ctx,
            normal_classes=normal_classes,
            output_dir=None,
            leave_pbar=False,
        )
        return metrics
    

    def _build_inference_context(self):
        return EfficientADContext(
            teacher=self.teacher,
            student=self.student,
            autoencoder=self.autoencoder,

            teacher_mean=self.teacher_mean,
            teacher_std=self.teacher_std,

            qa_st=self.qa_st,
            qb_st=self.qb_st,
            qa_ae=self.qa_ae,
            qb_ae=self.qb_ae,

            transform=self.default_tf,
            device=self.device,
            image_size=self.image_size,
            inference_params=self.cfg.inference,
        )


    def _log_eval_results(self, step, img_auroc, precision, recall, f1, px_auroc, px_aupro):
        log_file = self.output_dir / f"eval_results_{self.timestamp}.csv"
        file_exists = log_file.exists()

        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "step",
                    "image_auroc",
                    "precision",
                    "recall",
                    "f1_score",
                    "pixel_auroc",
                    "pixel_aupro",
                ])

            writer.writerow([
                step,
                f"{img_auroc:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                f"{px_auroc:.4f}",
                f"{px_aupro:.4f}",
            ])

    def _extract_metrics(self, metrics):
        c = metrics["classification"]
        l = metrics["localization"]
        return (
            float(c["image_auroc"]),
            float(c["precision"]),
            float(c["recall"]),
            float(c["f1_score"]),
            float(l["pixel_auroc"]),
            float(l["pixel_aupro"]),
        )

    def _save_run_config(self):
        config_file = self.output_dir / f"run_config_{self.timestamp}.json"

        config_data = {
            "timestamp": self.timestamp,
            "dataset": str(self.dataset),
            "subdataset": self.subdataset if self.subdataset else "NA",
            "model_size": self.model_size,
            "num_train_images": len(self.train_set),
            "run_config": namespace_to_dict(self.cfg.training),
        }

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)


    def _build_checkpoint_payload(self):
        required = ["qa_st", "qb_st", "qa_ae", "qb_ae"]
        if not all(hasattr(self, k) for k in required):
            raise RuntimeError("Stats not initialized. Run map normalization first.")
        
        stats = {
            "teacher_mean": self.teacher_mean,
            "teacher_std": self.teacher_std,
            "qa_st": self.qa_st,
            "qb_st": self.qb_st,
            "qa_ae": self.qa_ae,
            "qb_ae": self.qb_ae,
        }

        return {
            "teacher": self.teacher.state_dict(),
            "student": self.student.state_dict(),
            "autoencoder": self.autoencoder.state_dict(),
            "model_size": self.model_size,
            "out_channels": self.out_channels,
            "image_size": self.image_size,
            "stats": stats,
        }


    def _save_checkpoint(self, step: int, width: int, is_final: bool = False):
        checkpoint = self._build_checkpoint_payload()

        if is_final:
            filename = f"efficientad_{self.model_size}_step_{step:0{width}d}_{self.timestamp}_final.pth"
        else:
            filename = f"efficientad_{self.model_size}_step_{step:0{width}d}_{self.timestamp}.pth"
        
        torch.save(checkpoint, self.checkpoint_dir / filename)


    def _is_best_model(self, img_auroc, px_auroc, px_aupro):
        score = (img_auroc + px_auroc + px_aupro) / 3.0

        if self.best_model_score is None or score > self.best_model_score:
            self.best_model_score = score
            return True

        return False


    def _save_loss_plot(
    self,
    steps_history,
    loss_history,
    loss_st_history,
    loss_ae_history,
    loss_stae_history,
):
        if not steps_history:
            return

        def _smooth(values, window=50):
            if len(values) < window:
                return values
            return [
                sum(values[i:i + window]) / window
                for i in range(len(values) - window + 1)
            ]

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        plots = [
            (loss_history, "Total Loss"),
            (loss_st_history, "Loss ST"),
            (loss_ae_history, "Loss AE"),
            (loss_stae_history, "Loss STAE"),
        ]

        for ax, (values, title) in zip(axs.flat, plots):
            ax.plot(steps_history, values, alpha=0.3, linewidth=1.0, label="raw")

            smooth_values = _smooth(values, window=50)
            if len(values) >= 50:
                smooth_steps = steps_history[49:]
            else:
                smooth_steps = steps_history

            ax.plot(smooth_steps, smooth_values, linewidth=1.0, label="smoothed")

            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig.tight_layout()

        plot_file = self.output_dir / f"loss_curve_{self.timestamp}.png"
        fig.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)