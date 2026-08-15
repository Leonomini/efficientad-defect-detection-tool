import shutil
import csv
import torch
import cv2
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.utils.timing import TimingCollector, print_sample_timings, timed_block
from src.inference.polygons import extract_polygons, polygons_to_labelme, save_labelme_json
from src.inference.visualization import overlay_heatmap, overlay_regions
from src.data.datasets import ImageFolderWithPath
from src.models.factory import build_models
from src.data.transforms import build_transforms
from src.utils.device import resolve_device


@dataclass(frozen=True)
class EfficientADContext:
    teacher: torch.nn.Module
    student: torch.nn.Module
    autoencoder: torch.nn.Module

    teacher_mean: torch.Tensor
    teacher_std: torch.Tensor

    qa_st: float | None
    qb_st: float | None
    qa_ae: float | None
    qb_ae: float | None

    transform: Callable
    device: torch.device
    image_size: int

    inference_params: object


@dataclass(frozen=True)
class InferenceResult:
    score: float
    map_combined_eval_res: torch.Tensor             # combined map with 256 x 256 resolution
    map_combined_full_res: torch.Tensor | None      # combined map with original image resolution (optional)
    map_st_full_res: torch.Tensor | None = None     # local anomaly map (optional)
    map_ae_full_res: torch.Tensor | None = None     # global anomaly map (optional)


def load_efficientad(cfg, checkpoint: str) -> EfficientADContext:
    device = resolve_device(cfg.general.device)

    ckpt = torch.load(checkpoint, map_location=device)

    model_size = ckpt.get("model_size")
    out_channels = ckpt.get("out_channels")
    image_size = ckpt.get("image_size")
    stats = ckpt["stats"]

    teacher, student, autoencoder = build_models(out_channels, model_size)

    teacher.load_state_dict(ckpt["teacher"])
    student.load_state_dict(ckpt["student"])
    autoencoder.load_state_dict(ckpt["autoencoder"])

    for m in (teacher, student, autoencoder):
        m.to(device)
        m.eval()

    default_transform, _ = build_transforms(image_size)

    return EfficientADContext(
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,

        teacher_mean=stats["teacher_mean"],
        teacher_std=stats["teacher_std"],

        qa_st=stats["qa_st"],
        qb_st=stats["qb_st"],
        qa_ae=stats["qa_ae"],
        qb_ae=stats["qb_ae"],

        transform=default_transform,
        device=device,
        image_size=image_size,

        inference_params=cfg.inference,
    )


def compute_anomaly_maps(
    image: torch.Tensor,
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    autoencoder: torch.nn.Module,
    teacher_mean: torch.Tensor,
    teacher_std: torch.Tensor,
    qa_st=None, 
    qb_st=None, 
    qa_ae=None, 
    qb_ae=None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_out = (teacher(image) - teacher_mean) / teacher_std
    student_out = student(image)
    autoencoder_out = autoencoder(image)

    out_channels = teacher_out.shape[1]

    # Compute anomaly maps
    map_st = torch.mean(
        (teacher_out - student_out[:, :out_channels])**2,
        dim=1, 
        keepdim=True
    )
    map_ae = torch.mean(
        (autoencoder_out - student_out[:, out_channels:])**2,
        dim=1, 
        keepdim=True
    )
    
    # Compute normalized anomaly maps
    # NOTE:
    # qa/qb normalization statistics are computed during training
    # from raw feature-resolution anomaly maps (before resizing).
    # Therefore normalization is applied before resizing the maps
    # to evaluation resolution.
    if qa_st is not None:
        map_st = 0.1 * (map_st - qa_st) / (qb_st - qa_st)

    if qa_ae is not None:
        map_ae = 0.1 * (map_ae - qa_ae) / (qb_ae - qa_ae)

    # Compute combined anomaly map
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae


@torch.inference_mode()
def run_image_inference(
    image: Image.Image,
    ctx: EfficientADContext,
    need_full_res: bool = True,
    timings: dict | None = None,
):
    
    if timings is None:
        timings = {}

    orig_width = image.width
    orig_height = image.height

    image_t = ctx.transform(image)
    image_t = image_t[None].to(ctx.device)

    with timed_block("forward", timings, ctx.device):
        map_combined, map_st, map_ae = compute_anomaly_maps(
            image=image_t,
            teacher=ctx.teacher,
            student=ctx.student,
            autoencoder=ctx.autoencoder,
            teacher_mean=ctx.teacher_mean,
            teacher_std=ctx.teacher_std,
            qa_st=ctx.qa_st,
            qb_st=ctx.qb_st,
            qa_ae=ctx.qa_ae,
            qb_ae=ctx.qb_ae,
        )

    # Resize anomaly map to evaluation resolution (256 x 256)
    # Normalization was already applied above using the training-time qa/qb statistics.
    image_size = ctx.image_size
    map_combined_eval_res = F.interpolate(
        map_combined,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    map_combined_eval_res = torch.clamp(map_combined_eval_res, 0.0, 1.0)

    # Compute image-level anomaly score
    score = torch.max(map_combined_eval_res).item()

    # Resize anomaly maps to original image resolution (if needed)
    map_combined_full_res = None
    map_st_full_res = None
    map_ae_full_res = None

    if need_full_res:
        map_combined_full_res = F.interpolate(
            map_combined,
            size=(orig_height, orig_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        map_combined_full_res = torch.clamp(map_combined_full_res, 0.0, 1.0)

    if need_full_res and ctx.inference_params.save_component_maps:
        map_st_full_res = F.interpolate(
            map_st,
            size=(orig_height, orig_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        map_ae_full_res = F.interpolate(
            map_ae,
            size=(orig_height, orig_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        map_st_full_res = torch.clamp(
            map_st_full_res,
            0.0,
            1.0,
        )

        map_ae_full_res = torch.clamp(
            map_ae_full_res,
            0.0,
            1.0,
        )

    return InferenceResult(
        score=score,
        map_combined_eval_res=map_combined_eval_res,
        map_combined_full_res=map_combined_full_res,
        map_st_full_res=map_st_full_res,
        map_ae_full_res=map_ae_full_res,
    )


def run_dataset_inference(
    dataset: Path | ImageFolderWithPath,
    ctx: EfficientADContext,
    output_dir: Path,
):
    if isinstance(dataset, Path):
        dataset = ImageFolderWithPath(dataset)

    cfg = ctx.inference_params
    device = ctx.device

    roots, csv_file, csv_writer = prepare_output(output_dir, save_component_maps=cfg.save_component_maps,)

    collector = TimingCollector()
    num_warmup_samples = int(cfg.warmup)
    num_processed_samples = 0

    try:
        for image, _, path in tqdm(dataset, desc="Inference"):
            timings = {}

            with timed_block("total_sample", timings, device):
                result = run_image_inference(
                    image=image,
                    ctx=ctx,
                    need_full_res=True,
                    timings=timings,
                )

                score = result.score
                class_name = Path(path).parent.name

                with timed_block("outputs", timings, device):
                    write_score_row(csv_writer, path, class_name, score)

                    assert result.map_combined_full_res is not None
                    map_full_res = result.map_combined_full_res.cpu().numpy()

                    polygons = extract_and_save_polygons(
                        anomaly_map=map_full_res,
                        score=score,
                        path=path,
                        class_name=class_name,
                        roots=roots,
                        cfg=cfg,
                    )

                    render_and_save_visualizations(
                        image=image,
                        anomaly_map=map_full_res,
                        polygons=polygons,
                        path=path,
                        class_name=class_name,
                        roots=roots,
                    )

                    if cfg.save_component_maps:
                        save_component_maps(
                            result=result,
                            path=path,
                            class_name=class_name,
                            roots=roots,
                        )
            
            if num_processed_samples >= num_warmup_samples:
                collector.add(timings)
            num_processed_samples += 1

    finally:
        if csv_file is not None:
            csv_file.close()

    per_image_timings = collector.summary()
    print_sample_timings(summary=per_image_timings)

    return num_processed_samples


def prepare_output(
        output_dir: Path | None,
        with_labels: bool = False,
        save_component_maps: bool = False,
    ):
    if output_dir is None:
        return None, None, None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roots = {
        "heatmap_overlays": output_dir / "heatmap_overlays",
        "region_overlays": output_dir / "region_overlays",
        "labelme": output_dir / "labelme",
    }

    if save_component_maps:
        roots["anomaly_maps"] = {
            "student_teacher": output_dir / "anomaly_maps" / "student_teacher",
            "autoencoder": output_dir / "anomaly_maps" / "autoencoder",
            "combined": output_dir / "anomaly_maps" / "combined",
        }

    for key, value in roots.items():
        if isinstance(value, dict):
            for subdir in value.values():
                subdir.mkdir(parents=True, exist_ok=True)
        else:
            value.mkdir(parents=True, exist_ok=True)

    csv_file = open(output_dir / "image_scores.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    header = ["image", "class", "score"]
    if with_labels:
        header.append("label")
    
    csv_writer.writerow(header)

    return roots, csv_file, csv_writer


def write_score_row(csv_writer, path, class_name, score, label):
    if csv_writer is None:
        return
    
    csv_writer.writerow([
        Path(path).name,
        class_name,
        f"{score:.6f}",
        label,
    ])


def render_and_save_visualizations(
    image,
    anomaly_map,
    path,
    class_name,
    roots,
    polygons=None,
):
    stem = Path(path).stem
    heatmap_dir = roots["heatmap_overlays"] / class_name
    region_dir = roots["region_overlays"] / class_name

    heatmap_dir.mkdir(parents=True, exist_ok=True)
    region_dir.mkdir(parents=True, exist_ok=True)

    image_np = np.array(image)

    overlay = overlay_heatmap(image_np, anomaly_map)
    cv2.imwrite(str(heatmap_dir / f"{stem}_overlay.png"), overlay)

    if polygons is not None and len(polygons) > 0:
        region_overlay = overlay_regions(image_np, polygons)
        cv2.imwrite(str(region_dir / f"{stem}_regions.png"), region_overlay)
    # else:
    #     shutil.copy(path, region_dir / f"{stem}_no_regions{Path(path).suffix}")


def extract_and_save_polygons(
    anomaly_map,
    score,
    path,
    class_name,
    roots,
    cfg,
):
    polygon_dir = roots["labelme"] / class_name
    polygon_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(path, polygon_dir / Path(path).name)

    if score < cfg.image_level.threshold: # extract polygons only from anomalous images
        return []

    polygons = extract_polygons(
        anomaly_map=anomaly_map,
        threshold=cfg.pixel_level.threshold,
        quantile=cfg.pixel_level.quantile,
        min_area=cfg.postprocessing.min_polygon_area,
        morph_kernel_size=cfg.postprocessing.morph_kernel_size,
        close_iter=cfg.postprocessing.close_iterations,
        open_iter=cfg.postprocessing.open_iterations,
        epsilon_factor=cfg.postprocessing.epsilon_factor,
    )

    if not polygons:
        return []

    labelme_dict = polygons_to_labelme(
        polygons=polygons,
        image_path=Path(path).name,
        image_shape=anomaly_map.shape,
        label="anomaly",
    )

    save_labelme_json(
        labelme_dict,
        polygon_dir / f"{Path(path).stem}.json",
    )

    return polygons

def save_component_maps(
    result: InferenceResult,
    path: str | Path,
    class_name: str,
    roots,
):
    stem = Path(path).stem

    maps = {
        "combined": result.map_combined_full_res,
        "student_teacher": result.map_st_full_res,
        "autoencoder": result.map_ae_full_res,
    }

    for name, anomaly_map in maps.items():
        if anomaly_map is None:
            continue

        output_dir = roots["anomaly_maps"][name] / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        img = (
            anomaly_map.detach()
            .cpu()
            .numpy()
            * 255
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(
            img,
            cv2.COLORMAP_JET,
        )

        cv2.imwrite(
            str(output_dir / f"{stem}.png"),
            heatmap,
        )
