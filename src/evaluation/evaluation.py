import cv2
import numpy as np
import json

from pathlib import Path
from typing import Any, Mapping
from pathlib import Path
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm

from src.utils.cli_utils import print_kv
from src.utils.timing import TimingCollector, print_sample_timings, timed_block
from src.data.datasets import ImageFolderWithGT
from src.evaluation.pro_curve_util import compute_pro, trapezoid
from src.inference.inference import EfficientADContext, prepare_output, run_image_inference, extract_and_save_polygons, render_and_save_visualizations, write_score_row, save_component_maps
from src.evaluation.plotting import plot_evaluation_summary


def evaluate(
    dataset: Path | ImageFolderWithGT,
    ctx: EfficientADContext,
    normal_classes: set[str],
    output_dir: Path | None = None,
    leave_pbar: bool = True,
):
    if isinstance(dataset, Path):
        dataset = ImageFolderWithGT(dataset)

    cfg = ctx.inference_params
    device = ctx.device
    save_component_maps_enabled = cfg.save_component_maps and output_dir is not None

    roots, csv_file, csv_writer = prepare_output(
        output_dir,
        with_labels=True,
        save_component_maps=save_component_maps_enabled,
    )

    y_true_img, y_score_img = [], []
    y_true_px, y_score_px = [], []
    gt_maps, anomaly_maps = [], []

    collector = TimingCollector()
    num_warmup_samples = int(cfg.warmup or 0)
    num_processed_samples = 0

    try:
        for image, gt_full_res, path in tqdm(dataset, desc="Evaluation", leave=leave_pbar):
            timings = {}

            with timed_block("total_latency", timings, device):
                result = run_image_inference(
                    image=image,
                    ctx=ctx,
                    need_full_res=(output_dir is not None),  # full-res only when visuals are saved
                    timings=timings,
                )

                class_name = Path(path).parent.name
                label = 0 if class_name in normal_classes else 1
                score = result.score

                map_eval_res = result.map_combined_eval_res
                map_eval_res = map_eval_res.detach().cpu().numpy()

                # --- GT -> eval_res (256 x 256) ---
                gt_bin = (gt_full_res > 0).astype(np.uint8)
                gt_eval_res = cv2.resize(
                    gt_bin,
                    (map_eval_res.shape[1], map_eval_res.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

                # --- image-level collect ---
                y_true_img.append(label)
                y_score_img.append(score)

                # --- pixel-level collect ---
                y_true_px.append(gt_eval_res.ravel())
                y_score_px.append(map_eval_res.ravel())

                # --- AUPRO collect ---
                if gt_eval_res.sum() > 0:
                    gt_maps.append(gt_eval_res)
                    anomaly_maps.append(map_eval_res.astype(np.float32))

                if output_dir is not None:
                    with timed_block("outputs", timings, device):
                        write_score_row(csv_writer, path, class_name, score, label)

                        map_full_res = result.map_combined_full_res
                        assert map_full_res is not None
                        map_full_res = map_full_res.detach().cpu().numpy()

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

                        if save_component_maps_enabled:
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

    # --- Compute metrics ---
    print("\nCompute image-level metrics... ", end="", flush=True)
    y_true_img = np.asarray(y_true_img)
    y_score_img = np.asarray(y_score_img)

    fpr_img, tpr_img, _ = roc_curve(y_true_img, y_score_img)
    img_auroc = auc(fpr_img, tpr_img)

    scores_normal = y_score_img[y_true_img == 0]
    scores_anom = y_score_img[y_true_img == 1]

    threshold = cfg.image_level.threshold
    y_pred_img = y_score_img >= threshold

    tn, fp, fn, tp = confusion_matrix(
        y_true_img,
        y_pred_img,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(y_true_img, y_pred_img, zero_division=0)
    recall = recall_score(y_true_img, y_pred_img, zero_division=0)
    f1 = f1_score(y_true_img, y_pred_img, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("✓")

    print("Compute pixel-level metrics... ", end="", flush=True)
    y_true_px = np.concatenate(y_true_px)
    y_score_px = np.concatenate(y_score_px)

    px_auroc = roc_auc_score(y_true_px, y_score_px)
    
    fprs, pros = compute_pro(anomaly_maps=anomaly_maps, ground_truth_maps=gt_maps)
    integration_limit = 0.3
    px_aupro = trapezoid(fprs, pros, x_max=integration_limit)
    px_aupro /= integration_limit

    print("✓")

    # --- Plotting ---
    plot_evaluation_summary(
        fpr_img=fpr_img,
        tpr_img=tpr_img,
        img_auroc=img_auroc,
        scores_normal=scores_normal,
        scores_anom=scores_anom,
        fprs=fprs,
        pros=pros,
        score_threshold=threshold,
        save_path=(
            output_dir / "plots.png"
            if output_dir is not None
            else None
        ),
    )

    metrics = {
        "classification": {
            "threshold": threshold,
            "image_auroc": img_auroc,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
        },
        "localization": {
            "pixel_auroc": px_auroc,
            "pixel_aupro": px_aupro,
        },
    }

    print("\n[Evaluation Results]")
    print_kv("Image AUROC", f"{img_auroc:.4f}")
    print_kv("Pixel AUROC", f"{px_auroc:.4f}")
    print_kv("Pixel AUPRO", f"{px_aupro:.4f}")
    print_kv("TP / FP / TN / FN", f"{tp} / {fp} / {tn} / {fn}")
    print_kv("Precision", f"{precision:.4f}")
    print_kv("Recall", f"{recall:.4f}")
    print_kv("Specificity", f"{specificity:.4f}")
    print_kv("F1-Score", f"{f1:.4f}")

    per_sample_timings = collector.summary()    
    if output_dir is not None:
        print_sample_timings(summary=per_sample_timings)

    return (
        metrics,
        per_sample_timings,
        num_processed_samples,
    )


def save_metrics_json(
    output_dir: Path,
    metrics: Mapping[str, Any],
    filename: str = "metrics.json",
    indent: int = 2,
) -> Path:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / filename
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=indent)

    return out_file
