import cv2
import json
import numpy as np
from typing import List
from pathlib import Path


def extract_polygons(
    anomaly_map: np.ndarray,
    threshold: float | None = None,
    quantile: float = 0.95,
    min_area: int = 200,
    morph_kernel_size: int = 5,
    close_iter: int = 2,
    open_iter: int = 1,
    epsilon_factor: float = 0.01,
) -> List[np.ndarray]:

    # --- Thresholding ---
    if threshold is None:
        threshold = np.quantile(anomaly_map, quantile)

    binary = (anomaly_map > threshold).astype(np.uint8) * 255

    # --- Morphological cleanup ---
    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_size, morph_kernel_size),
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=close_iter,
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=open_iter,
        )

    # --- Find contours ---
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons = []

    # --- Contour filtering & polygon approximation ---
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, epsilon, True)

        if poly.shape[0] >= 3:  # valid polygon
            polygons.append(poly)

    return polygons


def polygons_to_labelme(
    polygons: List[np.ndarray],
    image_path: str,
    image_shape: tuple[int, int],
    label: str = "anomaly",
    version: str = "5.8.3",
) -> dict:

    shapes = []

    for poly in polygons:
        # Ensure shape (N, 2)
        poly = np.squeeze(poly)
        if poly.ndim != 2 or poly.shape[0] < 3:
            continue

        points = poly.astype(float).tolist()

        shapes.append({
            "label": label,
            "points": points,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        })

    labelme_dict = {
        "version": version,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_shape[0],
        "imageWidth": image_shape[1],
    }

    return labelme_dict


def save_labelme_json(labelme_dict: dict, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(labelme_dict, f, indent=2)
