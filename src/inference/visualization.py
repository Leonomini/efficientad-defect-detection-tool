import cv2
import numpy as np


def overlay_heatmap(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
):
    # Normalize anomaly map to [0, 255]
    amap = anomaly_map - anomaly_map.min()
    amap = amap / (amap.max() + 1e-8)
    amap = (amap * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(amap, colormap)

    # OpenCV uses BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Overlay
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay


def overlay_regions(
    image: np.ndarray,
    polygons: list[np.ndarray],
    alpha: float = 0.35,
    fill_color: tuple[int, int, int] = (0, 0, 255),  # BGR: rot
    outline_color: tuple[int, int, int] = (0, 255, 255),  # BGR: gelb
    outline_thickness: int = 2,
):

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if not polygons:
        return image_bgr

    # Ensure polygons are int32 OpenCV format
    polys = []
    for p in polygons:
        p = np.asarray(p)
        if p.ndim == 2:
            p = p[:, None, :]  # (N, 1, 2)
        polys.append(p.astype(np.int32))

    overlay = image_bgr.copy()
    cv2.fillPoly(overlay, polys, fill_color)

    # Blend fill
    blended = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)

    # Draw outlines on top (optional)
    # cv2.polylines(blended, polys, isClosed=True, color=outline_color, thickness=outline_thickness)

    return blended
