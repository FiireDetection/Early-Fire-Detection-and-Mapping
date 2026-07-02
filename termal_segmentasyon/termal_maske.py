from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class YanginBolgesi:
    region_id: int
    pixel_area: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    centroid_x: float
    centroid_y: float
    mean_intensity: float
    max_intensity: float


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        raise ValueError("Termal frame bos olamaz.")
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _normalize01(gray: np.ndarray) -> np.ndarray:
    arr = gray.astype(np.float32)
    finite = np.isfinite(arr)
    if finite.any():
        fill = float(np.median(arr[finite]))
    else:
        fill = 0.0
    arr = np.where(finite, arr, fill).astype(np.float32)
    lo = float(np.percentile(arr, 2.0)) if arr.size else 0.0
    hi = float(np.percentile(arr, 98.0)) if arr.size else 1.0
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def maske_olustur(
    termal_frame: np.ndarray,
    threshold_mode: str = "hybrid",
    threshold_value: float = 180.0,
    percentile: float = 97.0,
    blur_kernel: int = 5,
    morph_kernel: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Termal frame'den 0/255 yangin maskesi ve normalize termal harita uretir."""
    gray = _gray(termal_frame)
    norm = _normalize01(gray)

    mode = str(threshold_mode or "hybrid").lower().strip()
    if blur_kernel and int(blur_kernel) > 1:
        k = int(blur_kernel)
        if k % 2 == 0:
            k += 1
        norm_for_threshold = cv2.GaussianBlur(norm, (k, k), 0)
    else:
        norm_for_threshold = norm

    if mode == "fixed":
        threshold = float(threshold_value) / 255.0 if threshold_value > 1.0 else float(threshold_value)
    elif mode == "percentile":
        threshold = float(np.percentile(norm_for_threshold, float(percentile)))
    elif mode == "hybrid":
        fixed = float(threshold_value) / 255.0 if threshold_value > 1.0 else float(threshold_value)
        perc = float(np.percentile(norm_for_threshold, float(percentile)))
        threshold = max(fixed, perc)
    else:
        raise ValueError("threshold_mode fixed, percentile veya hybrid olmali.")

    mask = (norm_for_threshold >= threshold).astype(np.uint8) * 255
    if morph_kernel and int(morph_kernel) > 1:
        k = int(morph_kernel)
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, norm


def bolgeleri_bul(mask: np.ndarray, norm: np.ndarray, min_area: int = 130) -> list[YanginBolgesi]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[YanginBolgesi] = []
    for idx, contour in enumerate(contours):
        area = int(cv2.contourArea(contour))
        if area < int(min_area):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-6:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx = float(x + w / 2.0)
            cy = float(y + h / 2.0)
        roi = norm[y : y + h, x : x + w]
        roi_mask = mask[y : y + h, x : x + w] > 0
        vals = roi[roi_mask] if roi_mask.any() else roi.reshape(-1)
        regions.append(
            YanginBolgesi(
                region_id=len(regions),
                pixel_area=area,
                bbox_x=int(x),
                bbox_y=int(y),
                bbox_w=int(w),
                bbox_h=int(h),
                centroid_x=cx,
                centroid_y=cy,
                mean_intensity=float(np.mean(vals)) if vals.size else 0.0,
                max_intensity=float(np.max(vals)) if vals.size else 0.0,
            )
        )
    return regions


def maske_ustune_ciz(frame: np.ndarray, mask: np.ndarray, regions: list[YanginBolgesi]) -> np.ndarray:
    gray = _gray(frame)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    heat = np.zeros_like(base)
    heat[:, :, 2] = mask
    overlay = cv2.addWeighted(base, 0.75, heat, 0.35, 0)
    for region in regions:
        x, y, w, h = region.bbox_x, region.bbox_y, region.bbox_w, region.bbox_h
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.circle(overlay, (int(region.centroid_x), int(region.centroid_y)), 4, (255, 255, 255), -1)
        cv2.putText(overlay, f"fire {region.pixel_area}px", (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return overlay
