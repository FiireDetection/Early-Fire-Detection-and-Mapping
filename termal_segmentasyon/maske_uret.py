from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

try:
    from termal_segmentasyon.termal_maske import bolgeleri_bul, maske_olustur, maske_ustune_ciz
except ImportError:
    from termal_maske import bolgeleri_bul, maske_olustur, maske_ustune_ciz


def _ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "masks": root / "masks",
        "overlays": root / "overlays",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _process_frame(frame, frame_idx: int, dirs: dict[str, Path], args, writer) -> int:
    mask, norm = maske_olustur(
        frame,
        threshold_mode=args.threshold_mode,
        threshold_value=args.threshold_value,
        percentile=args.percentile,
        blur_kernel=args.blur_kernel,
        morph_kernel=args.morph_kernel,
    )
    regions = bolgeleri_bul(mask, norm, min_area=args.min_area)
    mask_path = dirs["masks"] / f"frame_{frame_idx:06d}_mask.png"
    overlay_path = dirs["overlays"] / f"frame_{frame_idx:06d}_overlay.png"
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(overlay_path), maske_ustune_ciz(frame, mask, regions))

    for region in regions:
        writer.writerow({
            "frame_idx": frame_idx,
            "region_id": region.region_id,
            "pixel_area": region.pixel_area,
            "bbox_x": region.bbox_x,
            "bbox_y": region.bbox_y,
            "bbox_w": region.bbox_w,
            "bbox_h": region.bbox_h,
            "centroid_x": round(region.centroid_x, 3),
            "centroid_y": round(region.centroid_y, 3),
            "mean_intensity": round(region.mean_intensity, 6),
            "max_intensity": round(region.max_intensity, 6),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
        })
    return len(regions)


def termal_video_maske_uret(video_path: str | Path, output_dir: str | Path, args) -> tuple[int, int]:
    output_dir = Path(output_dir)
    dirs = _ensure_dirs(output_dir)
    regions_csv = output_dir / "termal_bolgeler.csv"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Termal video acilamadi: {video_path}")

    processed = 0
    total_regions = 0
    try:
        with regions_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "frame_idx",
                "region_id",
                "pixel_area",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "centroid_x",
                "centroid_y",
                "mean_intensity",
                "max_intensity",
                "mask_path",
                "overlay_path",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if args.max_frames is not None and processed >= int(args.max_frames):
                    break
                if frame_idx % max(1, int(args.frame_step)) == 0:
                    total_regions += _process_frame(frame, frame_idx, dirs, args, writer)
                    processed += 1
                frame_idx += 1
    finally:
        cap.release()
    return processed, total_regions


def termal_gorsel_maske_uret(image_path: str | Path, output_dir: str | Path, args) -> tuple[int, int]:
    frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise RuntimeError(f"Termal gorsel okunamadi: {image_path}")
    output_dir = Path(output_dir)
    dirs = _ensure_dirs(output_dir)
    regions_csv = output_dir / "termal_bolgeler.csv"
    with regions_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame_idx",
            "region_id",
            "pixel_area",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "centroid_x",
            "centroid_y",
            "mean_intensity",
            "max_intensity",
            "mask_path",
            "overlay_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        count = _process_frame(frame, 0, dirs, args, writer)
    return 1, count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Termal video veya gorselden yangin maskesi uretir.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--termal-video", help="Termal video yolu")
    source.add_argument("--termal-gorsel", help="Tek termal gorsel yolu")
    parser.add_argument("--cikti-klasoru", default="outputs/termal_segmentasyon", help="Maske/overlay cikti klasoru")
    parser.add_argument("--frame-step", type=int, default=10, help="Videoda kac karede bir islenecek")
    parser.add_argument("--max-frames", type=int, default=None, help="Maksimum islenecek kare sayisi")
    parser.add_argument("--threshold-mode", choices=["fixed", "percentile", "hybrid"], default="hybrid")
    parser.add_argument("--threshold-value", type=float, default=180.0)
    parser.add_argument("--percentile", type=float, default=97.0)
    parser.add_argument("--min-area", type=int, default=130)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--morph-kernel", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.termal_video:
        processed, regions = termal_video_maske_uret(args.termal_video, args.cikti_klasoru, args)
    else:
        processed, regions = termal_gorsel_maske_uret(args.termal_gorsel, args.cikti_klasoru, args)
    print(f"Islenen kare: {processed}")
    print(f"Bulunan bolge: {regions}")
    print(f"Cikti klasoru: {Path(args.cikti_klasoru)}")


if __name__ == "__main__":
    main()
