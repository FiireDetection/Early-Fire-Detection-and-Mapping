"""
Egitilmis dual-branch modeli ile RGB/termal video uzerinde yangin tahmini yapar.

Bu dosya bilerek model_egitimi disinda tutulur: model_egitimi yalnizca egitim
kodlarini, video_cikarim ise egitilmis modeli kullanarak tahmin akisini icerir.

Ornek:
  python video_cikarim/video_tahmin.py --rgb-video data/rgb.mp4 --termal-video data/thermal.mp4
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_EGITIMI_ROOT = PROJECT_ROOT / "model_egitimi"
if str(MODEL_EGITIMI_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_EGITIMI_ROOT))

from config import CKPT_DUAL_BRANCH, OUTPUTS_DIR  # noqa: E402
from src.inference.alarm import AlarmConfig, AlarmStateMachine  # noqa: E402
from src.models import make_classifier  # noqa: E402


def _load_checkpoint(checkpoint_path: str | Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state = checkpoint["state"]
    model_family = str(checkpoint.get("model_family") or checkpoint.get("arch") or "dual_branch_gated_fusion")
    backbone = str(checkpoint.get("backbone", "resnet50"))
    training_args = checkpoint.get("training_args") or {}
    thermal_init = str(training_args.get("thermal_init", "mean_rgb")) if isinstance(training_args, dict) else "mean_rgb"

    model = make_classifier(
        model_family=model_family,
        backbone=backbone,
        mode="fusion",
        num_classes=2,
        pretrained=False,
        thermal_init=thermal_init,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    threshold = float(checkpoint.get("threshold", 0.5))
    temperature = float(checkpoint.get("temperature", 1.0))
    return model, device, threshold, temperature


def _prepare_rgb(frame: np.ndarray | None, size: int) -> np.ndarray:
    if frame is None:
        return np.zeros((3, size, size), dtype=np.float32)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    rgb = rgb.astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)


def _prepare_thermal(frame: np.ndarray | None, size: int) -> np.ndarray:
    if frame is None:
        return np.zeros((1, size, size), dtype=np.float32)
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    lo = float(np.percentile(gray, 2.0)) if gray.size else 0.0
    hi = float(np.percentile(gray, 98.0)) if gray.size else 1.0
    if hi - lo < 1e-6:
        norm = np.zeros_like(gray, dtype=np.float32)
    else:
        norm = np.clip((gray - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return norm[None, ...]


def _read_frame(cap: cv2.VideoCapture | None):
    if cap is None:
        return True, None
    return cap.read()


def _open_video(path: str | None, label: str):
    if not path:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"{label} video acilamadi: {path}")
    return cap


def video_tahmini_yap(
    rgb_video: str | None,
    termal_video: str | None,
    checkpoint: str | Path,
    cikti_csv: str | Path,
    boyut: int = 224,
    kare_adimi: int = 10,
) -> Path:
    model, device, threshold, temperature = _load_checkpoint(checkpoint)
    alarm = AlarmStateMachine(
        AlarmConfig(
            high_threshold=max(0.5, threshold),
            low_threshold=max(0.2, threshold * 0.65),
            confirm_frames=3,
            cooldown_frames=6,
        )
    )

    rgb_cap = _open_video(rgb_video, "RGB")
    thermal_cap = _open_video(termal_video, "Termal")

    try:
        rgb_total = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if rgb_cap else 0
        thermal_total = int(thermal_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if thermal_cap else 0
        if rgb_total and thermal_total:
            toplam_kare = min(rgb_total, thermal_total)
        else:
            toplam_kare = max(rgb_total, thermal_total)

        fps_source = rgb_cap or thermal_cap
        fps = float(fps_source.get(cv2.CAP_PROP_FPS)) if fps_source else 30.0
        if fps <= 0:
            fps = 30.0

        cikti_csv = Path(cikti_csv)
        cikti_csv.parent.mkdir(parents=True, exist_ok=True)

        with cikti_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_idx",
                    "saniye",
                    "prob_fire",
                    "decision_prob",
                    "threshold",
                    "pred_fire",
                    "alarm_state",
                    "alarm_event",
                    "alarm_reason",
                ],
            )
            writer.writeheader()

            frame_idx = 0
            while True:
                ok_rgb, rgb_frame = _read_frame(rgb_cap)
                ok_th, thermal_frame = _read_frame(thermal_cap)
                if not ok_rgb or not ok_th:
                    break
                if toplam_kare and frame_idx >= toplam_kare:
                    break

                if frame_idx % max(1, int(kare_adimi)) != 0:
                    frame_idx += 1
                    continue

                rgb_arr = _prepare_rgb(rgb_frame, boyut)
                thermal_arr = _prepare_thermal(thermal_frame, boyut)
                x_np = np.concatenate([rgb_arr, thermal_arr], axis=0)
                x = torch.from_numpy(x_np).unsqueeze(0).float().to(device)

                with torch.no_grad():
                    logits = model(x)
                    logits = logits / max(1e-6, float(temperature))
                    prob_fire = float(torch.softmax(logits, dim=1)[0, 1].detach().cpu().item())

                alarm_state, alarm_event, decision_prob, alarm_reason = alarm.update(decision_prob=prob_fire)
                writer.writerow(
                    {
                        "frame_idx": frame_idx,
                        "saniye": round(frame_idx / fps, 3),
                        "prob_fire": round(prob_fire, 6),
                        "decision_prob": round(decision_prob, 6),
                        "threshold": round(threshold, 6),
                        "pred_fire": int(prob_fire >= threshold),
                        "alarm_state": alarm_state,
                        "alarm_event": int(alarm_event),
                        "alarm_reason": alarm_reason,
                    }
                )

                frame_idx += 1

        return cikti_csv
    finally:
        if rgb_cap is not None:
            rgb_cap.release()
        if thermal_cap is not None:
            thermal_cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="RGB/termal video uzerinde yangin tahmini yapar.")
    parser.add_argument("--rgb-video", default=None, help="RGB video yolu")
    parser.add_argument("--termal-video", default=None, help="Termal video yolu")
    parser.add_argument("--model", default=str(CKPT_DUAL_BRANCH), help="Model checkpoint yolu")
    parser.add_argument("--cikti", default=str(OUTPUTS_DIR / "video_tahminleri.csv"), help="Cikti CSV yolu")
    parser.add_argument("--boyut", type=int, default=224, help="Model giris boyutu")
    parser.add_argument("--kare-adimi", type=int, default=10, help="Kac karede bir tahmin yapilacak")
    args = parser.parse_args()

    if not args.rgb_video and not args.termal_video:
        raise SystemExit("En az bir video verilmeli: --rgb-video veya --termal-video")

    out = video_tahmini_yap(
        rgb_video=args.rgb_video,
        termal_video=args.termal_video,
        checkpoint=args.model,
        cikti_csv=args.cikti,
        boyut=args.boyut,
        kare_adimi=args.kare_adimi,
    )
    print("Video tahmin CSV yazildi:", out)


if __name__ == "__main__":
    main()
