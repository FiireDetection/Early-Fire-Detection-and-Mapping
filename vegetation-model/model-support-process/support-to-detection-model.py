from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import pandas as pd


sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "outputs" / "video_predictions_scored.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "ilk45sn_duzeltilmis.csv"

# Logistic regression coefficients printed by vegetation-model/train_model.py.
INTERCEPT = -0.5907
NDVI_COEF = -1.3428
NDMI_COEF = -1.3042
LC_COEFS = {
    0: -2.7629,
    1: 2.0896,
    2: -0.0382,
    3: -0.3532,
    4: 1.1267,
    5: 1.6451,
    6: -0.1594,
    7: -0.2806,
    8: -1.2702,
}


def sigmoid(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def hesapla_yanicilik(ndvi: float, ndmi: float, land_cover: int) -> float:
    if int(land_cover) not in LC_COEFS:
        raise ValueError(f"land_cover {land_cover} desteklenmiyor. Gecerli degerler: {sorted(LC_COEFS)}")
    ham_skor = INTERCEPT + (NDVI_COEF * float(ndvi)) + (NDMI_COEF * float(ndmi)) + LC_COEFS[int(land_cover)]
    return sigmoid(ham_skor)


def entegre_skor(kamera: float, bitki: float, max_etki: float) -> float:
    # Bitki 0.50 ise notr; 1.00 ise +max_etki, 0.00 ise -max_etki.
    modifiye = ((float(bitki) - 0.50) / 0.50) * float(max_etki)
    final = float(kamera) + modifiye
    return max(0.0, min(1.0, final))


def sec_olasilik_kolonu(df: pd.DataFrame) -> str:
    for col in ("decision_prob", "prob_fire", "fire_probability"):
        if col in df.columns:
            return col
    raise ValueError("CSV icinde decision_prob, prob_fire veya fire_probability kolonu bulunamadi.")


def risk_skoru_uygula(
    input_csv: str | os.PathLike,
    output_csv: str | os.PathLike,
    ndvi: float,
    ndmi: float,
    land_cover: int,
    max_etki: float = 0.05,
    esik: float = 0.65,
    max_frame: int | None = 1350,
) -> Path:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    df = pd.read_csv(input_csv)
    if "frame_idx" not in df.columns:
        raise ValueError("CSV icinde frame_idx kolonu bulunamadi.")

    prob_col = sec_olasilik_kolonu(df)
    calisma_df = df.copy()
    if max_frame is not None:
        calisma_df = calisma_df[calisma_df["frame_idx"] <= int(max_frame)].copy()

    yaniclik = hesapla_yanicilik(ndvi=ndvi, ndmi=ndmi, land_cover=land_cover)
    modifiye_deger = ((yaniclik - 0.50) / 0.50) * float(max_etki)

    sonuclar = []
    yangin_onceki = 0
    yangin_sonraki = 0
    karar_degisen = 0
    degisen_listesi = []

    print("=" * 90)
    print("  KAMERA TAHMINI + BITKI YANICILIK SKORU ENTEGRASYONU")
    print("=" * 90)
    print(f"  Girdi CSV: {input_csv}")
    print(f"  Bitki verileri: NDVI={ndvi}, NDMI={ndmi}, land_cover={land_cover}")
    print(f"  Bitki yanicilik skoru: %{yaniclik * 100:.1f}")
    print("  Referans: %50 = notr")
    print(f"  Her frame etkisi: {modifiye_deger * 100:+.2f} yuzde puan")
    if max_frame is not None:
        print(f"  Islenen aralik: frame_idx <= {max_frame}")
    else:
        print("  Islenen aralik: tum frame'ler")
    print("=" * 90)
    print()
    print("{:>6} | {:>10} {:>8} | {:>10} {:>8} | {:>7} {:>8}".format(
        "Frame", "ONCEKI", "Karar", "SONRAKI", "Karar", "Fark", "Degisim"
    ))
    print("-" * 90)

    for _, row in calisma_df.iterrows():
        frame = int(row["frame_idx"])
        onceki = float(row[prob_col])
        sonraki = entegre_skor(onceki, yaniclik, max_etki=max_etki)
        fark = (sonraki - onceki) * 100

        karar_onceki = "YANGIN" if onceki >= esik else "YOK"
        karar_sonraki = "YANGIN" if sonraki >= esik else "YOK"

        if karar_onceki == "YANGIN":
            yangin_onceki += 1
        if karar_sonraki == "YANGIN":
            yangin_sonraki += 1

        if karar_onceki != karar_sonraki:
            degisim = "DEGISTI"
            karar_degisen += 1
            degisen_listesi.append((frame, onceki * 100, sonraki * 100, karar_onceki, karar_sonraki))
        else:
            degisim = "-"

        print("{:>6} | {:>9.1f}% {:>8} | {:>9.1f}% {:>8} | {:>+6.2f}% {:>8}".format(
            frame,
            onceki * 100,
            karar_onceki,
            sonraki * 100,
            karar_sonraki,
            fark,
            degisim,
        ))

        sonuclar.append({
            "frame_idx": frame,
            "saniye": round(frame / 30, 1),
            "kamera_olasiligi": round(onceki, 6),
            "kamera_karari": karar_onceki,
            "bitki_yanicilik_skoru": round(yaniclik, 6),
            "bitki_etki_puani": round(modifiye_deger, 6),
            "entegre_olasilik": round(sonraki, 6),
            "entegre_karar": karar_sonraki,
            "fark_puan": round(fark, 2),
            "karar_degisti": karar_onceki != karar_sonraki,
        })

    print()
    print("=" * 90)
    print("  OZET")
    print("=" * 90)
    print(f"  Toplam frame:              {len(calisma_df)}")
    print(f"  ONCEKI kamera yangin:      {yangin_onceki} frame")
    print(f"  SONRAKI entegre yangin:    {yangin_sonraki} frame")
    print(f"  Karari degisen frame:      {karar_degisen}")
    print(f"  Bitki yanicilik:           %{yaniclik * 100:.1f}")
    print(f"  Her frame'e eklenen:       {modifiye_deger * 100:+.2f} yuzde puan")

    if degisen_listesi:
        print("\n  KARAR DEGISEN FRAME'LER:")
        for frame, onceki_yuzde, sonraki_yuzde, karar_onceki, karar_sonraki in degisen_listesi:
            print(f"    Frame {frame}: {onceki_yuzde:.1f}% ({karar_onceki}) -> {sonraki_yuzde:.1f}% ({karar_sonraki})")

    print("=" * 90)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sonuclar).to_csv(output_csv, index=False)
    print(f"\n  CSV kaydedildi: {output_csv}")
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video tahminlerine bitki yanicilik skoru ekler.")
    parser.add_argument("--tahmin-csv", default=str(DEFAULT_INPUT), help="Video tahmin CSV yolu")
    parser.add_argument("--cikti", default=str(DEFAULT_OUTPUT), help="Cikti CSV yolu")
    parser.add_argument("--ndvi", type=float, default=0.4062, help="Bolgenin NDVI degeri")
    parser.add_argument("--ndmi", type=float, default=-0.1605, help="Bolgenin NDMI degeri")
    parser.add_argument("--land-cover", type=int, default=5, help="Dynamic World land cover sinifi")
    parser.add_argument("--max-etki", type=float, default=0.05, help="Bitki skorunun maksimum etkisi")
    parser.add_argument("--esik", type=float, default=0.65, help="Yangin karar esigi")
    parser.add_argument(
        "--max-frame",
        type=int,
        default=1350,
        help="Islenecek maksimum frame_idx. Tum frame'ler icin -1 kullan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_frame = None if int(args.max_frame) < 0 else int(args.max_frame)
    risk_skoru_uygula(
        input_csv=args.tahmin_csv,
        output_csv=args.cikti,
        ndvi=args.ndvi,
        ndmi=args.ndmi,
        land_cover=args.land_cover,
        max_etki=args.max_etki,
        esik=args.esik,
        max_frame=max_frame,
    )


if __name__ == "__main__":
    main()
