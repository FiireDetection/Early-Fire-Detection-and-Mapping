from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from haritalama.ortam import postgis_ayarlarini_al
except ImportError:
    from ortam import postgis_ayarlarini_al


PROBABILITY_COLUMNS = ("entegre_olasilik", "riskli_olasilik", "decision_prob", "prob_fire", "fire_probability")
LATITUDE_COLUMNS = ("latitude", "lat", "drone_latitude", "enlem")
LONGITUDE_COLUMNS = ("longitude", "lon", "lng", "drone_longitude", "boylam")


def _first_existing(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _prediction(probability: float, threshold: float) -> str:
    return "fire" if float(probability) >= float(threshold) else "no_fire"


def _connect():
    try:
        import psycopg2
    except ImportError as exc:
        raise SystemExit("psycopg2-binary kurulu degil. Once `pip install -r requirements.txt` calistirin.") from exc
    return psycopg2.connect(**postgis_ayarlarini_al())


def tabloyu_hazirla(conn) -> None:
    sql_path = Path(__file__).resolve().parent / "postgis_tablolari.sql"
    with conn.cursor() as cur:
        cur.execute(sql_path.read_text(encoding="utf-8"))
    conn.commit()


def csv_postgis_yaz(
    csv_path: str | Path,
    threshold: float = 0.5,
    default_latitude: float | None = None,
    default_longitude: float | None = None,
    default_altitude: float | None = None,
    clear_table: bool = False,
) -> int:
    df = pd.read_csv(csv_path)
    prob_col = _first_existing(df, PROBABILITY_COLUMNS)
    if not prob_col:
        raise ValueError(f"Olasilik kolonu bulunamadi. Beklenen kolonlardan biri: {PROBABILITY_COLUMNS}")

    lat_col = _first_existing(df, LATITUDE_COLUMNS)
    lon_col = _first_existing(df, LONGITUDE_COLUMNS)
    if (lat_col is None or lon_col is None) and (default_latitude is None or default_longitude is None):
        raise ValueError("CSV icinde konum kolonu yok. --latitude ve --longitude parametrelerini verin.")

    frame_col = "frame_idx" if "frame_idx" in df.columns else None
    time_col = "saniye" if "saniye" in df.columns else ("video_time_s" if "video_time_s" in df.columns else None)
    altitude_col = "altitude_m" if "altitude_m" in df.columns else None

    conn = _connect()
    try:
        tabloyu_hazirla(conn)
        with conn.cursor() as cur:
            if clear_table:
                cur.execute("TRUNCATE TABLE drone_frame_points RESTART IDENTITY;")

            written = 0
            for _, row in df.iterrows():
                probability = float(pd.to_numeric(row.get(prob_col), errors="coerce"))
                if probability != probability:
                    continue

                lat = row.get(lat_col) if lat_col else default_latitude
                lon = row.get(lon_col) if lon_col else default_longitude
                lat = float(pd.to_numeric(lat, errors="coerce"))
                lon = float(pd.to_numeric(lon, errors="coerce"))
                if lat != lat or lon != lon:
                    continue

                frame_idx = int(row.get(frame_col)) if frame_col and pd.notna(row.get(frame_col)) else None
                video_time_s = float(row.get(time_col)) if time_col and pd.notna(row.get(time_col)) else None
                altitude = (
                    float(row.get(altitude_col))
                    if altitude_col and pd.notna(row.get(altitude_col))
                    else default_altitude
                )

                cur.execute(
                    """
                    INSERT INTO drone_frame_points (
                        frame_idx, video_time_s, prediction, fire_probability,
                        latitude, longitude, altitude_m, location_source, geom
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326));
                    """,
                    (
                        frame_idx,
                        video_time_s,
                        _prediction(probability, threshold),
                        probability,
                        lat,
                        lon,
                        altitude,
                        "csv",
                        lon,
                        lat,
                    ),
                )
                written += 1
        conn.commit()
        return written
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Video tahmin CSV'sini PostGIS drone_frame_points tablosuna yazar.")
    parser.add_argument("--csv", required=True, help="Tahmin CSV yolu")
    parser.add_argument("--threshold", type=float, default=0.5, help="Yangin karar esigi")
    parser.add_argument("--latitude", type=float, default=None, help="CSV'de konum yoksa kullanilacak sabit enlem")
    parser.add_argument("--longitude", type=float, default=None, help="CSV'de konum yoksa kullanilacak sabit boylam")
    parser.add_argument("--altitude", type=float, default=None, help="Opsiyonel sabit yukseklik")
    parser.add_argument("--clear-table", action="store_true", help="Yazmadan once drone_frame_points tablosunu temizle")
    args = parser.parse_args()

    count = csv_postgis_yaz(
        csv_path=args.csv,
        threshold=args.threshold,
        default_latitude=args.latitude,
        default_longitude=args.longitude,
        default_altitude=args.altitude,
        clear_table=args.clear_table,
    )
    print(f"PostGIS'e yazilan satir sayisi: {count}")


if __name__ == "__main__":
    main()
