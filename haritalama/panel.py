from __future__ import annotations

import os

from flask import Flask, jsonify, render_template

try:
    from haritalama.ortam import postgis_ayarlarini_al
except ImportError:
    from ortam import postgis_ayarlarini_al


app = Flask(__name__)


GEOSERVER_URL = os.getenv("GEOSERVER_URL", "http://localhost:8080/geoserver").rstrip("/")
GEOSERVER_WORKSPACE = os.getenv("GEOSERVER_WORKSPACE", "fire_mapping")
GEOSERVER_WMS_URL = os.getenv("GEOSERVER_WMS_URL", f"{GEOSERVER_URL}/{GEOSERVER_WORKSPACE}/wms")
LAYER_ACTIVE_FIRE = os.getenv("GEOSERVER_LAYER_ACTIVE_FIRE", f"{GEOSERVER_WORKSPACE}:fire_heatmap_points")
LAYER_DRONE_PATH = os.getenv("GEOSERVER_LAYER_DRONE_PATH", f"{GEOSERVER_WORKSPACE}:drone_flight_path")
LAYER_FIRE_HEATMAP = os.getenv("GEOSERVER_LAYER_FIRE_HEATMAP", f"{GEOSERVER_WORKSPACE}:fire_heatmap_points")


def _connect():
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError("psycopg2-binary kurulu degil. Once `pip install -r requirements.txt` calistirin.") from exc
    return psycopg2.connect(**postgis_ayarlarini_al())


@app.route("/")
def index():
    return render_template(
        "index.html",
        geoserver_wms_url=GEOSERVER_WMS_URL,
        layer_active_fire=LAYER_ACTIVE_FIRE,
        layer_drone_path=LAYER_DRONE_PATH,
        layer_fire_heatmap=LAYER_FIRE_HEATMAP,
    )


@app.route("/api/points")
def points():
    features = []
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT frame_idx, video_time_s, prediction, fire_probability, latitude, longitude
                    FROM drone_frame_points
                    WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                    ORDER BY video_time_s NULLS LAST, frame_idx NULLS LAST, id ASC;
                    """
                )
                for frame_idx, video_time_s, prediction, probability, lat, lon in cur.fetchall():
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                        "properties": {
                            "frame_idx": frame_idx,
                            "video_time_s": video_time_s,
                            "prediction": prediction,
                            "fire_probability": float(probability or 0.0),
                        },
                    })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc), "features": []}), 500
    return jsonify({"type": "FeatureCollection", "features": features})


@app.route("/api/fire-points")
def fire_points():
    features = []
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT frame_idx, video_time_s, prediction, fire_probability, latitude, longitude
                    FROM drone_frame_points
                    WHERE latitude IS NOT NULL
                      AND longitude IS NOT NULL
                      AND COALESCE(fire_probability, 0) >= 0.5
                    ORDER BY fire_probability DESC, video_time_s NULLS LAST, frame_idx NULLS LAST;
                    """
                )
                for frame_idx, video_time_s, prediction, probability, lat, lon in cur.fetchall():
                    probability = float(probability or 0.0)
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                        "properties": {
                            "frame_idx": frame_idx,
                            "prediction": prediction,
                            "fire_probability": probability,
                            "yangin_yuzdesi": round(probability * 100, 2),
                            "zaman": f"{round(float(video_time_s or 0.0), 2)} sn",
                            "image_url": "",
                            "resim_adi": "",
                        },
                    })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc), "features": []}), 500
    return jsonify({"type": "FeatureCollection", "features": features})


@app.route("/api/stats")
def stats():
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE prediction = 'fire') AS fire_count,
                        COALESCE(MAX(fire_probability), 0) AS max_probability,
                        MIN(latitude), MAX(latitude), MIN(longitude), MAX(longitude)
                    FROM drone_frame_points;
                    """
                )
                total, fire_count, max_prob, min_lat, max_lat, min_lon, max_lon = cur.fetchone()
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

    if not total or min_lat is None or min_lon is None:
        return jsonify({
            "status": "empty",
            "center": [39.0, 35.0],
            "stats": {"toplam": 0, "yangin": 0, "yangin_degil": 0, "max_yuzde": 0},
        })

    return jsonify({
        "status": "success",
        "total": int(total or 0),
        "fire_count": int(fire_count or 0),
        "max_probability": float(max_prob or 0.0),
        "bounds": [[min_lat, min_lon], [max_lat, max_lon]] if min_lat is not None else None,
        "center": [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0],
        "stats": {
            "toplam": int(total or 0),
            "yangin": int(fire_count or 0),
            "yangin_degil": int((total or 0) - (fire_count or 0)),
            "max_yuzde": round(float(max_prob or 0.0) * 100, 2),
        },
    })


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PostGIS yangin harita panelini baslatir.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
