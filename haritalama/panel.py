from __future__ import annotations

import json

from flask import Flask, jsonify, render_template

try:
    from haritalama.ortam import postgis_ayarlarini_al
except ImportError:
    from ortam import postgis_ayarlarini_al


app = Flask(__name__)


def _connect():
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError("psycopg2-binary kurulu degil. Once `pip install -r requirements.txt` calistirin.") from exc
    return psycopg2.connect(**postgis_ayarlarini_al())


@app.route("/")
def index():
    return render_template("index.html")


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

    return jsonify({
        "status": "success",
        "total": int(total or 0),
        "fire_count": int(fire_count or 0),
        "max_probability": float(max_prob or 0.0),
        "bounds": [[min_lat, min_lon], [max_lat, max_lon]] if min_lat is not None else None,
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
