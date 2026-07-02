from __future__ import annotations

import os
from pathlib import Path


def proje_koku() -> Path:
    return Path(__file__).resolve().parents[1]


def env_yukle(env_path: str | os.PathLike | None = None) -> None:
    """Basit .env yukleyici. Var olan ortam degiskenlerini ezmez."""
    path = Path(env_path) if env_path else proje_koku() / ".env"
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def postgis_ayarlarini_al() -> dict:
    env_yukle()
    return {
        "host": os.getenv("POSTGIS_HOST", "localhost"),
        "port": int(os.getenv("POSTGIS_PORT", "5432")),
        "dbname": os.getenv("POSTGIS_DB", "fire_mapping"),
        "user": os.getenv("POSTGIS_USER", "postgres"),
        "password": os.getenv("POSTGIS_PASSWORD", "postgres"),
    }
