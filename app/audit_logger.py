import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


LOG_DIR = os.getenv("LOG_DIR", "./logs")


def ensure_log_dir() -> Path:
    path = Path(LOG_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_audit_log(record: dict[str, Any]) -> str:
    log_dir = ensure_log_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = log_dir / f"audit_{timestamp}.json"

    payload = {
        "logged_at": datetime.now().isoformat(),
        **record,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(file_path)