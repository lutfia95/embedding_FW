from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .errors import ValidationError

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        if not _SAFE_NAME.fullmatch(job_id):
            raise ValidationError("Unsafe job identifier")
        path = self.root / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path(self, job_id: str, name: str) -> Path:
        if not _SAFE_NAME.fullmatch(name):
            raise ValidationError("Unsafe artifact name")
        return self.job_dir(job_id) / name

    def save_bytes(self, job_id: str, name: str, data: bytes) -> dict:
        target = self.path(job_id, name)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return {
            "name": name,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "media_type": _media_type(name),
        }

    def save_json(self, job_id: str, name: str, value: Any) -> dict:
        data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        return self.save_bytes(job_id, name, data)

    def save_text(self, job_id: str, name: str, value: str) -> dict:
        return self.save_bytes(job_id, name, value.encode("utf-8"))

    def list(self, job_id: str) -> list[dict]:
        result = []
        for path in sorted(self.job_dir(job_id).iterdir()):
            if path.is_file():
                data = path.read_bytes()
                result.append(
                    {
                        "name": path.name,
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "media_type": _media_type(path.name),
                    }
                )
        return result


def _media_type(name: str) -> str:
    extension = Path(name).suffix.lower()
    return {
        ".json": "application/json",
        ".html": "text/html",
        ".svg": "image/svg+xml",
        ".csv": "text/csv",
        ".npz": "application/octet-stream",
    }.get(extension, "application/octet-stream")
