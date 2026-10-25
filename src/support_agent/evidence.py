import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .results import json_text


class Evidence:
    """One fsynced JSON file per event; no credentials or provider exceptions recorded."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)

    def write(self, context, event: str, **payload):
        record = {
            "timestamp": datetime.now(timezone.utc),
            "event": event,
            "session_id": context.session_id,
            "interaction_id": context.interaction_id,
            "authenticated_customer_id": context.authenticated_customer_id,
            **payload,
        }
        name = f"{uuid4()}.json"
        path = self.root / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json_text(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return name
