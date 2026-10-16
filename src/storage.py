"""Content addressed, atomic checkpoints and bounded transient retries."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def retry_network(fn, attempts=3, sleep=time.sleep):
    """At most three total attempts; deterministic HTTP failures are not retried."""
    for attempt in range(attempts):
        try:
            return fn(), attempt
        except HTTPError as e:
            if e.code not in (408, 429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt == attempts - 1:
                raise
        sleep(2 ** attempt)


def cache_key(*, model_id, revision, processor_revision, image_sha256,
              prompt, options, processor_config, code_sha256):
    if len(revision) != 40 or len(processor_revision) != 40:
        raise ValueError("Model and processor require immutable commit revisions")
    return digest(locals())


def read_checkpoint(path, key):
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    if row["run_key"] != key or row["payload_sha256"] != digest(row["payload"]):
        raise ValueError("Checkpoint hash mismatch")
    return row["payload"]


def checkpoint(path, key, payload):
    atomic_json(path, {"run_key": key, "payload": payload, "payload_sha256": digest(payload)})
