# -*- coding: utf-8 -*-
# Copyright 2026 Yiming. Licensed under the Apache License, Version 2.0.
"""
Crash-safe / concurrency-safe JSON persistence.

Why this exists
---------------
state / pending / policy-stage JSON files are read-modify-written by several
processes (policy rollout, env rollout, rollout workers of different nodes that
share one filesystem, a re-launched job after a crash ...). The old code used a
plain `open(path, "w"); json.dump(...)`, which has two problems:

1. Torn writes: if the process is killed while dumping, the file is left
   truncated and the next epoch fails with JSONDecodeError -> cannot resume.
2. Write races: two writers that open the same path with "w" interleave bytes,
   and a fixed tmp name (`path + ".tmp"`) makes two atomic writers clobber
   each other's tmp file.

Fix
---
- `atomic_write_json`: write to a unique tmp file (pid + random suffix) in the
  same directory, fsync, then `os.replace` (atomic on POSIX).
- `file_lock`: advisory `fcntl.flock` on a sidecar `.lock` file so that a
  read-modify-write sequence (`locked_update_json`) is serialized.
"""

import os
import json
import time
import uuid
import errno
import contextlib
from typing import Any, Callable

try:
    import fcntl  # POSIX only
    _HAS_FCNTL = True
except Exception:  # pragma: no cover - windows fallback
    _HAS_FCNTL = False


@contextlib.contextmanager
def file_lock(path: str, timeout_s: float = 600.0, poll_s: float = 0.05):
    """Exclusive advisory lock on `<path>.lock`."""
    lock_path = path + ".lock"
    d = os.path.dirname(lock_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        if _HAS_FCNTL:
            t0 = time.time()
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as e:
                    if e.errno not in (errno.EAGAIN, errno.EACCES):
                        raise
                    if time.time() - t0 > timeout_s:
                        raise TimeoutError(f"[safe_io] timeout acquiring lock: {lock_path}")
                    time.sleep(poll_s)
        yield
    finally:
        try:
            if _HAS_FCNTL:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def atomic_write_json(path: str, obj: Any, indent: Any = 2) -> None:
    """Write JSON atomically (unique tmp file + fsync + os.replace)."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def locked_write_json(path: str, obj: Any, indent: Any = 2) -> None:
    with file_lock(path):
        atomic_write_json(path, obj, indent=indent)


def locked_read_json(path: str, default: Any = None) -> Any:
    with file_lock(path):
        return read_json(path, default)


def locked_update_json(path: str, fn: Callable[[Any], Any], default: Any = None, indent: Any = 2) -> Any:
    """Serialized read-modify-write: new = fn(old); returns new."""
    with file_lock(path):
        old = read_json(path, default)
        new = fn(old)
        atomic_write_json(path, new, indent=indent)
        return new
