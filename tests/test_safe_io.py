"""Tests for crash-safe JSON persistence (common/safe_io.py)."""
import json
import multiprocessing as mp
import os

from common.safe_io import atomic_write_json, locked_update_json, read_json


def test_atomic_write_roundtrip_and_no_tmp_left(tmp_path):
    path = str(tmp_path / "state.json")
    atomic_write_json(path, {"a": [1, 2, 3], "中文": "ok"})
    assert read_json(path) == {"a": [1, 2, 3], "中文": "ok"}
    assert [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] == []


def test_read_missing_returns_default(tmp_path):
    assert read_json(str(tmp_path / "missing.json"), default=[]) == []


def _increment(path, n):
    for _ in range(n):
        locked_update_json(path, lambda d: {"count": (d or {"count": 0})["count"] + 1}, default=None)


def test_locked_update_is_race_free(tmp_path):
    path = str(tmp_path / "counter.json")
    procs = [mp.get_context("fork").Process(target=_increment, args=(path, 25)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    with open(path) as f:
        assert json.load(f)["count"] == 100
