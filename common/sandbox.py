# -*- coding: utf-8 -*-
# Copyright 2026 Yiming. Licensed under the Apache License, Version 2.0.
"""
Process-isolated code sandbox for GT / synthetic unit tests.

- Every (code, test_input) pair is executed in its own `multiprocessing.Process`
  (fork context on Linux, so no interpreter re-spawn cost). The child redirects
  sys.stdin / sys.stdout (and overrides `input()`) and returns captured stdout
  through a Queue.
- Chunked scheduling: at most `max_procs` (default 128, `execute.num_chunk`)
  processes are alive at the same time.
- Two time limits:
    * `time_limit` (per test, from the dataset, e.g. 1~4s): used for judging.
      When exceeded the result is "Timeout Error" and the child gets SIGTERM.
    * `hard_timeout_s` (default 30s): absolute upper bound of a child's life.
      Anything still alive (ignored SIGTERM, stuck in C code, zombie feeder
      thread ...) is SIGKILLed, so one bad process can never stall a chunk.
- Queues are drained while children are running. (A child that puts a large
  string into a Queue can't exit until the parent reads it; the old code only
  read after `is_alive()` became False, which turned big-output programs into
  false "Timeout Error"s.)
- Best-effort hardening of every child (see `_harden_child`): it runs inside a
  private scratch directory, gets an address-space budget (RLIMIT_AS, relative
  to the forked parent's footprint), a max file size (RLIMIT_FSIZE) and no core
  dumps, and the most destructive os / shutil / subprocess calls are disabled
  while the untrusted code runs.

  This is NOT a security boundary: model-generated code still runs as the
  current user with network access. Run training inside a container / VM.
"""

import io
import os
import sys
import time
import shutil
import tempfile
import subprocess
import multiprocessing as mp
from queue import Empty
from typing import Any, Dict, List, Sequence, Tuple

DEFAULT_MAX_PROCS = 128
DEFAULT_HARD_TIMEOUT_S = 30.0
TIMEOUT_RESULT = "Timeout Error"
INVALID_TEST = "Invalid Test"


# Extra address space a child may allocate on top of what it inherited from the
# parent at fork time (MB). 0 disables the limit.
MEM_LIMIT_MB = int(os.environ.get("VICO_SANDBOX_MEM_MB", "2048"))
# Largest file a child may write (MB). 0 disables the limit.
FSIZE_LIMIT_MB = int(os.environ.get("VICO_SANDBOX_FSIZE_MB", "64"))

# Functions that model-generated code has no business calling while being judged.
_GUARDED_CALLS = [
    (os, ("system", "kill", "killpg", "fork", "forkpty", "putenv", "remove", "unlink",
          "rmdir", "removedirs", "rename", "renames", "replace", "truncate", "chmod",
          "chown", "lchown", "setuid", "setgid")),
    (shutil, ("rmtree", "move", "chown")),
    (subprocess, ("Popen", "run", "call", "check_call", "check_output", "getoutput",
                  "getstatusoutput")),
]


def _blocked(name: str):
    def _fn(*args, **kwargs):
        raise PermissionError(f"[sandbox] '{name}' is disabled while judging")
    return _fn


def _harden_child() -> List[Tuple[Any, str, Any]]:
    """Best-effort limits for the forked child. Returns the patches to undo."""
    try:
        # one scratch dir per judging process (children of the same parent share it)
        scratch = os.path.join(tempfile.gettempdir(), f"vico_sbx_{os.getppid()}")
        os.makedirs(scratch, exist_ok=True)
        os.chdir(scratch)
    except Exception:
        pass
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if FSIZE_LIMIT_MB > 0:
            b = FSIZE_LIMIT_MB * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (b, b))
        if MEM_LIMIT_MB > 0:
            # budget is relative to the inherited footprint, so a big parent
            # process (tokenizers etc.) does not make every child fail at once
            with open("/proc/self/statm") as f:
                vm_bytes = int(f.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
            b = vm_bytes + MEM_LIMIT_MB * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (b, b))
    except Exception:
        pass
    patches = []
    for mod, names in _GUARDED_CALLS:
        for name in names:
            if hasattr(mod, name):
                patches.append((mod, name, getattr(mod, name)))
                setattr(mod, name, _blocked(f"{mod.__name__}.{name}"))
    return patches


def _restore(patches) -> None:
    for mod, name, fn in patches:
        setattr(mod, name, fn)


# ----------------------------------------------------------------------------
# child
# ----------------------------------------------------------------------------
def _exec_worker(script: str, input_val: str, output_queue, harden: bool = True) -> None:
    input_lines = iter((input_val or "").splitlines())

    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")

    stdout_capture = io.StringIO()
    original_stdout, original_stdin = sys.stdout, sys.stdin
    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(input_val or "")
    context = {"__name__": "__main__", "input": fake_input}
    patches = _harden_child() if harden else []
    result = None
    try:
        exec(script, context)
        result = stdout_capture.getvalue()
    except SystemExit:
        result = stdout_capture.getvalue()
    except BaseException as e:  # noqa
        result = f"error: {e}"
    finally:
        _restore(patches)
        sys.stdout, sys.stdin = original_stdout, original_stdin
    output_queue.put(result)


def _ctx():
    if os.name == "posix":
        try:
            return mp.get_context("fork")
        except Exception:
            pass
    return mp.get_context()


# ----------------------------------------------------------------------------
# one chunk
# ----------------------------------------------------------------------------
def _run_chunk(
    scripts: Sequence[str],
    inputs: Sequence[str],
    time_limits: Sequence[float],
    hard_timeout_s: float,
) -> List[str]:
    n = len(scripts)
    results: List[Any] = [None] * n
    timed_out = [False] * n
    ctx = _ctx()
    procs, queues, starts = [], [], []

    for i in range(n):
        q = ctx.Queue()
        p = ctx.Process(target=_exec_worker, args=(scripts[i], inputs[i], q), daemon=True)
        p.start()
        procs.append(p)
        queues.append(q)
        starts.append(time.time())

    def _drain(i):
        if results[i] is not None or timed_out[i]:
            return
        try:
            results[i] = queues[i].get_nowait()
        except Empty:
            pass
        except Exception as e:  # broken pipe etc.
            results[i] = f"Execution Error: {e}"

    alive = set(range(n))
    while alive:
        now = time.time()
        for i in list(alive):
            p = procs[i]
            _drain(i)
            if not p.is_alive():
                _drain(i)
                alive.discard(i)
                continue
            elapsed = now - starts[i]
            if results[i] is None and not timed_out[i] and elapsed >= float(time_limits[i]):
                timed_out[i] = True
                p.terminate()                      # SIGTERM at judge time limit
            if elapsed >= hard_timeout_s:
                try:
                    p.kill()                       # SIGKILL at hard cap (30s)
                except Exception:
                    pass
                timed_out[i] = True
                alive.discard(i)
        time.sleep(0.001)

    for i, p in enumerate(procs):
        p.join(timeout=0.1)
        if timed_out[i] and results[i] is None:
            results[i] = TIMEOUT_RESULT
        if results[i] is None:
            _drain(i)
        if results[i] is None:
            results[i] = "Execution Error: no output"
        try:
            queues[i].cancel_join_thread()
            queues[i].close()
        except Exception:
            pass
    return results


def run_jobs(
    jobs: Sequence[Tuple[str, str, float]],
    max_procs: int = DEFAULT_MAX_PROCS,
    hard_timeout_s: float = DEFAULT_HARD_TIMEOUT_S,
) -> List[str]:
    """jobs: [(code, stdin, time_limit)] -> [stdout or error string]"""
    max_procs = max(1, int(max_procs))
    out: List[str] = []
    for s in range(0, len(jobs), max_procs):
        sub = jobs[s:s + max_procs]
        out.extend(_run_chunk(
            [j[0] for j in sub], [j[1] for j in sub], [float(j[2]) for j in sub], float(hard_timeout_s)
        ))
    return out


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def test_if_eq(x: str, y: str) -> bool:
    return " ".join((x or "").split()) == " ".join((y or "").split())


def is_valid_test(inp: Any, out: Any) -> bool:
    return isinstance(inp, str) and isinstance(out, str) and inp != "" and out != ""


def _safe_list(x):
    return x if isinstance(x, list) else []


def execute_code_matrix(
    requests: List[Dict[str, Any]],
    max_procs: int = DEFAULT_MAX_PROCS,
    hard_timeout_s: float = DEFAULT_HARD_TIMEOUT_S,
) -> List[Dict[str, List]]:
    """
    Batched execution of many (code, test-suite) requests in ONE scheduling pass.

    request = {"code": str, "test_input": [...], "test_output": [...], "time_limit": float}
    returns per request: {"exec": [str]*m, "correct": [bool]*m}
    Invalid tests (empty input/output) -> "Invalid Test", False (not executed).
    """
    jobs, owners = [], []
    outs: List[Dict[str, List]] = []
    for r_idx, req in enumerate(requests):
        tin, tout = _safe_list(req.get("test_input")), _safe_list(req.get("test_output"))
        m = min(len(tin), len(tout))
        outs.append({"exec": [INVALID_TEST] * m, "correct": [False] * m})
        code = req.get("code", "") or ""
        tl = float(req.get("time_limit", 4) or 4)
        for k in range(m):
            if is_valid_test(tin[k], tout[k]):
                jobs.append((code, tin[k], tl))
                owners.append((r_idx, k))

    if jobs:
        res = run_jobs(jobs, max_procs=max_procs, hard_timeout_s=hard_timeout_s)
        for (r_idx, k), r in zip(owners, res):
            outs[r_idx]["exec"][k] = r
            outs[r_idx]["correct"][k] = test_if_eq(r, requests[r_idx]["test_output"][k])
    return outs


def execute_unit_tests_on_key(
    data: List[dict],
    input_key: str,
    output_key: str,
    exec_key: str,
    corr_key: str,
    max_procs: int = DEFAULT_MAX_PROCS,
    hard_timeout_s: float = DEFAULT_HARD_TIMEOUT_S,
    code_key: str = "extracted_output",
) -> List[dict]:
    """Writes item[exec_key] / item[corr_key] as [m_code][m_case] matrices."""
    requests, owners = [], []
    for idx, item in enumerate(data):
        tl = float(item.get("test_time_limit", 4) or 4)
        for c_idx, code in enumerate(_safe_list(item.get(code_key, []))):
            requests.append({
                "code": code,
                "test_input": _safe_list(item.get(input_key, [])),
                "test_output": _safe_list(item.get(output_key, [])),
                "time_limit": tl,
            })
            owners.append((idx, c_idx))
        n_code = len(_safe_list(item.get(code_key, [])))
        item[exec_key] = [None] * n_code
        item[corr_key] = [None] * n_code

    results = execute_code_matrix(requests, max_procs=max_procs, hard_timeout_s=hard_timeout_s)
    for (idx, c_idx), r in zip(owners, results):
        data[idx][exec_key][c_idx] = r["exec"]
        data[idx][corr_key][c_idx] = r["correct"]
    return data
