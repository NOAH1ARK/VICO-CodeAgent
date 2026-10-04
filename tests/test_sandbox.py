"""CPU-only tests for the process-isolated code sandbox (common/sandbox.py)."""
from common.sandbox import (
    INVALID_TEST,
    TIMEOUT_RESULT,
    execute_code_matrix,
    execute_unit_tests_on_key,
    run_jobs,
    test_if_eq as outputs_equal,
)

SUM_CODE = "n = int(input())\nprint(sum(map(int, input().split())))\n"


def test_correct_and_wrong_solutions():
    reqs = [
        {"code": SUM_CODE, "test_input": ["5\n1 2 3 4 5\n", "1\n7\n"], "test_output": ["15", "7"], "time_limit": 2},
        {"code": "print(0)", "test_input": ["5\n1 2 3 4 5\n"], "test_output": ["15"], "time_limit": 2},
    ]
    out = execute_code_matrix(reqs, max_procs=4)
    assert out[0]["correct"] == [True, True]
    assert out[1]["correct"] == [False]


def test_sys_stdin_and_exit_are_supported():
    code = "import sys\ndata = sys.stdin.read().split()\nprint(int(data[0]) * 2)\nexit()\nprint('unreachable')\n"
    assert run_jobs([(code, "21\n", 2)]) == ["42\n"]


def test_runtime_error_is_reported():
    (res,) = run_jobs([("raise ValueError('boom')", "", 2)])
    assert res.startswith("error:") and "boom" in res


def test_infinite_loop_times_out_without_blocking_others():
    jobs = [("while True: pass", "", 0.5), ("print('ok')", "", 2)]
    res = run_jobs(jobs, max_procs=2, hard_timeout_s=5)
    assert res[0] == TIMEOUT_RESULT
    assert res[1].strip() == "ok"


def test_large_output_is_not_misjudged_as_timeout():
    # ~2 MB through the Queue: the parent must drain while the child is alive
    code = "print('x' * 2_000_000)"
    (res,) = run_jobs([(code, "", 3)])
    assert len(res.strip()) == 2_000_000


def test_chunked_scheduling_preserves_order():
    jobs = [(f"print({i})", "", 2) for i in range(20)]
    res = run_jobs(jobs, max_procs=3)
    assert [r.strip() for r in res] == [str(i) for i in range(20)]


def test_destructive_calls_are_blocked():
    code = "import os, shutil\ntry:\n    os.system('echo hi')\nexcept PermissionError:\n    print('blocked')\n"
    (res,) = run_jobs([(code, "", 2)])
    assert res.strip() == "blocked"


def test_memory_bomb_is_contained():
    code = "x = bytearray(16 * 1024 ** 3)\nprint('allocated')"
    (res,) = run_jobs([(code, "", 5)])
    assert "allocated" not in res


def test_invalid_tests_are_skipped():
    out = execute_code_matrix([{"code": "print(1)", "test_input": ["", "1"], "test_output": ["1", "1"]}])
    assert out[0]["exec"][0] == INVALID_TEST and out[0]["correct"] == [False, True]


def test_whitespace_insensitive_compare():
    assert outputs_equal("1  2\n3\n", "1 2 3")
    assert not outputs_equal("1 2", "1 3")


def test_execute_unit_tests_on_key_shapes():
    data = [{"extracted_output": [SUM_CODE, "print(1)"], "test_input": ["2\n3 4\n"], "test_output": ["7"]}]
    execute_unit_tests_on_key(data, "test_input", "test_output", "exec", "corr", max_procs=4)
    assert data[0]["corr"] == [[True], [False]]
