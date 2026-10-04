# -*- coding: utf-8 -*-
# Copyright 2026 Yiming. Licensed under the Apache License, Version 2.0.
# Derived from RLAnything / Open-AgentRL (https://github.com/Gen-Verse/Open-AgentRL),
# Apache-2.0. Modified in this repository; see NOTICE for a summary of changes.
"""
Multi-step agentic policy rollout for Code Agent RL.

Single-turn "generate once" is turned into an  execute -> feedback -> fix  loop:

  step t:
    1. vLLM (tensor parallel, one engine per gpu_group, engines stay alive
       across steps) generates a solution for every *active* trajectory from
       its message stack.
    2. The extracted python code is executed in the sandbox
       (common/sandbox.py: multiprocessing.Process isolation, stdin/stdout
       redirection, <=128 concurrent processes per chunk, 30s hard kill) on
         - ALL GT unit tests + ALL synthetic unit tests  -> step-level reward
         - a feedback subset (first `num_feedback_gt_tests` GT tests, i.e. the
           public examples, + synthetic tests from the reward model)
    3. The execution_result of the feedback subset is appended to the message
       stack as the next user turn.
    4. Termination (per trajectory):
         success         : all public GT examples in the feedback passed
                           (synthetic tests are shown as reference only: a wrong
                           synthetic test can no longer block a correct solution;
                           set success_criterion=all_feedback for the strict rule)
         no_improvement  : feedback score did not improve for `patience` (=2)
                           consecutive steps
         max_steps       : reached `max_steps`

Every step is recorded with (trajectory_uid, step_index) so that the reward
stage can compute step-level advantages (reward/llm_rl_reward.py).

Output (policy-stage file, overwritten atomically), per task:
  prompt, full_output, extracted_output, response_length      (final step)
  execution_result / correctness / syn_execution_result /
  syn_correctness                                               (final step)
  trajectories: [{trajectory_uid, num_steps, stop_reason, steps: [...]}]
"""
import os
import re
import sys
import json
import random
import time
import multiprocessing as mp
from queue import Empty
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from termcolor import cprint
from transformers import AutoTokenizer
from omegaconf import OmegaConf

from common.safe_io import atomic_write_json
from common.sandbox import execute_code_matrix, is_valid_test


os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHUNK_SIZE = 1024
GEN_TIMEOUT_S = float(os.environ.get("GEN_TIMEOUT_S", "1800"))


def get_token_lengths(strings, tokenizer):
    return [len(tokenizer.encode(s, add_special_tokens=False)) for s in strings]


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


def get_policy_stage_output_path(project_name, outputs_name, num_node, node_index):
    return f"../{project_name}/temp_data/outputs-{int(node_index)}-{outputs_name}.json"


def _as_list(x):
    return x if isinstance(x, list) else []


NO_CODE = "We can not extract the code in the output. "


def extract_code(full_output: str):
    matches = re.findall(r"```python(.*?)```", full_output or "", re.DOTALL)
    if matches:
        return matches[-1].strip()
    return NO_CODE


############################
# vLLM Worker Pool (one TP engine per gpu_group, kept alive across agent steps)
############################
def split_prompts(prompts, n):
    k, m = divmod(len(prompts), n)
    return [prompts[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def worker_fn(pretrained_model, gpu_ids, task_queue, result_queue,
              max_model_len, max_generation_token, temp):
    # IMPORTANT: set per-process visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(f"[vLLM] (worker {gpu_ids}) Loading model: {pretrained_model}", flush=True)

    try:
        import inspect
        import torch  # noqa
        from vllm import LLM, SamplingParams  # noqa

        llm_kwargs = dict(
            model=pretrained_model,
            dtype="bfloat16",
            tensor_parallel_size=len(gpu_ids),
            gpu_memory_utilization=0.85,
            max_model_len=int(max_model_len),
            enforce_eager=False,
            disable_custom_all_reduce=True,
            trust_remote_code=True,
        )
        sig = inspect.signature(LLM.__init__)
        llm_kwargs = {k: v for k, v in llm_kwargs.items() if k in sig.parameters}

        llm = LLM(**llm_kwargs)

        sampling_params = SamplingParams(
            temperature=float(temp),
            top_p=0.95,
            top_k=-1,
            min_p=0.0,
            max_tokens=int(max_generation_token),
            stop=["</answer>", "User:", "Human:", "Assistant:", "<|im_end|>", "<|endoftext|>"],
        )
    except Exception as e:
        result_queue.put(("ERROR", f"init failed on GPUs {gpu_ids}: {repr(e)}"))
        return

    def _manual_vllm_shutdown(llm_obj):
        for eng_name in ("llm_engine", "_llm_engine", "_engine", "engine"):
            eng = getattr(llm_obj, eng_name, None)
            if eng is None:
                continue
            me = getattr(eng, "model_executor", None)
            if me is not None and hasattr(me, "shutdown"):
                me.shutdown()
                return True
            if hasattr(eng, "shutdown"):
                eng.shutdown()
                return True
        return False

    while True:
        task = task_queue.get()
        if task == "STOP":
            print(f"[vLLM] (worker {gpu_ids}) STOP received, shutting down...", flush=True)
            break
        task_id, prompts = task
        try:
            outputs = llm.generate(prompts, sampling_params)
            result_texts = [
                out.outputs[0].text if (out.outputs and len(out.outputs) > 0) else ""
                for out in outputs
            ]
            result_queue.put((task_id, result_texts))
        except Exception as e:
            result_queue.put(("ERROR", f"generate failed on GPUs {gpu_ids}: {repr(e)}"))
            break

    try:
        _manual_vllm_shutdown(llm)
    except Exception:
        pass
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


def start_workers(pretrained_model, gpu_configs, max_model_len, max_generation_token, temp):
    task_queues, result_queues, processes = [], [], []
    for gpu_ids in gpu_configs:
        tq = mp.Queue()
        rq = mp.Queue()
        p = mp.Process(
            target=worker_fn,
            args=(pretrained_model, gpu_ids, tq, rq, max_model_len, max_generation_token, temp),
        )
        p.start()
        task_queues.append(tq)
        result_queues.append(rq)
        processes.append(p)
    return task_queues, result_queues, processes


def stop_workers(task_queues, result_queues, processes, join_timeout_s: float = 120.0):
    for q in task_queues:
        try:
            q.put("STOP")
        except Exception:
            pass

    for i, p in enumerate(processes):
        p.join(timeout=join_timeout_s)
        if p.is_alive():
            print(f"[WARN] worker {i} pid={p.pid} still alive after {join_timeout_s}s, terminate()", flush=True)
            p.terminate()
            p.join(timeout=10.0)
            if p.is_alive():
                print(f"[WARN] worker {i} pid={p.pid} still alive after terminate, kill()", flush=True)
                try:
                    p.kill()
                except Exception:
                    pass
                p.join(timeout=10.0)

    # IMPORTANT: do NOT join_thread() (it can hang forever). cancel it.
    for q in (task_queues or []):
        try:
            q.cancel_join_thread()
            q.close()
        except Exception:
            pass
    for q in (result_queues or []):
        try:
            q.cancel_join_thread()
            q.close()
        except Exception:
            pass


def generate_results(
    all_prompts,
    gpu_groups,
    task_queues,
    result_queues,
    desc: str = "policy",
    timeout_s: float = 1800.0,
):
    """
    Same contract as your “good” code:
    - split prompts by worker count
    - send one job per worker
    - poll result queues with timeout
    """
    if not all_prompts:
        return []

    chunks = split_prompts(all_prompts, len(gpu_groups))

    jobs = []
    for i, (q, prompts) in enumerate(zip(task_queues, chunks)):
        if prompts:
            q.put((i, prompts))
            jobs.append(i)

    results_by_job = {}
    remaining = set(jobs)
    start_time = time.time()

    while remaining:
        now = time.time()
        if now - start_time > timeout_s:
            raise RuntimeError(f"[{desc}] timeout waiting results; still missing jobs {sorted(remaining)}")

        for i, rq in enumerate(result_queues):
            if i not in remaining:
                continue
            try:
                task_id, result = rq.get(timeout=0.1)
            except Empty:
                continue

            if task_id == "ERROR":
                raise RuntimeError(f"[{desc}] worker {i} reported error: {result}")

            results_by_job[task_id] = result
            remaining.remove(task_id)

    out = []
    for i, prompts in enumerate(chunks):
        if not prompts:
            continue
        if i not in results_by_job:
            raise RuntimeError(f"[{desc}] missing result for job {i} (this should not happen)")
        out.extend(results_by_job[i])
    return out


############################
# Multi-step agent: prompts / feedback
############################
SYSTEM_PROMPT = "You are a helpful assistant help user solve problems."

USER_PROMPT = (
    "You need to think first then write python script. You should use input() to input and print() "
    "to output in your script. Put the complete script in one ```python``` code block at the end of your answer.\n"
    "This is the problem:\n{problem}"
)
USER_PROMPT_EVAL = (
    "You need to think first then write python script. You should use input() to input and print() "
    "to output in your script. Your code should output the results based on the input read in, rather than "
    "generating the given test example. Put the complete script in one ```python``` code block at the end of "
    "your answer.\nThis is the problem:\n{problem}"
)


def render_messages(messages: List[Dict[str, str]], start_with_think: bool) -> str:
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    s += "<|im_start|>assistant\n"
    if start_with_think:
        s += "<think>"
    return s


def _clip(s: Any, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + f"...(truncated, {len(s)} chars)"


def build_feedback(code: str, fb_tests: List[Dict[str, Any]], max_chars: int) -> str:
    """fb_tests: [{kind, input, expected, got, passed}] -> next user turn (execution_result)."""
    if code == NO_CODE:
        return ("Execution result: no ```python``` code block was found in your answer, so nothing was executed.\n"
                "Please output the COMPLETE python script in one ```python``` code block.")
    lines = ["Your code has been executed. Execution results:"]
    n_pass = 0
    for i, t in enumerate(fb_tests):
        n_pass += int(t["passed"])
        tag = "public example" if t["kind"] == "gt" else "auto-generated test, may be inaccurate"
        got = t["got"]
        if got == "Timeout Error":
            verdict = "TIME LIMIT EXCEEDED"
        elif isinstance(got, str) and (got.startswith("error:") or got.startswith("Execution Error")):
            verdict = "RUNTIME ERROR"
        else:
            verdict = "PASSED" if t["passed"] else "WRONG ANSWER"
        lines.append(
            f"[Test {i + 1}] ({tag}) {verdict}\n"
            f"Input:\n{_clip(t['input'], max_chars)}\n"
            f"Expected output:\n{_clip(t['expected'], max_chars)}\n"
            f"Your output:\n{_clip(got, max_chars)}"
        )
    lines.append(f"Passed {n_pass}/{len(fb_tests)} tests.")
    lines.append("Analyze the failures, fix your code and output the COMPLETE corrected python script in one "
                 "```python``` code block. If you are confident an auto-generated test is wrong, keep your logic.")
    return "\n\n".join(lines)


def feedback_indices(item: Dict[str, Any], num_fb_gt: int):
    tin, tout = _as_list(item.get("test_input")), _as_list(item.get("test_output"))
    sin, sout = _as_list(item.get("syn_input")), _as_list(item.get("syn_output"))
    gt_idx = [k for k in range(min(len(tin), len(tout))) if is_valid_test(tin[k], tout[k])][:max(0, num_fb_gt)]
    syn_idx = [k for k in range(min(len(sin), len(sout))) if is_valid_test(sin[k], sout[k])]
    return gt_idx, syn_idx


def build_context(traj, tokenizer, start_with_think, max_history_turns, budget_tokens):
    """System + original problem are always kept; only the last `max_history_turns`
    (assistant, feedback) rounds are kept, and older rounds are dropped until
    the prompt fits into the vLLM context budget."""
    msgs = traj["messages"]
    head, tail = msgs[:2], msgs[2:]
    keep = 2 * max(0, int(max_history_turns))
    tail = tail[-keep:] if keep > 0 else []
    while True:
        prompt = render_messages(head + tail, start_with_think)
        if not tail or len(tokenizer.encode(prompt, add_special_tokens=False)) <= budget_tokens:
            return prompt
        tail = tail[2:]


############################
# Main: multi-step POLICY rollout
############################
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    config = get_config()
    project_name = config.experiment.project

    if int(config.experiment.current_epoch) == 1:
        policy_model = config.model.policy_model
    else:
        policy_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    fn = str(config.experiment.function)
    is_train = (fn == "train")
    pcfg = config.rollout.policy if is_train else config.evaluation.policy
    dataset = config.dataset.train_dataset if is_train else config.dataset.eval_dataset
    max_model_len = int(pcfg.model_length)
    max_generation_token = int(pcfg.max_gen_length)
    temp = float(pcfg.temperature)
    k_sample = int(pcfg.num_response_per_task)
    start_with_think = bool(OmegaConf.select(pcfg, "start_with_think", default=None)
                            or OmegaConf.select(pcfg, "if_start_with_think", default=False))
    max_steps = int(OmegaConf.select(pcfg, "max_steps", default=5))
    patience = int(OmegaConf.select(pcfg, "patience", default=2))
    num_fb_gt = int(OmegaConf.select(pcfg, "num_feedback_gt_tests", default=1))
    max_history_turns = int(OmegaConf.select(pcfg, "max_history_turns", default=2))
    fb_max_chars = int(OmegaConf.select(pcfg, "feedback_max_chars", default=300))
    success_criterion = str(OmegaConf.select(pcfg, "success_criterion", default="public_gt"))
    gpu_groups = [list(g) for g in OmegaConf.select(pcfg, "gpu_groups", default=[[0, 1, 2, 3]])]
    num_node = int(config.experiment.num_node)
    node_index = int(config.experiment.node_index)
    max_procs = int(OmegaConf.select(config, "execute.num_chunk", default=128))
    hard_timeout_s = float(OmegaConf.select(config, "execute.hard_timeout_s", default=30.0))

    outputs_name = ("rl-" if is_train else "eval-") + str(policy_model).replace("/", ".") + "-" + dataset
    stage_path = get_policy_stage_output_path(project_name, outputs_name, num_node, node_index)
    if not os.path.exists(stage_path):
        raise FileNotFoundError(f"task file not found (run build_tasks.py + llm_reward_rollout.py first): {stage_path}")
    with open(stage_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(policy_model, trust_remote_code=True)
    user_tmpl = USER_PROMPT if is_train else USER_PROMPT_EVAL
    budget_tokens = max_model_len - max_generation_token - 16

    # ---- init trajectories: task i has k_sample trajectories ----
    trajs: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        task_uid = item.get("task_uid", f"n{node_index}-t{i}")
        first_msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_tmpl.format(problem=item.get("question", ""))}]
        item["prompt"] = render_messages(first_msgs, start_with_think)
        item["fb_gt_idx"], item["fb_syn_idx"] = feedback_indices(item, num_fb_gt)
        for k in range(k_sample):
            trajs.append({
                "task_idx": i, "trajectory_uid": f"{task_uid}-r{k}",
                "messages": [dict(m) for m in first_msgs],       # independent copy per trajectory
                "steps": [], "best_score": -1.0, "no_improve": 0,
                "done": False, "stop_reason": None,
            })

    visible = len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()])
    if visible == 0:
        import torch
        visible = torch.cuda.device_count()
    need = max(max(g) for g in gpu_groups) + 1
    if visible < need:
        raise RuntimeError(f"gpu_groups={gpu_groups} needs {need} visible GPUs, got {visible}")
    cprint(f"[vLLM] engines={len(gpu_groups)} gpu_groups={gpu_groups} (TP={[len(g) for g in gpu_groups]})", "cyan")

    task_queues, result_queues, processes = start_workers(
        policy_model, gpu_groups, max_model_len, max_generation_token, temp
    )
    try:
        for step_index in range(1, max_steps + 1):
            active = [t for t in trajs if not t["done"]]
            if not active:
                break
            cprint(f"[agent] step {step_index}/{max_steps}: {len(active)} active trajectories", "green")

            # 1) generate from message stacks
            prompts = [build_context(t, tokenizer, start_with_think, max_history_turns, budget_tokens) for t in active]
            order = list(range(len(prompts)))
            random.shuffle(order)                               # balance long/short prompts across engines
            outs_shuf: List[str] = []
            for s in range(0, len(order), CHUNK_SIZE):
                sub = [prompts[j] for j in order[s:s + CHUNK_SIZE]]
                outs_shuf.extend(generate_results(sub, gpu_groups, task_queues, result_queues,
                                                  desc=f"policy-step{step_index}", timeout_s=GEN_TIMEOUT_S))
            outputs = [""] * len(prompts)
            for pos, j in enumerate(order):
                outputs[j] = outs_shuf[pos] or ""
            lengths = get_token_lengths(outputs, tokenizer)
            codes = [extract_code(o) for o in outputs]

            # 2) sandbox execution: full GT + full SYN for every active trajectory
            requests = []
            for t, code in zip(active, codes):
                item = data[t["task_idx"]]
                tl = float(item.get("test_time_limit", 4) or 4)
                requests.append({"code": code, "test_input": _as_list(item.get("test_input")),
                                 "test_output": _as_list(item.get("test_output")), "time_limit": tl})
                requests.append({"code": code, "test_input": _as_list(item.get("syn_input")),
                                 "test_output": _as_list(item.get("syn_output")), "time_limit": tl})
            t0 = time.time()
            exec_res = execute_code_matrix(requests, max_procs=max_procs, hard_timeout_s=hard_timeout_s)
            cprint(f"[sandbox] step {step_index}: {len(requests)} suites executed in {time.time() - t0:.1f}s", "cyan")

            # 3) record step, build feedback, check termination
            for a_i, t in enumerate(active):
                item = data[t["task_idx"]]
                gt, syn = exec_res[2 * a_i], exec_res[2 * a_i + 1]
                gt_corr, syn_corr = gt["correct"], syn["correct"]
                sin, sout = _as_list(item.get("syn_input")), _as_list(item.get("syn_output"))
                syn_valid = [k for k in range(len(syn_corr)) if is_valid_test(sin[k], sout[k])]

                fb_tests = []
                for k in item["fb_gt_idx"]:
                    fb_tests.append({"kind": "gt", "input": item["test_input"][k], "expected": item["test_output"][k],
                                     "got": gt["exec"][k], "passed": bool(gt_corr[k])})
                for k in item["fb_syn_idx"]:
                    fb_tests.append({"kind": "syn", "input": sin[k], "expected": sout[k],
                                     "got": syn["exec"][k], "passed": bool(syn_corr[k])})
                fb_score = (sum(x["passed"] for x in fb_tests) / len(fb_tests)) if fb_tests else 0.0

                assistant_text = ("<think>" if start_with_think else "") + outputs[a_i]
                t["steps"].append({
                    "step_index": step_index,
                    "prompt": prompts[a_i],
                    "full_output": outputs[a_i],
                    "code": codes[a_i],
                    "response_length": int(lengths[a_i]),
                    "gt_correct": [bool(x) for x in gt_corr],
                    "syn_correct": [bool(x) for x in syn_corr],
                    "gt_ratio": (sum(gt_corr) / len(gt_corr)) if gt_corr else 0.0,
                    "syn_ratio": (sum(syn_corr[k] for k in syn_valid) / len(syn_valid)) if syn_valid else 0.0,
                    "gt_pass_all": bool(gt_corr) and all(gt_corr),
                    "feedback_score": fb_score,
                })
                t["last_exec"] = (gt["exec"], syn["exec"])

                # termination: success / no improvement for `patience` steps / max steps
                public_gt = [x for x in fb_tests if x["kind"] == "gt"]
                # success judged on public GT examples; fall back to all feedback tests if there are none
                judge = public_gt if (success_criterion == "public_gt" and public_gt) else fb_tests
                if judge and codes[a_i] != NO_CODE and all(x["passed"] for x in judge):
                    t["done"], t["stop_reason"] = True, "success"
                elif not fb_tests:
                    t["done"], t["stop_reason"] = True, "no_feedback_tests"
                else:
                    if fb_score > t["best_score"] + 1e-9:
                        t["best_score"], t["no_improve"] = fb_score, 0
                    else:
                        t["no_improve"] += 1
                    if t["no_improve"] >= patience:
                        t["done"], t["stop_reason"] = True, "no_improvement"
                    elif step_index >= max_steps:
                        t["done"], t["stop_reason"] = True, "max_steps"

                if not t["done"]:
                    t["messages"].append({"role": "assistant", "content": assistant_text})
                    t["messages"].append({"role": "user",
                                          "content": build_feedback(codes[a_i], fb_tests, fb_max_chars)})
        for t in trajs:
            if not t["done"]:
                t["done"], t["stop_reason"] = True, "max_steps"
        cprint("[agent] multi-step rollout done!", "green")
    finally:
        stop_workers(task_queues, result_queues, processes)
        try:
            import torch
            time.sleep(2)
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- write back: final-step view (for env / reward-model stages) + full trajectories ----
    for item in data:
        for key in ("full_output", "extracted_output", "response_length", "execution_result", "correctness",
                    "syn_execution_result", "syn_correctness", "trajectories"):
            item[key] = []
    for t in trajs:
        item = data[t["task_idx"]]
        last = t["steps"][-1]
        item["full_output"].append(last["full_output"])
        item["extracted_output"].append(last["code"])
        item["response_length"].append(last["response_length"])
        item["correctness"].append(last["gt_correct"])
        item["syn_correctness"].append(last["syn_correct"])
        item["execution_result"].append(t["last_exec"][0])
        item["syn_execution_result"].append(t["last_exec"][1])
        item["trajectories"].append({
            "trajectory_uid": t["trajectory_uid"],
            "num_steps": len(t["steps"]),
            "stop_reason": t["stop_reason"],
            "steps": t["steps"],
        })

    atomic_write_json(stage_path, data)
    n_steps = [len(t["steps"]) for t in trajs]
    cprint(f"[policy] saved {stage_path}  avg_steps={sum(n_steps) / max(1, len(n_steps)):.2f}", "cyan")
    os._exit(0)
