# -*- coding: utf-8 -*-
# Copyright 2026 Yiming. Licensed under the Apache License, Version 2.0.
# Derived from RLAnything / Open-AgentRL (https://github.com/Gen-Verse/Open-AgentRL),
# Apache-2.0. Modified in this repository; see NOTICE for a summary of changes.
"""
Reward & advantage computation for the multi-step Code Agent.

Policy (step level)
-------------------
Every step of every trajectory (tracked by trajectory_uid + step_index) gets a
dual-dimension reward

    r = GT correctness            (pass ratio on GT unit tests,        in [0, 1])
      + syn_weight * syn ratio    (pass ratio on synthetic unit tests, in [0, 0.3])
                                                                total in [0, 1.3]

instead of the pure binary pass/fail signal, which gives a much finer quality
signal (partial passes / near-correct code are distinguished).

Step-level advantage: for each question, the 32 candidate trajectories form a
group. For step s, we z-score the rewards of the group at step index s
(trajectories that already terminated contribute the reward of their last step,
i.e. "the score they hold"), so every step is compared with what the other
candidates achieved at the same point of the interaction:

    A(traj, s) = (r(traj, s) - mean_s) / std_s

Each step becomes one training sample: prompt = message stack before the step,
response = the step's output, advantage = A(traj, s).

Reward model (UT generator) - unchanged spec, on the FINAL code of each trajectory.
"""
import os
import sys
import json
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from termcolor import cprint
from omegaconf import OmegaConf

from common.safe_io import atomic_write_json
from reward.advantage import (
    population_std as _std,
    step_level_advantages,
    step_reward,
    z_score_normalize,
)


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


def _pass_all(row: Any) -> bool:
    return isinstance(row, list) and len(row) > 0 and all(bool(x) for x in row)


def _safe_list(x):
    return x if isinstance(x, list) else []


if __name__ == "__main__":
    config = get_config()
    project_name = config.experiment.project

    if int(config.experiment.current_epoch) == 1:
        pretrained_model = config.model.policy_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    fn = str(config.experiment.function)
    is_train = (fn == "train")
    is_eval = not is_train
    pcfg = config.rollout.policy if is_train else config.evaluation.policy
    dataset = config.dataset.train_dataset if is_train else config.dataset.eval_dataset
    outputs_name = ("rl-" if is_train else "eval-") + str(pretrained_model).replace("/", ".") + "-" + dataset

    syn_weight = float(OmegaConf.select(config, "reward.syn_weight", default=0.3))
    max_gen_length = int(OmegaConf.select(pcfg, "max_gen_length", default=0))
    acc_range = OmegaConf.select(config, "reward.first_step_acc_range", default=None)

    file_name = f"../{project_name}/temp_data/outputs-{outputs_name}.json"
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("outputs file must be a list of task dicts")

    final_data: List[Dict[str, Any]] = []         # policy RL data (step level)
    reward_final_data: List[Dict[str, Any]] = []  # reward-model RL data (UT generator)

    # metrics
    n_traj = 0
    final_pass, first_pass, steps_used = [], [], []
    repair_den, repair_num = 0, 0
    final_rewards: List[float] = []
    group_std_dual, group_std_binary = [], []
    stop_reasons: Dict[str, int] = {}
    response_length_list: List[int] = []
    syn_ut_total_gt = syn_ut_correct_total = 0
    syn_ut_total_gt_ngt = syn_ut_perfect_total = 0

    for it in data:
        is_temp = (it.get("source") == "temp")
        if is_temp and not bool(it.get("temp_accept", False)):
            continue

        trajs = _safe_list(it.get("trajectories", []))
        if not trajs:
            continue

        # ---------- per-step dual rewards ----------
        R: List[List[float]] = []
        for tr in trajs:
            steps = _safe_list(tr.get("steps", []))
            rs = [step_reward(st, syn_weight, max_gen_length) for st in steps]
            for st, r in zip(steps, rs):
                st["reward"] = r
            R.append(rs)

        # ---------- metrics ----------
        gt_pass_all_final = []
        for tr, rs in zip(trajs, R):
            steps = tr["steps"]
            if not steps:
                continue
            n_traj += 1
            f_ok, l_ok = bool(steps[0]["gt_pass_all"]), bool(steps[-1]["gt_pass_all"])
            first_pass.append(float(f_ok))
            final_pass.append(float(l_ok))
            gt_pass_all_final.append(l_ok)
            steps_used.append(len(steps))
            final_rewards.append(rs[-1])
            response_length_list.extend(int(st.get("response_length", 0)) for st in steps)
            stop_reasons[tr.get("stop_reason", "?")] = stop_reasons.get(tr.get("stop_reason", "?"), 0) + 1
            if not f_ok:
                repair_den += 1
                repair_num += int(l_ok)
        it["gt_pass_all"] = gt_pass_all_final
        it["acc"] = (sum(gt_pass_all_final) / len(gt_pass_all_final)) if gt_pass_all_final else 0.0

        step1_dual = [rs[0] for rs in R if rs]
        step1_bin = [float(tr["steps"][0]["gt_pass_all"]) for tr in trajs if tr.get("steps")]
        if len(step1_dual) > 1:
            group_std_dual.append(_std(step1_dual))
            group_std_binary.append(_std(step1_bin))

        # ---------- step-level advantage (group = same question, same step_index) ----------
        if is_train:
            first_acc = (sum(step1_bin) / len(step1_bin)) if step1_bin else 0.0
            if acc_range is not None and not (float(acc_range[0]) <= first_acc <= float(acc_range[1])):
                pass  # task outside the difficulty window: skip policy samples (reward-model data still built)
            else:
                for ti, s, a in step_level_advantages(R):
                    tr = trajs[ti]
                    st = tr["steps"][s]
                    final_data.append({
                        "prompt": st["prompt"],
                        "response": st["full_output"],
                        "reward": a,
                        "raw_reward": float(R[ti][s]),
                        "trajectory_uid": tr.get("trajectory_uid"),
                        "step_index": int(st.get("step_index", s + 1)),
                    })

        # ======================================================================
        # Reward model (UT generator) data on FINAL codes  (spec unchanged)
        # ======================================================================
        m_code = len(_safe_list(it.get("extracted_output", [])))
        corr_gt = _safe_list(it.get("correctness", []))
        corr_syn = _safe_list(it.get("syn_correctness", []))
        gt_code_idx = [j for j in range(min(m_code, len(corr_gt))) if _pass_all(corr_gt[j])]
        gt_set = set(gt_code_idx)
        non_gt_idx = [j for j in range(m_code) if j not in gt_set]
        syn_rows = [row for row in corr_syn if isinstance(row, list)]
        K_syn = min((len(row) for row in syn_rows), default=0)

        def syn_ok(j: int, k: int) -> bool:
            if j >= len(corr_syn) or not isinstance(corr_syn[j], list) or k >= len(corr_syn[j]):
                return False
            v = corr_syn[j][k]
            return (v > 0) if isinstance(v, (int, float)) and not isinstance(v, bool) else bool(v)

        if K_syn > 0 and gt_code_idx:
            syn_ut_total_gt += K_syn
            for k in range(K_syn):
                if all(syn_ok(j, k) for j in gt_code_idx):
                    syn_ut_correct_total += 1
            if non_gt_idx:
                syn_ut_total_gt_ngt += K_syn
                for k in range(K_syn):
                    if all(syn_ok(j, k) for j in gt_code_idx) and all(not syn_ok(j, k) for j in non_gt_idx):
                        syn_ut_perfect_total += 1

        syn_inputs = _safe_list(it.get("syn_input", []))
        syn_outputs = _safe_list(it.get("syn_output", []))
        syn_full = _safe_list(it.get("syn_full_output", []))
        syn_prompts = _safe_list(it.get("syn_prompt", []))
        K_text = min(len(syn_inputs), len(syn_outputs), len(syn_full)) if syn_full else min(len(syn_inputs), len(syn_outputs))
        K = min(K_text, K_syn)
        if K <= 0 or not gt_code_idx:
            continue

        raw_ut_rewards, ut_meta = [], []
        for k in range(K):
            inp, out = syn_inputs[k], syn_outputs[k]
            is_gt_ut = all(syn_ok(j, k) for j in gt_code_idx)
            if not non_gt_idx:
                r = 1.0 if is_gt_ut else -1.0
                n_dis = n_not = 0
            else:
                n_dis = sum(1 for j in non_gt_idx if not syn_ok(j, k))
                n_not = sum(1 for j in non_gt_idx if syn_ok(j, k))
                r = float(n_dis) if is_gt_ut else -float(n_not)
            raw_ut_rewards.append(r)
            ut_meta.append({
                "k": k, "valid_ut": bool(isinstance(inp, str) and isinstance(out, str) and inp and out),
                "is_gt_ut": bool(is_gt_ut), "raw_reward": r, "n_code": m_code,
                "n_gt_code": len(gt_code_idx), "n_non_gt_code": len(non_gt_idx),
                "distinguish_non_gt": n_dis, "not_distinguish_non_gt": n_not,
            })
        ut_rewards = z_score_normalize(raw_ut_rewards)
        if is_train:
            for k in range(K):
                reward_final_data.append({
                    "prompt": syn_prompts[k] if (k < len(syn_prompts) and isinstance(syn_prompts[k], str)) else it.get("prompt", ""),
                    "response": syn_full[k] if (k < len(syn_full) and isinstance(syn_full[k], str)) else "",
                    "reward": float(ut_rewards[k]),
                    "raw_reward": float(raw_ut_rewards[k]),
                    "meta": ut_meta[k],
                    "evaluated_task_question": it.get("question", ""),
                })

    # ------------------------------------------------------------------
    # write back + log
    # ------------------------------------------------------------------
    atomic_write_json(file_name, data)

    mean = lambda x: (sum(x) / len(x)) if x else 0.0  # noqa: E731
    metrics = {
        "step": int(config.experiment.current_epoch),
        "split": "train" if is_train else "eval",
        "num_trajectories": n_traj,
        "pass@1": mean(final_pass),                      # final-step strict GT pass-all
        "first_exec_pass_rate": mean(first_pass),        # step-1 strict GT pass-all
        "multi_step_repair_rate": (repair_num / repair_den) if repair_den else 0.0,
        "avg_steps": mean(steps_used),
        "reward_mean": mean(final_rewards),
        "group_reward_std_dual": mean(group_std_dual),     # std of dual reward inside a 32-candidate group
        "group_reward_std_binary": mean(group_std_binary), # same group, pure binary pass/fail reward
        "avg_response_length": mean(response_length_list),
        "syn_ut_acc": (syn_ut_correct_total / syn_ut_total_gt) if syn_ut_total_gt else 0.0,
        "perfect_ut_rate": (syn_ut_perfect_total / syn_ut_total_gt_ngt) if syn_ut_total_gt_ngt else 0.0,
        "stop_reasons": stop_reasons,
        "num_policy_samples": len(final_data),
        "num_reward_samples": len(reward_final_data),
    }

    res_dir = f"../{project_name}/results"
    os.makedirs(res_dir, exist_ok=True)
    line = (f"train step: {metrics['step']}  pass@1: {metrics['pass@1']:.6f}  "
            f"first_exec_pass: {metrics['first_exec_pass_rate']:.6f}  "
            f"repair_rate: {metrics['multi_step_repair_rate']:.6f}  avg_steps: {metrics['avg_steps']:.2f}  "
            f"reward_std(dual/binary): {metrics['group_reward_std_dual']:.4f}/{metrics['group_reward_std_binary']:.4f}  "
            f"avg length: {metrics['avg_response_length']:.2f}  syn_ut_acc: {metrics['syn_ut_acc']:.6f}  "
            f"perfect_ut_rate: {metrics['perfect_ut_rate']:.6f}")
    cprint("\n" + line, color="green")
    with open(f"{res_dir}/results-{outputs_name}.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    with open(f"{res_dir}/metrics-{'train' if is_train else 'eval'}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    if is_eval:
        sys.exit(0)

    atomic_write_json(f"../{project_name}/temp_data/{config.dataset.optimization_data}.json", final_data)
    atomic_write_json(f"../{project_name}/temp_data/{config.dataset.reward_optimization_data}.json", reward_final_data)
