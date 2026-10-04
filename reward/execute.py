# -*- coding: utf-8 -*-
# Copyright 2026 Yiming. Licensed under the Apache License, Version 2.0.
# Derived from RLAnything / Open-AgentRL (https://github.com/Gen-Verse/Open-AgentRL),
# Apache-2.0. Modified in this repository; see NOTICE for a summary of changes.
"""
Execute unit tests (GT + SYN) for the FINAL code of every trajectory in a
policy-stage output JSON file.

In the multi-step coding pipeline the policy rollout already executes every
step inside its "execute -> feedback -> fix" loop (see sample/llm_policy_rollout.py),
so this script is only needed when you want to re-judge final codes
standalone (e.g. single-turn ablation with rollout.policy.max_steps=1, or
re-scoring an old rollout file with a different time limit).

Input / Output (same file, overwritten atomically):
  execution_result:      [m_code][m_gt]  (str)
  correctness:           [m_code][m_gt]  (bool)
  syn_execution_result:  [m_code][m_syn] (str)
  syn_correctness:       [m_code][m_syn] (bool)

Execution core lives in common/sandbox.py:
  multiprocessing.Process isolation, stdin/stdout redirection,
  <= execute.num_chunk (128) concurrent processes per chunk,
  per-test time limit for judging + 30s hard kill for abnormal processes.
"""

import os
import sys
import json
import argparse
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from termcolor import cprint

from common.safe_io import atomic_write_json
from common.sandbox import execute_unit_tests_on_key


def get_policy_stage_output_path(project_name, outputs_name, num_node, node_index):
    return f"../{project_name}/temp_data/outputs-{int(node_index)}-{outputs_name}.json"


def get_config():
    cli_conf = OmegaConf.from_cli()
    cfg_path = OmegaConf.select(cli_conf, "config", default=None)
    if cfg_path is None:
        raise ValueError('Missing "config=...".')
    yaml_conf = OmegaConf.load(cfg_path)
    return OmegaConf.merge(yaml_conf, cli_conf)


def _resolve_policy_stage_path_from_config(cfg) -> str:
    project_name = cfg.experiment.project
    is_train = (str(cfg.experiment.function) == "train")
    num_node = int(OmegaConf.select(cfg, "experiment.num_node", default=1))
    node_index = int(OmegaConf.select(cfg, "experiment.node_index", default=0))

    if int(cfg.experiment.current_epoch) == 1:
        policy_model = cfg.model.policy_model
    else:
        policy_model = "../" + project_name + "/ckpt/" + cfg.model.optimized_name

    if is_train:
        dataset = cfg.dataset.train_dataset
    else:
        dataset = cfg.dataset.eval_dataset if OmegaConf.select(cfg, "dataset.eval_dataset") else cfg.dataset.train_dataset

    outputs_name = ("rl-" if is_train else "eval-") + str(policy_model).replace("/", ".") + "-" + dataset
    return get_policy_stage_output_path(project_name, outputs_name, num_node, node_index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_stage_path", type=str, default=None)
    args, _unknown = parser.parse_known_args()

    cfg = get_config()
    policy_stage_path = args.policy_stage_path or _resolve_policy_stage_path_from_config(cfg)
    max_procs = int(OmegaConf.select(cfg, "execute.num_chunk", default=128))
    hard_timeout_s = float(OmegaConf.select(cfg, "execute.hard_timeout_s", default=30.0))

    if not os.path.exists(policy_stage_path):
        raise FileNotFoundError(f"policy_stage_path not found: {policy_stage_path}")

    cprint(f"[load] {policy_stage_path}", "cyan")
    with open(policy_stage_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        data, payload_is_dict = payload["data"], True
    elif isinstance(payload, list):
        data, payload_is_dict = payload, False
    else:
        raise ValueError("policy_stage file must be a list OR a dict with key 'data' as a list")

    cprint("[exec] running ground-truth unit tests ...", "yellow")
    data = execute_unit_tests_on_key(
        data, "test_input", "test_output", "execution_result", "correctness",
        max_procs=max_procs, hard_timeout_s=hard_timeout_s,
    )
    cprint("[exec] running synthetic unit tests ...", "yellow")
    data = execute_unit_tests_on_key(
        data, "syn_input", "syn_output", "syn_execution_result", "syn_correctness",
        max_procs=max_procs, hard_timeout_s=hard_timeout_s,
    )

    if payload_is_dict:
        payload["data"] = data
        out_obj = payload
    else:
        out_obj = data
    atomic_write_json(policy_stage_path, out_obj)
    cprint(f"[done] wrote execution results back to: {policy_stage_path}", "cyan")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
