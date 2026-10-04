# -*- coding: utf-8 -*-
# Copyright 2026 Yiming. Licensed under the Apache License, Version 2.0.
"""
Stage 0 of every epoch: build this round's coding tasks.

Moved out of llm_policy_rollout.py because the multi-step policy rollout now
needs the synthetic unit tests of the reward model as execution feedback, so
the order of one epoch becomes

    build_tasks -> reward rollout (syn UTs) -> policy multi-step rollout
    (execute / feedback / fix, sandboxed) -> env rollout -> aggregate
    -> reward & advantage -> train policy / reward / env

Tasks = active version of dataset slots (state file) + env-generated "temp"
tasks waiting for acceptance (pending file). Both files are JSON-persisted per
node and written atomically under a file lock (common/safe_io.py), so the
evolving dataset survives crashes and can be resumed.
"""

import os
import sys
import random
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from termcolor import cprint

from common.safe_io import atomic_write_json, file_lock, read_json


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


def get_state_paths(project_name, dataset, node_index, config):
    state_prefix = OmegaConf.select(config, "dataset.dataset_state_prefix", default="dataset_state")
    pending_prefix = OmegaConf.select(config, "dataset.env_pending_prefix", default="env_pending")
    state_path = f"../{project_name}/temp_data/{state_prefix}-{dataset}-node{node_index}.json"
    pending_path = f"../{project_name}/temp_data/{pending_prefix}-{dataset}-node{node_index}.json"
    return state_path, pending_path


def get_policy_stage_output_path(project_name, outputs_name, num_node, node_index):
    return f"../{project_name}/temp_data/outputs-{int(node_index)}-{outputs_name}.json"


def _as_list(x):
    return x if isinstance(x, list) else []


def load_or_init_state(base_dataset_path, state_path, num_node, node_index):
    """Create the per-node state file once (locked, atomic); afterwards just read it."""
    with file_lock(state_path):
        state = read_json(state_path)
        if state is not None:
            return state

        base = read_json(base_dataset_path, default=[])
        total = len(base)
        chunk_size = (total + int(num_node) - 1) // int(num_node)
        start_idx = int(node_index) * chunk_size
        end_idx = min((int(node_index) + 1) * chunk_size, total)

        slots = []
        for local_i, item in enumerate(base[start_idx:end_idx]):
            v0 = {
                "version": 0,
                "question": item.get("question", ""),
                "test_input": _as_list(item.get("test_input", [])),
                "test_output": _as_list(item.get("test_output", [])),
                "test_method": item.get("test_method", "stdio"),
                "test_time_limit": item.get("test_time_limit", 1),
                "created_step": 0,
                "source": "init",
            }
            if v0["test_method"] == "function":
                v0["test_list"] = _as_list(item.get("test_list", []))
            slots.append({"slot_id": start_idx + local_i, "active": v0, "history": [v0]})

        state = {"meta": {"dataset": os.path.basename(base_dataset_path), "node_index": int(node_index)},
                 "slots": slots}
        atomic_write_json(state_path, state)
        return state


def resolve_policy_model(config) -> str:
    project_name = config.experiment.project
    if int(config.experiment.current_epoch) == 1:
        return str(config.model.policy_model)
    return "../" + project_name + "/ckpt/" + str(config.model.optimized_name)


if __name__ == "__main__":
    config = get_config()
    project_name = config.experiment.project
    is_train = (str(config.experiment.function) == "train")
    dataset = config.dataset.train_dataset if is_train else config.dataset.eval_dataset
    num_node = int(config.experiment.num_node)
    node_index = int(config.experiment.node_index)
    policy_model = resolve_policy_model(config)
    outputs_name = ("rl-" if is_train else "eval-") + policy_model.replace("/", ".") + "-" + dataset

    base_dataset_path = "../data/" + dataset + ".json"
    state_path, pending_path = get_state_paths(project_name, dataset, node_index, config)
    state = load_or_init_state(base_dataset_path, state_path, num_node, node_index)
    slots = state["slots"]

    with file_lock(pending_path):
        pending = read_json(pending_path, default=[])
    pending = pending if isinstance(pending, list) else []
    pending_slot_ids = {p.get("slot_id") for p in pending if "slot_id" in p}

    if is_train:
        num_task = max(1, int(int(config.rollout.policy.num_task) / max(1, num_node)))
    else:
        num_task = len(slots)

    candidate_slots = [s for s in slots if s["slot_id"] not in pending_slot_ids]
    if is_train:
        random.shuffle(candidate_slots)
    normal_slots = candidate_slots[: min(num_task, len(candidate_slots))]

    data: List[Dict[str, Any]] = []
    for s in normal_slots:
        a = s["active"]
        item = {
            "source": "dataset",
            "slot_id": s["slot_id"],
            "active_version": a.get("version", 0),
            "question": a.get("question", ""),
            "test_input": _as_list(a.get("test_input", [])),
            "test_output": _as_list(a.get("test_output", [])),
            "test_method": a.get("test_method", "stdio"),
            "test_time_limit": a.get("test_time_limit", 1),
        }
        if item["test_method"] == "function":
            item["test_list"] = _as_list(a.get("test_list", []))
        data.append(item)

    if is_train:
        for p in pending:
            data.append({
                "source": "temp",
                "slot_id": p.get("slot_id"),
                "goal": p.get("goal"),
                "prev_acc": p.get("prev_acc"),
                "created_step": p.get("created_step"),
                "env_prompt": p.get("env_prompt"),
                "env_full_output": p.get("env_full_output"),
                "env_group": p.get("env_group", []),
                "question": p.get("question", ""),
                "test_input": _as_list(p.get("test_input", [])),
                "test_output": _as_list(p.get("test_output", [])),
                "test_method": p.get("test_method", "stdio"),
                "test_time_limit": p.get("test_time_limit", 1),
            })

    for i, item in enumerate(data):
        item["task_uid"] = f"e{int(config.experiment.current_epoch)}-n{node_index}-t{i}"

    out_path = get_policy_stage_output_path(project_name, outputs_name, num_node, node_index)
    atomic_write_json(out_path, data)
    cprint(f"[build_tasks] {len(normal_slots)} dataset tasks + {len(data) - len(normal_slots)} temp tasks "
           f"-> {out_path}", "cyan")
