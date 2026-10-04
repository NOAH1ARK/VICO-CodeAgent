# Copyright 2026 Yiming. Licensed under the Apache License, Version 2.0.
# Derived from RLAnything / Open-AgentRL (https://github.com/Gen-Verse/Open-AgentRL),
# Apache-2.0. Modified in this repository; see NOTICE for a summary of changes.
import os
import sys
import json
from termcolor import cprint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.safe_io import atomic_write_json

from omegaconf import OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

if __name__ == "__main__":

    config = get_config()

    project_name = config.experiment.project

    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.policy_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    if config.experiment.function == "train":
        if config.dataset.environment_type == "alfworld":
            outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + config.dataset.environment_type
        else:
            outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + config.dataset.train_dataset
    else:
        if config.dataset.environment_type == "alfworld":
            outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + config.dataset.environment_type + "-" + config.dataset.alfworld_eval_type
        else:
            outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + config.dataset.eval_dataset

    num_slice = config.experiment.num_node
    final_data = []
    for i in range(num_slice):
        file_name_sub = "../" + project_name + f"/temp_data/outputs-{i}-" + outputs_name + ".json"
        with open(file_name_sub, 'r') as f:
            data = json.load(f)
        #os.remove(file_name_sub)
        final_data = final_data + data

    file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    atomic_write_json(file_name, final_data)
    cprint(f"[aggregate] {len(final_data)} tasks from {num_slice} node(s) -> {file_name}", "cyan")

    # merge env-model RL samples of all nodes (written by sample/llm_env_rollout.py)
    if config.experiment.function == "train":
        env_name = OmegaConf.select(config, "dataset.env_optimization_data", default="env_rl_data")
        env_all = []
        for i in range(num_slice):
            p = "../" + project_name + f"/temp_data/{env_name}-node{i}.json"
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    env_all.extend(json.load(f))
        atomic_write_json("../" + project_name + f"/temp_data/{env_name}.json", env_all)
        cprint(f"[aggregate] env RL samples: {len(env_all)}", "cyan")
