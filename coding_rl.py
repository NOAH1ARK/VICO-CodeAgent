# Copyright 2026 Yiming. Licensed under the Apache License, Version 2.0.
# Derived from RLAnything / Open-AgentRL (https://github.com/Gen-Verse/Open-AgentRL),
# Apache-2.0. Modified in this repository; see NOTICE for a summary of changes.
"""
VICO coding: Agentic RL driver (Policy - Environment - Reward alternating optimization).

One epoch (single node 8xA100 by default, multi-node still supported):
  1. build_tasks          : dataset slots (state) + env-generated temp tasks (pending)
  2. reward rollout       : reward model writes synthetic unit tests (vLLM TP)
  3. policy rollout       : multi-step  execute -> feedback -> fix  agent loop (vLLM TP + sandbox)
  4. env rollout          : accept/reject temp tasks, env model proposes harder/easier tasks
  5. aggregate            : merge node shards
  6. reward / advantage   : dual-dimension step reward + step-level group z-score advantage
  7. train policy -> reward -> env   (DeepSpeed ZeRO-3 + bf16, PPO clip + K3 KL)

Fault tolerance: state/pending JSON are persisted atomically; after every finished
epoch `<project>/progress.json` is written, so `experiment.resume=auto` (default)
restarts from the first unfinished epoch after a crash. Snapshots every save_every epochs.
"""
import os
import sys
import time
import shutil
import subprocess
import shlex
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.safe_io import atomic_write_json, read_json


def run_local(cmd: str, check: bool = True) -> None:
    subprocess.run(f"bash -lc {shlex.quote(cmd)}", shell=True, check=check)

def run_local_async(cmd: str) -> subprocess.Popen:
    return subprocess.Popen(f"bash -lc {shlex.quote(cmd)}", shell=True)

def run_remote(host: str, cmd: str, check: bool = True) -> None:
    ssh_cmd = f'ssh root@{host} "bash -lc {shlex.quote(cmd)}"'
    subprocess.run(ssh_cmd, shell=True, check=check)

def run_remote_async(host: str, cmd: str) -> subprocess.Popen:
    ssh_cmd = f'ssh root@{host} "bash -lc {shlex.quote(cmd)}"'
    return subprocess.Popen(ssh_cmd, shell=True)


def wait_all(procs) -> None:
    """Wait for every node; a failed stage aborts the epoch (progress.json is NOT advanced),
    so a relaunch with experiment.resume=auto re-runs this epoch from a clean state."""
    codes = [p.wait() for p in procs]
    bad = [i for i, c in enumerate(codes) if c != 0]
    if bad:
        raise RuntimeError(f"stage failed on node(s) {bad}, return codes={codes}")


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)

def begin_with(file_name: str):
    with open(file_name, "w"):
        pass


def make_init_bash(cfg) -> str:
    sc = cfg.system
    http_proxy  = sc.HTTP_PROXY
    https_proxy = sc.HTTP_PROXY
    hf_home     = sc.HF_HOME
    envs_dir    = sc.envs_dir

    lines = []
    lines.append("set -e")
    if http_proxy is not None:
        lines.append(f"echo 'export HTTP_PROXY={http_proxy}' >> ~/.bashrc")
    if https_proxy is not None:
        lines.append(f"echo 'export HTTPS_PROXY={https_proxy}' >> ~/.bashrc")
    if hf_home is not None:
        lines.append(f"echo 'export HF_HOME={hf_home}' >> ~/.bashrc")
    lines.append("")

    if envs_dir is not None:
        lines.append(f"conda config --append envs_dirs {envs_dir} || true")
        lines.append("")

    lines.append("echo 'source ~/.bashrc' >> ~/.bash_profile")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":


    def init_node(host: str):
        run_remote(host, INIT_BASH, check=False)

    def init_hosts(worker_hosts):
        for h in worker_hosts:
            if h is None:
                continue
            init_node(h)


    def env_prefix() -> str:
        return (
            "source ~/.bashrc && "
            f"source activate {env_name} && "
        )


    def build_tasks(worker_hosts, epoch, cfg, type):
        project = cfg.experiment.project
        procs = []
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR}/sample && "
                f"python build_tasks.py "
                f"config=../configs/{project}.yaml "
                f"experiment.current_epoch={epoch} "
                f"experiment.function={type} "
                f"experiment.node_index={idx}"
            )
            full_cmd = env_prefix() + body
            procs.append(run_local_async(full_cmd) if idx == 0 else run_remote_async(host, full_cmd))
        wait_all(procs)

    def policy_sample(worker_hosts, epoch, cfg, type):
        project = cfg.experiment.project
        procs = []
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR}/sample && "
                f"python llm_policy_rollout.py "
                f"config=../configs/{project}.yaml "
                f"experiment.current_epoch={epoch} "
                f"experiment.function={type} "
                f"experiment.node_index={idx}"
            )
            full_cmd = env_prefix() + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        wait_all(procs)

    def reward_sample(worker_hosts, epoch, cfg, type):
        project = cfg.experiment.project
        procs = []
        script_name = "llm_reward_rollout.py"
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR}/sample && "
                f"PYTHONUNBUFFERED=1 python -u {script_name} "
                f"config=../configs/{project}.yaml "
                f"experiment.current_epoch={epoch} "
                f"experiment.function={type} "
                f"experiment.node_index={idx}"
            )
            full_cmd = env_prefix() + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        wait_all(procs)


    def env_sample(worker_hosts, epoch, cfg, type):
        project = cfg.experiment.project
        procs = []
        script_name = "llm_env_rollout.py"
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR}/sample && "
                f"python {script_name} "
                f"config=../configs/{project}.yaml "
                f"experiment.current_epoch={epoch} "
                f"experiment.function={type} "
                f"experiment.node_index={idx}"
            )
            full_cmd = env_prefix() + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        wait_all(procs)


    def aggregate(epoch, cfg, type):
        project = cfg.experiment.project
        full_cmd = env_prefix() + (
            f"cd {BASE_DIR}/reward && "
            f"python rl_aggregate_data.py "
            f"config=../configs/{project}.yaml "
            f"experiment.function={type} "
            f"experiment.current_epoch={epoch}"
        )
        run_local(full_cmd)


    def execute(worker_hosts, epoch, cfg, type):
        project = cfg.experiment.project
        procs = []
        script_name = "execute.py"
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR}/reward && "
                f"python {script_name} "
                f"config=../configs/{project}.yaml "
                f"experiment.current_epoch={epoch} "
                f"experiment.function={type} "
                f"experiment.node_index={idx}"
            )
            full_cmd = env_prefix() + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        wait_all(procs)

    def reward(epoch, cfg, type):
        project = cfg.experiment.project
        script_name = "llm_rl_reward.py"
        full_cmd = env_prefix() + (
            f"cd {BASE_DIR}/reward && "
            f"python {script_name} "
            f"config=../configs/{project}.yaml "
            f"experiment.function={type} "
            f"experiment.current_epoch={epoch}"
        )
        run_local(full_cmd)


    def train(worker_hosts, epoch, cfg, target = None):
        project = cfg.experiment.project
        ds_file = cfg.experiment.deepspeed_file
        num_nodes = len(worker_hosts)
        master_ip = os.environ.get("MLP_WORKER_0_HOST", "127.0.0.1")
        master_port = os.environ.get("MLP_WORKER_0_PORT", "29500")
        procs = []
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR} && "
                "export DS_SKIP_CUDA_CHECK=1 && "
                "accelerate launch "
                f"--num_machines {num_nodes} "
                f"--machine_rank {idx} "
                f"--main_process_ip {master_ip} "
                f"--main_process_port {master_port} "
                f"--config_file accelerate_configs/{ds_file}.yaml "
                f"train/coding_train.py "
                f"config=configs/{project}.yaml "
                f'training.target={target} '
                f"experiment.current_epoch={epoch}"
            )
            full_cmd = env_prefix() + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
            print(f"[DISPATCH] train node {idx} → {host}")
        wait_all(procs)
        print("All train nodes finished.")


    cfg = get_config()
    INIT_BASH = make_init_bash(cfg)
    BASE_DIR = cfg.system.rl_base_dir
    env_name = cfg.system.env_name
    total_step = cfg.experiment.total_step
    project = cfg.experiment.project
    num_node = cfg.experiment.num_node
    #worker_hosts = [os.environ[f"MLP_WORKER_{i}_HOST"] for i in range(num_node)]
    if num_node <= 1:
        worker_hosts = [None]  # rank0 local placeholder
    else:
        worker_hosts = [os.environ[f"MLP_WORKER_{i}_HOST"] for i in range(num_node)]

    import time
    time.sleep(30)

    init_hosts(worker_hosts)

    import time
    time.sleep(10)

    progress_path = f"{project}/progress.json"
    resume_mode = str(OmegaConf.select(cfg, "experiment.resume", default="auto"))

    if cfg.experiment.start_from_scratch:
        os.makedirs(f"{project}/results", exist_ok=True)
        optimized = f"../{project}/ckpt/{cfg.model.optimized_name}"
        path = (
            f"{project}/results/results-rl-"
            f"{optimized.replace('/', '.')}-"
            f"{cfg.dataset.train_dataset}.txt"
        )
        begin_with(path)
        path = (
            f"{project}/results/results-eval-"
            f"{optimized.replace('/', '.')}-"
            f"{cfg.dataset.eval_dataset}.txt"
        )
        begin_with(path)

        def clear_dir(out_dir):
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
        clear_dir(f"{project}/temp_data")
        if os.path.exists(progress_path):
            os.remove(progress_path)

    epoch = int(cfg.experiment.current_epoch)
    if not cfg.experiment.start_from_scratch and resume_mode == "auto":
        prog = read_json(progress_path, default=None)
        if prog is not None:
            epoch = int(prog["last_completed_epoch"]) + 1
            print(f"[resume] progress.json found -> resume from epoch {epoch}")

    def mark_done(epoch):
        atomic_write_json(progress_path, {
            "last_completed_epoch": int(epoch),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    while epoch <= total_step:
        print(f"\n========== epoch {epoch} ==========")

        build_tasks(worker_hosts, epoch, cfg, "train")
        reward_sample(worker_hosts, epoch, cfg, "train")      # syn UTs first: used as feedback in the agent loop
        policy_sample(worker_hosts, epoch, cfg, "train")      # multi-step execute -> feedback -> fix (sandboxed)
        env_sample(worker_hosts, epoch, cfg, "train")

        aggregate(epoch, cfg, "train")
        reward(epoch, cfg, "train")

        train(worker_hosts, epoch, cfg, target="policy")
        train(worker_hosts, epoch, cfg, target="reward")
        train(worker_hosts, epoch, cfg, target="env")

        if epoch % cfg.experiment.eval_every == 0:
            build_tasks(worker_hosts, epoch, cfg, "evaluation")
            reward_sample(worker_hosts, epoch, cfg, "evaluation")
            policy_sample(worker_hosts, epoch, cfg, "evaluation")
            aggregate(epoch, cfg, "evaluation")
            reward(epoch, cfg, "evaluation")

        mark_done(epoch)
        epoch += 1
