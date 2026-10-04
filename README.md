# VICO-CodeAgent：面向 Code Agent 的 Agentic RL 训练系统

在 Qwen-4B 上做“执行—反馈—修正”多步 Code Agent 的强化学习，Policy / Reward / Environment 三个模型交替优化。
默认运行环境：单机 8×A100 80GB。训练用 DeepSpeed ZeRO-3 + BF16，rollout 用 vLLM 张量并行。

## 目录结构

```
VICO-CodeAgent/
├── coding_rl.py              # 训练主控：串联每个 epoch 的各阶段，支持断点续训
├── coding_eval.py            # 评测主控（多步 agent 评测）
├── configs/                  # coding_rl.yaml / coding_eval.yaml
├── accelerate_configs/       # DeepSpeed ZeRO-3 配置：默认 1 节点无 offload；带 CPU offload 的版本仅在 OOM 时使用
├── common/
│   ├── sandbox.py            # 代码沙箱：Process 隔离 + stdin/stdout 重定向 + 128 并发分块 + 30s 硬超时
│   └── safe_io.py            # JSON 原子写入 + 文件锁（state/pending/rollout 文件）
├── sample/
│   ├── build_tasks.py        # 从 state（数据集 slot）+ pending（env 新题）构建本轮任务
│   ├── llm_reward_rollout.py # Reward 模型生成合成单测
│   ├── llm_policy_rollout.py # 多步 agent rollout：执行—反馈—修正
│   └── llm_env_rollout.py    # Env 模型：新题验收 + 生成更难/更易的新题
├── reward/
│   ├── rl_aggregate_data.py  # 合并各节点结果
│   ├── llm_rl_reward.py      # 双维度 reward + 步级 advantage + 指标统计
│   └── execute.py            # 独立重判最终代码（单轮消融/重打分用）
├── train/
│   └── coding_train.py       # Policy / Reward / Env 训练：PPO Clip + K3 KL
├── data/                     # 数据下载与预处理（CodeContests_train / LiveBench-ReasonFlux）
└── 项目介绍/                  # 运行指南、项目文档、训练曲线
```

## 一个 epoch 的流程

| # | 阶段 | 脚本 | 说明 |
|---|---|---|---|
| 1 | 构建任务 | `sample/build_tasks.py` | 每个节点的 state/pending 都以 JSON 持久化 |
| 2 | 合成单测 | `sample/llm_reward_rollout.py` | 每道题生成 16 条合成单测 |
| 3 | 多步 rollout | `sample/llm_policy_rollout.py` | 每道题 32 条轨迹，每步在沙箱中执行 GT 与合成单测 |
| 4 | 环境演化 | `sample/llm_env_rollout.py` | 验收上一轮新题，按 acc 生成 harder / easier 题 |
| 5 | 聚合 | `reward/rl_aggregate_data.py` | |
| 6 | Reward / Advantage | `reward/llm_rl_reward.py` | 见下文 |
| 7 | 训练 | `train/coding_train.py` | 依次训练 policy → reward → env |

### 多步 agent（执行—反馈—修正）

- 每一步把 `execution_result` 追加到消息栈，作为下一轮的上下文。反馈里包含公开 GT 样例和合成单测的输入、期望输出、实际输出和判定结果；上下文中保留最近 `max_history_turns` 轮。
- 终止条件：
  - **成功**：公开 GT 样例全部通过。合成单测只作参考，不参与判定；设 `success_criterion: all_feedback` 可恢复严格规则。
  - **无改善**：连续 2 轮反馈分数没有提升。
  - **最大步数**：达到 `max_steps`（默认 5）。
- vLLM 引擎在各步之间一直保持加载，不重复加载模型。

### 沙箱执行

- 每个（代码，单测）组合用一个独立的 `multiprocessing.Process` 执行，通过 stdin/stdout 重定向采集输出。
- 最多同时运行 `execute.num_chunk`（128）个进程，分块调度。
- 判题按每题的 `test_time_limit` 记 TLE；进程存活超过 30 秒会被 SIGKILL，防止死循环或僵尸进程拖住整块调度。
- 执行过程中持续读取输出队列，避免大输出程序被误判为超时。

### 双维度 reward 与步级信用分配

- 每步 reward = GT 通过率（[0,1]）+ 0.3 × 合成单测通过率（[0,0.3]），总范围 [0, 1.3]。
- 用 `trajectory_uid` + `step_index` 跟踪每一步。同一道题的 32 条轨迹在相同 step 上做组内 z-score，得到 step-level advantage；已终止的轨迹用最后一步的分数参与统计。
- 每一步单独作为一条训练样本：prompt 是截至该步的消息栈，response 是该步的输出。
- Reward 模型：根据合成单测区分正确代码与错误代码的能力给奖励。
- Env 模型：延迟奖励。新题被接受 +1，难度调错方向 -0.5，policy 完全做不出记 -1，格式无效记 -1。

### 训练稳定性

- PPO Clip（`eps=0.2`）+ K3 estimator 估计的 KL penalty（`beta=0.01`）。
- 每次更新的 KL、clip 比例和 loss 写入 `coding_rl/logs/train_metrics.jsonl`。

### 训练显存

- 默认 ZeRO-3、不开 CPU offload：4B 模型的模型状态约 64GB，分到 8 卡后每卡约 8GB；开 offload 只会增加 PCIe 传输、拖慢训练。
- 长序列下真正的显存大头是 (L × 词表) 的 logits。训练时只取 response 部分的 logits（HF `logits_to_keep`），再按 `logp_chunk_size` 分块用 fp32 计算 log-softmax；已在小模型上验证与全量计算的数值和梯度一致。

### 容错与断点续训

- 所有 state / pending / rollout JSON 都用唯一临时文件、fsync、`os.replace` 原子写入，并用 `flock` 串行化“读—改—写”，修复多进程并发写入的竞态。
- Checkpoint 先写到临时目录再替换，每 30 轮另存一份快照（`policy-epoch-30` 等）。
- 每完成一个 epoch 写入 `coding_rl/progress.json`。任一阶段失败会中止，不推进进度。

## 快速开始

```bash
conda create -n vico python=3.10 -y && conda activate vico
pip install -r requirements.txt

cd data && python download_data.py --dataset CodeContests_train \
        && python download_data.py --dataset LiveBench-ReasonFlux && cd ..
# 用 data/preprocess_data.ipynb 预处理后，在 configs/coding_rl.yaml 中填写 system.* 与 model.* 路径

python coding_rl.py config=configs/coding_rl.yaml                                  # 从头训练
python coding_rl.py config=configs/coding_rl.yaml experiment.start_from_scratch=False  # 断点续训
python coding_eval.py config=configs/coding_eval.yaml                              # 评测
```

详细环境、参数和排障说明见 `项目介绍/VICO 怎么跑项目.md`。

## 输出与指标

- `coding_rl/results/metrics-{train,eval}.jsonl`：pass@1、首次执行通过率、多步修复成功率、平均交互步数、组内 reward 标准差（双维度 vs 二值）、合成单测准确率。
- `coding_rl/logs/train_metrics.jsonl`：KL、clip_frac、loss、学习率。
- `coding_rl/ckpt/`：`optimized`、`optimized_reward`、`optimized_env` 及定期快照。
