# VICO-CodeAgent 运行指南

[返回项目首页](../README.md)

## 1. 环境与依赖

完整训练使用 Linux、Bash、Conda 与 NVIDIA GPU。默认 YAML 面向单机 8 GPU：rollout 分为两组，每组 4 卡；训练使用对应的单机 8 卡 DeepSpeed 配置。项目目标配置为 8×A100 80GB 与 4B 规模模型，实际资源需求取决于模型、上下文长度与 batch size，仓库未附完整训练的资源实测报告。

```bash
git clone https://github.com/NOAH1ARK/VICO-CodeAgent.git
cd VICO-CodeAgent
conda create -n vico python=3.10 -y
conda activate vico
python -m pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import deepspeed, vllm, transformers; print(deepspeed.__version__, vllm.__version__, transformers.__version__)"
```

依赖固定版本见 [requirements.txt](../requirements.txt)。请在目标 GPU 服务器核对 CUDA 与各组件的安装结果，并实际加载模型验证。

主控通过 Bash 启动各阶段，并执行 `source ~/.bashrc` 与 `source activate <env_name>`，需要让非交互 shell 能找到 Conda。请使用不含空格的 Linux 项目路径。

## 2. 无 GPU 检查

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -v
python -m ruff check .
```

完整测试面向 Linux / POSIX。Windows 可运行 `python -m pytest tests/test_advantage.py -v`；它仅验证奖励与 advantage，不验证进程隔离和文件锁。

## 3. 模型与配置

准备能被 Transformers 与 vLLM 加载、具有适用 chat template 的 causal LM checkpoint，并将实际绝对路径写入 [configs/coding_rl.yaml](../configs/coding_rl.yaml)：

```yaml
system:
  HTTP_PROXY: null
  HF_HOME: /path/to/hf_cache
  env_name: vico
  envs_dir: /path/to/conda/envs
  rl_base_dir: /path/to/VICO-CodeAgent

model:
  policy_model: /path/to/models/policy
  reward_model: /path/to/models/reward
  environment_model: /path/to/models/environment
  optimized_name: optimized
  optimized_reward_name: optimized_reward
  optimized_env_name: optimized_env
```

三个模型可以从同一份基础 checkpoint 初始化，训练后分别保存。仓库未提供预训练或微调权重；模型适配需先做小规模验证。训练器使用 `logits_to_keep` 等接口，不能仅凭模型参数量判断兼容性。

同步检查 `rollout.*.gpu_groups`、`evaluation.policy.gpu_groups` 和 `experiment.deepspeed_file`。改变 GPU 数量时，需同时调整推理分组和 Accelerate 配置。

## 4. 数据准备

```bash
cd data
python download_data.py --dataset CodeContests_train
python download_data.py --dataset LiveBench-ReasonFlux
```

在此目录运行 [preprocess_data.ipynb](../data/preprocess_data.ipynb)，分别处理训练和评测数据。现有 notebook 需要手动切换读写文件；它会覆盖输出文件，截取每题前 8 个测试，并设置 `test_time_limit`，运行前应检查这些选项。记录实际使用的测试数量与时限，以便后续比较。

预处理后的每条记录包含：

```json
{
  "question": "读取两个整数并输出它们的和。",
  "test_method": "stdio",
  "test_time_limit": 2,
  "test_input": ["1 2\n"],
  "test_output": ["3"]
}
```

完成后返回仓库根目录。下载的数据 JSON 默认被 `.gitignore` 排除。

## 5. 小规模验证与完整训练

先在 `configs/coding_rl.yaml` 对应的嵌套位置修改这些值，保留其他必需字段：

| 参数 | 小规模验证示例 | 原始默认值 |
|---|---|---|
| `experiment.total_step` | 2 | 300 |
| `rollout.policy.num_task` | 4 | 64 |
| `rollout.policy.num_response_per_task` | 4 | 32 |
| `rollout.policy.max_steps` | 2 | 5 |
| `rollout.policy.max_gen_length` | 512 | 2048 |

这些设置用于检查阶段能否衔接，不用于报告模型性能。如果样本全部被难度窗口过滤，可在调试时设置 `reward.first_step_acc_range: null`，并记录这个配置差异。

**各阶段子进程会重新读取 YAML。** 调整 rollout、reward、training 等参数时应直接编辑配置文件，不要假设主控命令行中的同名覆盖会自动传递到子进程。保持训练的 `experiment.project: coding_rl` 和评测的 `experiment.project: coding_eval` 与配置文件名对应。

```bash
python coding_rl.py config=configs/coding_rl.yaml
```

依次核对任务构建、单测生成、多步采样、环境演化、聚合、奖励计算和三个模型更新的输出。第二轮可用于检查环境任务的延迟奖励。确认后再恢复完整训练参数。

从头运行会清理该实验目录中的临时状态。已有实验续跑时使用：

```bash
python coding_rl.py config=configs/coding_rl.yaml experiment.start_from_scratch=False
```

恢复通过 `progress.json` 选择下一未完成 epoch，并复用已保存模型权重与任务状态。当前 checkpoint 保存模型和 tokenizer，不包含精确恢复所需的完整优化器 / 调度器状态；也不承诺轮内故障后的事务回滚。

## 6. 独立评测与结果口径

先编辑 [configs/coding_eval.yaml](../configs/coding_eval.yaml)，填写环境、Policy / Reward checkpoint 路径和 GPU 分组：

```bash
python coding_eval.py config=configs/coding_eval.yaml
```

训练中指标写入 `coding_rl/results/metrics-{train,eval}.jsonl`，独立评测使用 `coding_eval` 目录。核心字段定义见 [reward/llm_rl_reward.py](../reward/llm_rl_reward.py)：

- `pass@1`：各轨迹末步通过完整 GT 测试的比例。
- `first_exec_pass_rate`：各轨迹第一步通过完整 GT 测试的比例。
- `multi_step_repair_rate`：首次失败的轨迹中，末步通过的比例。
- `avg_steps`：轨迹平均交互步数。
- `group_reward_std_dual` / `group_reward_std_binary`：同题首步候选在两种奖励定义下的组内标准差。

完整 GT 指**当前预处理数据中保留的测试**，不等同于原始数据集的全部测试。默认公开样例通过即可停止交互，因此需要把停止原因和最终完整 GT 通过率分开分析。

报告实验时至少记录：Git commit、模型来源、数据与预处理方式、完整配置、硬件、随机种子、候选数量、温度、最大步数、token 预算和原始指标日志。仓库中的曲线暂缺这些配套材料，参见 [曲线说明](训练曲线/README.md)。

## 7. 常见排查方向

| 现象 | 优先检查 |
|---|---|
| Conda 或配置文件找不到 | Bash 初始化、环境名、绝对路径、运行目录 |
| rollout 显存不足 | 模型长度、生成长度、TP 分组和其他 GPU 进程 |
| train 显存不足 | batch size、prompt / response 长度、gradient checkpointing、DeepSpeed 配置 |
| Policy 没有训练样本 | 首步正确率窗口、组内 reward 是否相同、生成是否接近长度上限 |
| 公开样例通过但最终准确率低 | 查看完整 GT 结果，区分交互停止条件与评测条件 |
| 恢复位置不符合预期 | start_from_scratch、resume、progress.json 与 checkpoint 路径 |

执行器包含尽力而为的加固，并非完整安全边界。模型生成的代码应在隔离的容器或虚拟机中运行。
