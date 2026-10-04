# VICO 运行指南

## 环境要求

### 硬件

| 资源 | 最低 | 推荐 |
|---|---|---|
| GPU | 单机 8×A100 80GB | 单机 8×A100 80GB |
| 单卡显存 | 80GB | 80GB |
| 内存 | 128GB | 256GB |
| 磁盘 | 200GB | 500GB（含模型 + checkpoint） |

### 软件

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.10 | 3.11+ 有 Ray 兼容问题 |
| PyTorch | 2.8.0 | 需配 CUDA 12.x |
| DeepSpeed | 0.16.0 | ZeRO-3 需此版本修复 all-gather 阻塞 |
| vLLM | 0.11.0 | 0.7.x 有 OOM bug，必须避开 |
| Ray | 2.49.2 | 分布式调度 |
| Transformers | 4.57.0 | Qwen-4B tokenizer 兼容 |

---

## 一、环境配置

### 1.1 创建环境

```bash
conda create -n vico python=3.10 -y
conda activate vico
```

### 1.2 安装核心框架

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install deepspeed==0.16.0
pip install transformers==4.57.0 accelerate==1.6.0
```

### 1.3 安装推理引擎

```bash
pip install vllm==0.11.0
pip install flashinfer-python==0.3.1
```

### 1.4 安装其余依赖

```bash
pip install -r requirements.txt
pip install omegaconf hydra-core wandb
```

### 1.5 验证安装

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
python -c "import vllm; print(vllm.__version__)"
python -c "import deepspeed; print(deepspeed.__version__)"
```

---

## 二、编译检查（无需 GPU）

```bash
cd VICO
python -m compileall coding_rl.py coding_eval.py train/ reward/ sample/
```

如果有语法错误这里会先报出来，不用浪费 GPU 时间。

---

## 三、模型准备

Qwen-4B 从 HuggingFace 下载（走 mirror）：

```bash
export HF_HOME=/path/to/hf_cache
huggingface-cli download Qwen/Qwen2.5-4B-Instruct --local-dir /path/to/models/qwen-4b
```

三个模型（policy/reward/env）都用同一个 Qwen-4B checkpoint 初始化，训练后各自存到 `ckpt/optimized`、`ckpt/optimized_reward`、`ckpt/optimized_env`。

---

## 四、数据准备

CodeContests 数据需要处理成 rollout 脚本能读的格式：题目描述 + 输入输出格式 + GT 单测。

```bash
# 数据放好后，在 config 里指定路径
# configs/coding_rl.yaml:
#   dataset.train_dataset: "CodeContests_train"
#   dataset.eval_dataset: "LiveBench-ReasonFlux"
```

**这里有个坑**：CodeContests 的原始格式是 Google Code Jam 竞赛格式，输入输出描述和单测分开存。需要预处理合并成一个 JSON，每项含 `description`（题目）、`public_tests`（GT 单测）、`generated_tests`（占位，环境模型训练后填充）。

---

## 五、启动训练

### 5.1 单机 smoke test（~1 小时验证管线）

先改 config 跑小规模验证：

```yaml
# configs/coding_rl.yaml 临时改：
experiment.num_node: 1
rollout.policy.num_task: 4
rollout.policy.num_response_per_task: 4
experiment.total_step: 5
rollout.policy.max_gen_length: 512
rollout.policy.max_steps: 2
```

```bash
python coding_rl.py config=configs/coding_rl.yaml
```

### 5.2 完整训练

```yaml
experiment.num_node: 1
rollout.policy.num_task: 64
rollout.policy.num_response_per_task: 32
experiment.total_step: 300
```

```bash
python coding_rl.py config=configs/coding_rl.yaml
```

### 5.3 从 checkpoint 恢复

```bash
# 自动续训：读取 coding_rl/progress.json，从第一个未完成的 epoch 继续（state/pending/ckpt 均已持久化）
python coding_rl.py config=configs/coding_rl.yaml experiment.start_from_scratch=False
# 手动指定 epoch
python coding_rl.py config=configs/coding_rl.yaml experiment.start_from_scratch=False experiment.resume=manual experiment.current_epoch=30
```

---

## 六、核心参数说明

### 数据和模型

| 参数 | 默认 | 说明 |
|---|---|---|
| model.policy_model | 必填 | 策略模型路径 |
| model.reward_model | 必填 | 奖励模型路径 |
| model.environment_model | 必填 | 环境模型路径 |
| dataset.train_dataset | CodeContests_train | 训练集 |

### RL 算法

| 参数 | 默认 | 说明 |
|---|---|---|
| training.eps | 0.20 | PPO clip 范围 |
| training.beta | 0.01 | KL penalty 系数 |
| training.use_kl_estimator_k3 | True | 用 k3 estimator |
| optimizer.params.learning_rate | 1e-6 | 学习率 |

### Agent 控制

| 参数 | 默认 | 说明 |
|---|---|---|
| rollout.policy.num_response_per_task | 32 | 每道题候选数（GRPO 组大小） |
| rollout.policy.temperature | 1.0 | 采样温度（高=探索） |
| rollout.policy.max_gen_length | 2048 | 单条候选最大 token |
| rollout.policy.max_steps | 5 | 多步 agent 最大交互步数 |
| rollout.policy.patience | 2 | 连续 N 步反馈分数无提升则终止 |
| rollout.policy.num_feedback_gt_tests | 1 | 作为公开样例反馈给 policy 的 GT 单测数 |
| reward.syn_weight | 0.3 | 合成单测奖励权重，总奖励 ∈ [0, 1.3] |
| execute.hard_timeout_s | 30 | 沙箱进程硬超时 |

### 显存和性能

| 参数 | 默认 | 说明 |
|---|---|---|
| training.mixed_precision | bf16 | 混合精度 |
| training.enable_tf32 | True | TF32 加速 matmul |
| training.batch_size_lm | 1 | 单卡 batch（显存优化后提到 4） |

---

## 七、多步任务特殊注意事项

- **vLLM 和训练要分 GPU**：config 里 `gpu_groups: [[0,1,2,3],[4,5,6,7]]`，rollout 用 0-3 号卡，训练用全部 8 张卡。rollout 完 kill vLLM 释放显存。
- **环境变量**：`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`TOKENIZERS_PARALLELISM=false` 必须在启动前设置。
- **HuggingFace 缓存**：单机环境下 HF_HOME 指向本地模型缓存即可，避免重复下载。

---

## 八、Trace 诊断

### 无 GPU 冒烟测试

```bash
# 验证 reward 计算逻辑
cd reward
python llm_rl_reward.py --help  # 看参数说明
```

### 有 GPU 的诊断

```bash
# 看 rollout 输出质量
cd sample
python llm_policy_rollout.py config=../configs/coding_rl.yaml experiment.current_epoch=1
# 检查生成的 outputs JSON
head -c 1000 ../coding_rl/temp_data/outputs-*.json
```

---

## 九、常见问题排查

| 症状 | 诊断 | 修复 |
|---|---|---|
| `CUDA out of memory` at rollout | vLLM 和训练抢显存 | rollout/train 分开跑，gpu_groups 配置正确 |
| `CUDA out of memory` at train | batch 太大或 pixel_values float32 | pixel_values cast bf16，batch 降到 1 |
| epoch 时间突然变长 | prompt 膨胀 | 检查 max_gen_length 限制，检查终止条件 |
| `No module named 'volcengine'` | 少了火山云 SDK | `pip install volcengine-python-sdk` 或注释掉 cleanup 函数 |
| pass@1 不涨 | 组内 advantage 信号弱 | 检查候选数是否 ≥2，reward 是否双维度 |
| KL divergence spike | 采样到极端 candidate | 换 k3 estimator（config 里 use_kl_estimator_k3: True） |
| trajectory 数据串扰 | Python 可变默认值 bug | 每个 sample 独立 deepcopy |
| 训练 loss NaN | advantage 除以接近 0 的 std | 检查 reward 计算逻辑，max_grad_norm clip |
| vLLM 生成乱码 | max_tokens 太小截断 | max_tokens 从 2048 加到 3072 |
| 多卡通信失败 | 卡间 NCCL 通信异常 | 检查 CUDA_VISIBLE_DEVICES 和 hostfile 配置 |
