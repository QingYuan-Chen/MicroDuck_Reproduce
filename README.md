# MicroDuck 强化学习复现

本仓库用于学习和复现 MicroDuck 双足机器人的强化学习步态训练流程，重点理解环境建模、PPO 训练、检查点评估以及策略回放，而不是重新发布或冒充官方项目。

> [!IMPORTANT]
> 本仓库是个人学习性复现，**不是 MicroDuck 官方仓库**。基础代码取自 [`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl) 的 `main` 分支快照，基准提交为 [`1e79c29`](https://github.com/pollen-robotics/microduck_rl/commit/1e79c29c97d8b38aee9eefde77a545860ba7658e)。上游项目及其原作者保留相应权利和署名。

## 复现目标

本次复现围绕 `Mjlab-Velocity-Flat-MicroDuck` 步态任务展开：

- 理解 61 维观测、14 维动作和速度命令的组成；
- 理解 `PPO` 的采样、优势估计和策略更新流程；
- 在 MuJoCo Warp 中完成冒烟训练与长轮次训练；
- 使用固定命令评估前进、后退、横移和转向能力；
- 对比本地训练检查点与官方 ONNX 策略；
- 分析摔倒、头部晃动、速度欠跟踪和关节顶限位等现象；
- 通过受控实验验证命令采样分布对步态学习的影响。

当前仓库保存源码、配置和评估工具，不提交本地虚拟环境、训练日志、检查点或 ONNX 权重。

## 分支说明

- `main`：可复现的基础快照和已经验证的评估工具；
- `gait-training`：当前步态训练、诊断和后续实验分支。

## 环境要求

- Ubuntu Linux；
- NVIDIA GPU 和可用的 CUDA 环境；
- Python 3.12；
- [`uv`](https://docs.astral.sh/uv/)；
- 支持 MuJoCo Warp 的显卡驱动。

安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

克隆并安装依赖：

```bash
git clone https://github.com/QingYuan-Chen/MicroDuck_Reproduce.git
cd MicroDuck_Reproduce
git switch gait-training
uv sync
```

确认任务已经注册：

```bash
uv run list-envs | grep MicroDuck
```

## 启动官方策略仿真

官方模型文件不保存在本仓库中。下载官方 `alpha_walking.onnx`：

```bash
curl -L -o alpha_walking.onnx \
  https://huggingface.co/pollen-robotics/microduck-policies/resolve/v5/alpha_walking.onnx
```

启动 CPU MuJoCo 回放：

```bash
uv run scripts/infer_policy.py \
  --walking alpha_walking.onnx \
  --new-cmd-obs
```

主要控制方式：

- `↑/↓`：增减前进速度；
- `←/→`：横向速度；
- `A/E`：左转/右转；
- `Space`：速度命令归零；
- `Q`：退出。

需要注意，视觉上能够行走不代表速度跟踪准确。当前固定命令测试显示，官方 `alpha_walking.onnx` 可以稳定行走和转向，但低速区域存在较明显的响应门槛，因此评估时应同时查看实际速度，而不能只看动画。

## 冒烟训练

长时间训练前先运行 64 个并行环境、5 个迭代的冒烟测试：

```bash
WANDB_MODE=offline uv run train \
  Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max_iterations 5
```

冒烟测试主要检查：环境能否创建、61 维观测是否正确、奖励是否可计算、是否出现 `NaN`，以及 PPO 是否能完成一次采样和更新。它不能证明步态已经学会。

## 正式训练

根据显存情况选择并行环境数量，例如：

```bash
WANDB_MODE=offline uv run train \
  Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 1024 \
  --agent.max_iterations 1000
```

每个训练迭代中，每个并行环境采集 24 个控制步。训练时应重点观察：

- `Mean reward` 和 `Mean episode length`；
- 线速度、角速度跟踪误差；
- `fell_over` 终止比例；
- `action_rate_l2` 与动作平滑程度；
- 足端打滑、腾空时间和抬脚高度；
- 头部姿态误差与持续偏置；
- 关节是否长期贴近机械限位。

## 固定命令评估

仓库增加了固定命令检查脚本，用相同条件比较不同检查点：

```bash
uv run python scripts/evaluate_walk_checkpoints.py \
  --checkpoint /path/to/model_XXXX.pt \
  --suite scan \
  --push none \
  --trials 16 \
  --output logs/evaluation/model_XXXX_scan.json
```

评估结果应与 MuJoCo 可视化回放一起判断。单看总奖励容易掩盖低速欠跟踪、左右不对称、关节顶限位或头部晃动。

## 专项采样实验

`Mjlab-Velocity-Specialist-Flat-MicroDuck` 是一个隔离的采样实验，只增加后退和低速转向命令的出现概率，不改变奖励、观测、动作或执行器：

```bash
uv run train Mjlab-Velocity-Specialist-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max_iterations 5
```

这个实验用于判断问题是否来自训练数据覆盖不足，不应直接当作最终步态配置。已有短程对照表明，仅增加专项命令采样不足以稳定改善所有方向，因此后续仍需检查奖励梯度、步态启动门槛和左右对称性。

## 已确认问题

本地步态策略从 `model_250` 开始出现右侧 `hip_yaw` 贴限位，最晚到
`model_750` 已演变为两侧在站立和全部测试步态下长期顶限位，并持续到
`model_3498`。这不是官方策略的正常行为。完整时间线、评估方法和证据见
[步态策略 hip_yaw 长期顶限位诊断](docs/gait/hip_yaw_limit_diagnosis.md)。

## 目录说明

```text
src/mjlab_microduck/
├── robot/                         # MicroDuck MJCF 模型和机器人配置
├── actuator/                      # BAM 执行器模型
└── tasks/                         # 环境、奖励、命令和课程配置

scripts/
├── infer_policy.py                # CPU MuJoCo 策略回放
├── evaluate_walk_checkpoints.py   # 固定命令检查点评估
├── check_official_cpu_rehearsal.py
└── train_sampling_ablation.py     # 命令采样对照实验
```

## 测试

```bash
uv run --with pytest pytest tests/
```

专项采样测试也可以直接执行：

```bash
uv run python tests/test_velocity_specialist_sampling.py
```

## 上游项目

- [`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl)：本仓库的基础训练代码；
- [`pollen-robotics/microduck`](https://github.com/pollen-robotics/microduck)：MicroDuck 主项目和机器人运行时；
- [`mujocolab/mjlab`](https://github.com/mujocolab/mjlab)：MuJoCo Warp 强化学习框架；
- [`Rhoban/bam`](https://github.com/Rhoban/bam)：执行器物理模型。

## 维护者

本复现仓库由 [`QingYuan-Chen`](https://github.com/QingYuan-Chen) 维护。

## 许可证

软件代码沿用上游的 Apache License 2.0，详见 [LICENSE](LICENSE)。硬件设计文件沿用上游声明的 Creative Commons BY-SA-NC 许可。引用、修改或再分发时，请同时遵守对应许可证并保留上游来源说明。
