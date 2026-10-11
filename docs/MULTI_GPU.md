# 多卡训练与 H100 资源利用

MJlab 1.3.0 默认选择可见 GPU 0。`cloud.py` 现在支持 `--gpu-ids`，显式转发给 MJlab；旧命令继续默认单卡。MJlab 用 torchrunx 每卡启动一个 worker，RSL-RL 通过 NCCL 同步 PPO 梯度，共同训练一个策略。不是四个互不相关的策略。[上游固定版本代码](https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/scripts/train.py)

## 四卡启动

已有训练不要直接叠加运行另一套训练。先记录当前保存的检查点，再在原终端停止当前训练，更新代码后启动新任务。更新代码不会让已经运行的进程自动变成多卡。

共用镜像用户先 `source /opt/mjlab-venv/activate-mjlab.sh`；独立环境用户先 `source .cache/cloud.env`，将下方 `python` 命令加 `uv run` 前缀。

```bash
git pull
python scripts/cloud.py doctor
# 每卡 64 个环境、5 次 PPO 更新，确认多卡链路并验证 ONNX。
python scripts/cloud.py smoke --gpu-ids 0 1 2 3
# 每卡 4096 个环境，共 16384 个环境，作为 H100 试测起点。
python scripts/cloud.py train --gpu-ids 0 1 2 3 --num-envs 4096 --run-name h100_4gpu
```

也可用 `--gpu-ids all` 使用当前进程所有可见 GPU。smoke 和 train 应使用相同写法；单卡 smoke 不授权四卡长训练。GPU 索引相对于 `CUDA_VISIBLE_DEVICES`；若实例只向 Python 暴露一张卡，`all` 也只会用一张。

`--num-envs` 是每个 worker 的环境数：四卡 × 4096 = 16384。本任务每环境每轮采样 24 步，合计每 PPO 更新 393216 个样本；并不是把 4096 平分到四卡。环境数扩大会改变 PPO 全局批量和每轮样本数；课程依然按环境步数（迭代数 × 24）推进，因此不能假定迭代数或收敛行为与单卡等价。

恢复原来单卡的训练时，用实际检查点路径替换占位符：

```bash
python scripts/cloud.py train --gpu-ids 0 1 2 3 --num-envs 4096 \
  --checkpoint logs/rsl_rl/microduck_forward_roll/<原运行目录>/model_N.pt \
  --iterations 3000 --run-name resume_4gpu
```

这保留原权重、优化器和课程进度，不是重新从头探索。更改全局采样批量可能改变后续学习曲线，需要观察。先完成四卡 smoke，再恢复。更新源码会使旧 smoke 凭据失效。

## 确认四卡实际参与

启动日志应出现 `Launching training with 4 GPUs`，以及 `device=cuda:0` 至 `cuda:3`、rank 0 至 3。各 worker 日志位于运行目录的 `torchrunx/`，主要日志和检查点由 rank 0 保存。

```bash
python -c 'import torch; print("Visible GPUs:", torch.cuda.device_count())'
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,power.draw --format=csv -l 2
```

训练稳定运行后，四卡都应有显著高于空闲的显存占用和周期性计算活动。容器中 `nvidia-smi` 的进程表可能为空，不能只用该表判断是否运行。平台汇总的 `0.63/4` 不能代替逐卡测量。

## 提高利用率，比较实际吞吐

如果目标是保持总计 1024 个环境，并尽量缩短每轮时间，不应使用“四卡 × 每卡 1024”来比较，那会把全局批量放大四倍。提供固定总量的云端对比工具：

```bash
# 保存检查点并停止现有训练后执行，避免竞争 CPU/GPU。
git pull
python scripts/benchmark_gpus.py --total-envs 1024
```

它依次测试单卡 × 1024、双卡 × 512、四卡 × 256。每组先做 64 环境/卡、5 次更新的 smoke 及 ONNX 检查，然后训练 100 轮，排除前 20 轮预热。共计 315 次短测试更新（不计导出）；只在云端手动运行该命令时执行，不会自动启动长期训练。

可加 `--checkpoint <当前检查点>`，让三组均从同一个已有策略和课程进度开始（检查点路径应满足 `cloud.py train` 的续训规则）。未指定时三组从头开始，早期接触状态的负载不一定代表已学会动作后的负载。原训练目录不会覆盖，各组有独立日志目录。

输出 JSON 在 `outputs/gpu_benchmark_*.json`。比较 `mean_wall_s_per_iteration` 和 `speedup_vs_first`；墙钟间隔包含日志/保存开销，另提供采样和 PPO 学习耗时帮助判断瓶颈。它报告本次测量最短每轮时间的配置，并不测量达到前滚翻成功率所需的时间。100 轮可能有测量波动，可用 `--iterations 200 --warmup 40` 延长测试。

工具只选择当前可见的前 N 张卡。结束后 smoke 凭据对应最后一组；如果最终选择单卡或双卡，要用对应 GPU 选择再做 smoke，然后启动正式训练。`--dry-run` 可检查计划，不导入 GPU 运行时或训练。

1024 个环境在单张 H100 上已经是较小负载，拆为四份后，启动算子、CPU 调度和 NCCL 通信占比可能增加；四卡未必比单卡快。训练过程还需要固定数量的串行物理步和 PPO 更新，无法只因闲置 GPU 数增加就线性缩短。若四卡共同训练没有收益，每卡独立训练不同种子能利用资源并增加发现技能的机会，但不会直接缩短同一个运行的每轮时间。

每卡 4096 是试测起点，不是已验证的最优配置。先检查 NaN、显存和训练日志，再尝试每卡 8192；每个规模稳定运行 100–200 轮，排除启动编译和初期预热，比较全局采样吞吐（samples/s）、每轮耗时、仿真采样/学习耗时、成功率与奖励曲线。不要只追求 GPU-Util 100%。

H100 上小策略网络和小仿真批量可能无法充分占满算力。检查 CPU 使用率和云实例的 CPU 配额；四卡需要四个仿真/训练进程，CPU 调度或 Python MDP 计算不足会让 GPU 等待。卡间通信、NaN 检查和日志也可能形成开销，必须测量后再优化。保持 NaN 保护用于首次多卡验证。

如果四卡共同训练的吞吐或收敛收益不明显，可以每卡独立跑一个不同 seed 的实验。每个实验先单独 smoke，最好用独立工作目录，避免 smoke 凭据和日志混用。四个 seed 提高实验覆盖率，不会共同更新同一个策略。

## 验证范围

本地 GPU 参数转发、单卡/多卡 smoke 隔离及已有流程测试共 12 项通过，四卡 dry-run 通过。真实 CLI 测试已添加并拦截训练启动，但本次 Windows 检查在 Warp 导入阶段超过 40 秒限时，未完成。没有在本地训练，也没有四张 H100 的吞吐数据。云端 CLI 解析、四卡 smoke、NCCL 通信、资源占用和技能效果仍需实际验证。
