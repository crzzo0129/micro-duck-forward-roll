# Ubuntu 24.04 + Python 3.12 基础镜像配置

适用于云平台已分配 NVIDIA GPU、镜像未安装 MJlab / PyTorch 的情况。脚本只安装环境并检查配置，**不会自动训练**。

## 1. 先确认 GPU 由平台接入

在云端终端执行：

```bash
nvidia-smi
python3 --version
```

应能看到分配的 GPU，Python 应为 3.12。容器中的 NVIDIA 驱动通常由宿主机和云平台提供；若 `nvidia-smi` 不存在或无法访问 GPU，先检查 GPU 实例、设备挂载和容器启动配置。不要通过在容器内安装驱动来代替平台的 GPU 接入。

预编译 Torch/Warp 依赖会提供需要的 CUDA 用户态组件，通常无需额外安装完整 CUDA Toolkit 或 `nvcc`。`nvidia-smi` 顶部的 CUDA Version 表示驱动支持的版本，不表示镜像已经装了 CUDA Toolkit；最终以项目 doctor 和 smoke 的实际结果为准。[Warp 1.12 安装说明](https://nvidia.github.io/warp/v1.12/user_guide/installation.html)

## 2. 安装最少的下载工具并拉取仓库

平台终端是 root 时：

```bash
apt-get update
apt-get install -y --no-install-recommends git curl ca-certificates
git clone https://github.com/crzzo0129/micro-duck-forward-roll.git
cd micro-duck-forward-roll
```

普通用户将前两条命令加 `sudo`，后面的项目命令用普通用户执行。仓库已有时在仓库目录执行 `git pull`。把仓库放在平台的持久化磁盘上，避免实例销毁后丢失训练结果。

## 3. 配置环境

```bash
bash scripts/setup_cloud.sh --install-system
source .cache/cloud.env
```

脚本会：

- 核对 Ubuntu 24.04、Python 3.12、GPU 可见性。
- 通过 root / sudo 安装编译工具、Python venv 支持、EGL/OpenGL 动态库和 tmux。
- 使用已有 uv，或将固定版本 uv 安装到项目 `.tools/uv/`。
- 用 `uv sync --frozen` 将训练依赖装进项目 `.venv`，保留官方锁定版本和 ARM CUDA Torch 路由。
- 执行 doctor，检查 Torch CUDA、Warp 和机器人模型配置。

不向系统 Python 安装训练依赖，避免 Ubuntu 的 `externally-managed-environment` 错误。下载需要访问 PyPI、GitHub 和相应的 PyTorch 源，首次同步会下载数 GB 依赖；建议为环境、缓存及训练结果保留充足磁盘空间。网络中断后可以重跑脚本，它会复用已有缓存。

系统库已准备好或没有 sudo 权限时，可不加 `--install-system`；缺少系统库仍需平台管理员补齐。用于保存一个尚未分配 GPU 的镜像时，可加 `--skip-gpu-check`，但启动 GPU 实例后必须再运行 doctor。

## 4. 在云端测试和训练

如果要供其他 MJlab 项目复用镜像，见 [共用基础镜像与已有环境迁移](SHARED_IMAGE.md)。下面的 `uv run` 命令用于默认的项目独立环境；共用环境使用该指南中的 `python` 命令。

```bash
uv run python scripts/cloud.py smoke
# 上一条必须成功，再开始正式训练
uv run python scripts/cloud.py train --num-envs 1024 --run-name first_roll
```

Smoke 会执行 64 个环境、5 次 PPO 更新和 ONNX 导出检查，属于云端短训练。正式训练建议用平台作业功能或 tmux 运行：

```bash
tmux new -s duck
source .cache/cloud.env
uv run python scripts/cloud.py train --num-envs 1024 --run-name first_roll
# Ctrl+B，然后 D：退出终端而保持会话运行
# 重新连接后：tmux attach -t duck
```

环境数随显存调整，显存不足可改成 256 或 512。日志和检查点在 `logs/rsl_rl/`，评估结果在 `outputs/`。

## 5. 新终端与视频

每次进入仓库的新终端先执行 `source .cache/cloud.env`，以找到项目安装的 uv。uv 自动使用项目 `.venv`，不用另外手动 activate。

不录视频的训练不需要显示器或桌面环境。EGL 视频需要平台把 NVIDIA 图形库也注入容器；自管 Docker 使用 NVIDIA Container Toolkit，并在启动容器时包含 `compute,utility,graphics` 驱动能力。只有 `export MUJOCO_GL=egl` 不会补齐缺失的宿主图形库。若视频失败但 smoke 成功，可以先完成训练，再配置视频环境。[NVIDIA Container Toolkit 驱动能力说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)

本文和脚本已做本地语法检查，未在你的 Ubuntu 云镜像中实际安装或训练。云端报错时保留完整错误输出，尤其是 doctor 中的 Torch/CUDA/Warp 版本。
