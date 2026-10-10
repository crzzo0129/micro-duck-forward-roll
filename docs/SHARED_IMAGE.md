# 共用 MJlab 基础镜像

基础环境放在 `/opt/mjlab-venv`，uv 放在 `/opt/mjlab-tools`。它们独立于项目目录。基础环境采用本项目的锁定依赖（MJlab 1.3.0、Torch 2.9.1、Warp 1.12.0，Python 3.12），包含 Micro Duck 需要的 BAM 等附加库，但保存基础镜像时不安装 Micro Duck 项目代码。

## 从已配置的独立环境迁移

在云端的原仓库目录运行；root 可直接执行，普通用户需要管理员提供可写的 `/opt` 目录。原 `.venv` 保留，不移动或删除它。虚拟环境脚本包含绝对路径，不能直接搬目录。

```bash
git pull
# 系统库已经配好，不需要再加 --install-system。
bash scripts/setup_cloud.sh --shared --base-only
source /opt/mjlab-venv/activate-mjlab.sh
python -c 'import sys, torch, mjlab; print(sys.executable); print(torch.__version__, torch.cuda.is_available())'
```

脚本重新创建共用环境，复用项目 `.cache/uv` 下载缓存；本机安装过且缓存仍在的依赖通常无需重新下载，但新环境仍需要磁盘空间。uv 工具本身可能需要下载。缓存被清理、架构变化或缺少安装文件时，会补充下载。未分配 GPU 的镜像构建可增加 `--skip-gpu-check`，之后在 GPU 实例上验证。

## 保存基础镜像

上面成功后，在云平台选择保存镜像，确认 `/opt/mjlab-venv` 和 `/opt/mjlab-tools` 被包含在镜像中。部分平台只保存容器根文件系统，不包含挂载磁盘；具体范围以平台规则为准。

环境版本记录在 `/opt/mjlab-venv/share/mjlab-base/`，包括原始 `pyproject.toml`、`uv.lock` 和实际安装列表。建议镜像命名为 `mjlab-1.3.0-py312-torch2.9.1`，并记录 GPU 架构。不同架构不能直接复用已安装的 Python/CUDA 二进制文件。

保存基础镜像后再安装当前项目，避免基础镜像引用一个仅在原实例上存在的源码目录。若只想保存当前项目的完整工作镜像，也可安装项目后保存，但必须同时保存源码。

## 在基础镜像中使用 Micro Duck

启动保存的镜像，进入已拉取的项目目录：

```bash
source /opt/mjlab-venv/activate-mjlab.sh
uv pip install --python /opt/mjlab-venv/bin/python --no-deps --editable .
python scripts/cloud.py doctor
python scripts/cloud.py smoke
# smoke 成功后才启动正式训练
python scripts/cloud.py train --num-envs 1024 --run-name first_roll
```

`--no-deps` 只安装项目代码，保留基础镜像中的依赖。构建项目包时可能需要下载小型构建工具。editable 安装依赖源码目录，删除或移动仓库后需要重新安装。每个新终端先 source 上面的激活脚本。原来的 `.venv` 不会被使用；检查 `python -c 'import sys; print(sys.executable)'` 应输出 `/opt/mjlab-venv/bin/python`。

## 其他 MJlab 项目

先比较另一个项目的依赖要求与基础镜像版本。兼容时进入它的仓库，用同样的 `uv pip install --python /opt/mjlab-venv/bin/python --no-deps --editable .` 安装代码，再用 `python` 或模块入口运行。缺少项目专用库时，按项目要求另行安装并验证；`--no-deps` 不会自动补齐它们。任务插件和命令也可能冲突，同名项目的不同源码副本不能同时通过一次 editable 安装生效。

不要在共用环境中使用自动同步依赖的 `uv run` 或默认 `uv sync`：它们可能另建项目 `.venv`，或改变共用环境中的依赖。直接使用激活后的 `python`。本脚本共用模式显式指定环境并使用 `--inexact` 保留其他包，但锁文件中已有依赖仍会被调整到本项目版本，因此只在构建或有意更新镜像时运行它。[uv 环境路径说明](https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path)、[同步说明](https://docs.astral.sh/uv/concepts/projects/sync/)。

不同 MJlab/Torch/Warp 版本要求使用不同镜像版本或独立环境。本方式提供一个版本确定、可复用的起点，并不能兼容所有 MJlab 项目的依赖组合。

本地仅验证脚本语法和现有轻量检查；未在本地安装环境或训练，云端安装及 GPU 检查仍需实际执行。
