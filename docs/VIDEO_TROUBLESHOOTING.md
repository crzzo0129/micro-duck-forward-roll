# 云端视频中止与隔离录制

`Recording video to ...` 只表示录制开始，不代表已完成首帧或已写入 MP4。`SIGABRT: 6` 是底层中止信号，单凭信号无法判定是仿真、渲染、编码还是清理阶段。保留完整日志和 Python 原生错误堆栈。

## 先尝试隔离录制

```bash
git pull
python scripts/cloud.py record \
  --checkpoint logs/rsl_rl/microduck_forward_roll/<运行目录>/model_N.pt \
  --seconds 5 --isolated-render
```

它使用同一检查点、GPU 仿真、BAM 执行器、61D 输入、站立起步和相同种子。GPU 仿真进程不创建 OpenGL 渲染器，先保存真实 qpos 轨迹及编译后的 MuJoCo 二进制模型。独立子进程只导入 MuJoCo、NumPy 和视频工具，通过运动学重放已有姿态并生成 MP4，不重新运行 CPU 动力学。渲染采用跟随机器人身体的相机。

输出为 `outputs/videos/<名称>.mp4`、参数 JSON 和 `<名称>.trace/`。轨迹包含 `model.mjb`、`trajectory.npz`、`metadata.json`。后续渲染必须使用保存模型时同版本的 MuJoCo。该进程隔离是针对同进程 CUDA/OpenGL 问题的可尝试绕行方式，并不是已确认本次 H100 中止的根因。上游有其他 GPU/版本的类似报告：[MJlab issue #1160](https://github.com/mujocolab/mjlab/issues/1160)。

日志阶段用于定位：

- `First GPU simulation step passed`：首次纯仿真步及状态读取成功。
- `GPU trajectory saved`：仿真阶段完成，可反复渲染，无需再跑策略。
- `First frame rendered`：首次 OpenGL 读回成功。
- `Encoding ... frames`：渲染结束，开始编码。
- `Video saved`：已写入非空 MP4。

所有录制入口启用 faulthandler，原生中止时尽量输出 Python 调用堆栈。没有收到成功输出时不要把未完成文件当成有效视频。

## 只重试渲染

如果轨迹已保存而 EGL 子进程中止，先用终端打印的实际 `.trace` 路径重试：

```bash
python -u scripts/render_rollout.py \
  --trace-dir outputs/videos/<名称>.trace \
  --output outputs/videos/retry.mp4 --backend egl
```

若平台的 EGL 路径仍不可用，可在云端安装软件渲染库，再用 OSMesa 重放相同轨迹：

```bash
# root；普通用户需加 sudo。仅云端执行。
apt-get update
apt-get install -y --no-install-recommends libosmesa6
python -u scripts/render_rollout.py \
  --trace-dir outputs/videos/<名称>.trace \
  --output outputs/videos/retry_osmesa.mp4 --backend osmesa
```

OSMesa 用 CPU 渲染，可能较慢，但已保存的机器人运动不变；它不会重跑训练或把仿真改成另一套 CPU 动力学。也可在首次隔离录制时加 `--render-backend osmesa`。MuJoCo 支持 Linux 的 EGL 硬件离屏渲染和 OSMesa 软件离屏渲染：[官方说明](https://mujoco.readthedocs.io/en/3.9.0/programming/index.html)。脚本会在导入 MuJoCo 前设置后端，并使用已有 imageio-ffmpeg 的编码器，不要求重新安装 Python 依赖。

本地仅运行轻量检查与运动学 API 检查，不进行训练或云端 EGL 测试。若仍中止，保留阶段日志和 faulthandler 堆栈继续定位。
