"""GPU rollout without OpenGL, followed by a separate MuJoCo video process."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import faulthandler
import json
from pathlib import Path
import subprocess
import sys

from record_forward_roll import recording_steps


def capture_trajectory(env, wrapped, policy, steps, torch, np):
    obs = wrapped.get_observations()
    if obs["actor"].shape[-1] != 61:
        raise RuntimeError("Actor observation must be 61D")
    frames = []
    ended_early = False
    with torch.inference_mode():
        for index in range(steps):
            actions = policy(obs)
            if not torch.isfinite(actions).all():
                raise RuntimeError("Policy returned non-finite actions")
            if index == 0:
                print("[record] First GPU simulation step (OpenGL disabled)", flush=True)
            obs, _, dones, _ = wrapped.step(actions)
            qpos = env.sim.data.qpos[0].cpu().numpy().copy()
            if not np.isfinite(qpos).all():
                raise RuntimeError("Simulation returned non-finite positions")
            frames.append(qpos)
            if index == 0:
                print("[record] First GPU simulation step passed", flush=True)
            if bool(dones[0]):
                ended_early = True
                break
    return np.stack(frames), ended_early


def render_command(trace_dir: Path, video: Path, backend: str) -> list[str]:
    return [sys.executable, "-u", str(Path(__file__).with_name("render_rollout.py")),
            "--trace-dir", str(trace_dir), "--output", str(video), "--backend", backend]


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/videos"))
    parser.add_argument("--render-backend", choices=("egl", "osmesa"), default="egl")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("Checkpoint file does not exist")
    recording_steps(args.seconds, 0.02)

    import numpy as np
    import torch
    import mujoco
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    task = "Mjlab-ForwardRoll-Flat-MicroDuck"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.terminations.pop("time_out", None)
    print("[record] Building GPU simulation without an OpenGL renderer", flush=True)
    env = ManagerBasedRlEnv(cfg, device=args.device, render_mode=None)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    name = f"{args.checkpoint.stem}_{stamp}"
    trace_dir = args.output_dir.resolve() / f"{name}.trace"
    video = args.output_dir.resolve() / f"{name}.mp4"
    try:
        agent_cfg = load_rl_cfg(task)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = load_runner_cls(task)(wrapped, asdict(agent_cfg), device=args.device)
        runner.load(str(args.checkpoint.resolve()), load_cfg={"actor": True},
                    strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        qpos, ended_early = capture_trajectory(env, wrapped, policy,
            recording_steps(args.seconds, env.step_dt), torch, np)
        trace_dir.mkdir(parents=True, exist_ok=False)
        mujoco.mj_saveModel(env.sim.mj_model, str(trace_dir / "model.mjb"))
        np.savez_compressed(trace_dir / "trajectory.npz", qpos=qpos)
        viewer = cfg.viewer
        # Track the robot's actual model body ID; do not hardcode qpos offsets.
        camera_body = int(env.scene["robot"].indexing.root_body_id)
        if viewer.body_name:
            matches = [i for i in range(env.sim.mj_model.nbody)
                       if env.sim.mj_model.body(i).name.split("/")[-1] == viewer.body_name]
            if len(matches) == 1:
                camera_body = matches[0]
        report = dict(checkpoint=str(args.checkpoint.resolve()), video=str(video),
            trace_dir=str(trace_dir), requested_seconds=args.seconds,
            simulated_seconds=len(qpos) * env.step_dt, fps=1.0 / env.step_dt,
            seed=args.seed, standing_start_only=True, ended_early=ended_early,
            observation_noise=False, physics_randomization=True,
            mujoco_version=mujoco.__version__, status="trajectory_ready", render_backend=args.render_backend,
            camera=dict(body_id=camera_body, width=viewer.width, height=viewer.height,
                distance=viewer.distance, azimuth=viewer.azimuth, elevation=viewer.elevation,
                lookat=list(viewer.lookat), fovy=viewer.fovy,
                enable_shadows=viewer.enable_shadows, enable_reflections=viewer.enable_reflections))
        (trace_dir / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[record] GPU trajectory saved: {trace_dir}", flush=True)
    finally:
        env.close()

    print("[record] Starting isolated video renderer (no MJlab/Torch/Warp imports)", flush=True)
    subprocess.run(render_command(trace_dir, video, args.render_backend), check=True)
    report["status"] = "video_ready"
    (trace_dir / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    video.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Video saved: {video}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"{exc}\nIf rendering failed, reuse the saved .trace directory; no rollout is needed.") from exc
