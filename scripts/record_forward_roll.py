"""Record one standing-start rollout to MP4, without a browser or viewer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path


def recording_steps(seconds: float, dt: float) -> int:
    if not math.isfinite(seconds) or not math.isfinite(dt) or seconds <= 0 or dt <= 0:
        raise ValueError("Recording duration and simulation step must be finite and positive")
    return max(1, math.ceil(seconds / dt))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/videos"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("Checkpoint file does not exist")
    try:
        recording_steps(args.seconds, 0.02)
    except ValueError as exc:
        parser.error(str(exc))
    if os.name != "nt":
        os.environ.setdefault("MUJOCO_GL", "egl")

    import torch
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.wrappers import VideoRecorder

    task = "Mjlab-ForwardRoll-Flat-MicroDuck"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    # Record one continuous attempt; do not concatenate episodes after timeout.
    cfg.terminations.pop("time_out", None)
    env = ManagerBasedRlEnv(cfg, device=args.device, render_mode="rgb_array")
    recorder = None
    closed = False
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    name = f"{args.checkpoint.stem}_{stamp}"
    video_path = args.output_dir.resolve() / f"{name}-step-0.mp4"
    try:
        steps = recording_steps(args.seconds, env.step_dt)
        recorder = VideoRecorder(env, video_folder=args.output_dir.resolve(),
            step_trigger=lambda step: step == 0, video_length=steps,
            name_prefix=name, disable_logger=False)
        agent_cfg = load_rl_cfg(task)
        wrapped = RslRlVecEnvWrapper(recorder, clip_actions=agent_cfg.clip_actions)
        runner = load_runner_cls(task)(wrapped, asdict(agent_cfg), device=args.device)
        runner.load(str(args.checkpoint.resolve()), load_cfg={"actor": True},
                    strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        obs = wrapped.get_observations()
        if obs["actor"].shape[-1] != 61:
            raise RuntimeError("Actor observation must be 61D")
        ended_early = False
        with torch.inference_mode():
            for index in range(steps):
                actions = policy(obs)
                if not torch.isfinite(actions).all():
                    raise RuntimeError("Policy returned non-finite actions")
                obs, _, dones, _ = wrapped.step(actions)
                if bool(dones[0]):
                    ended_early = True
                    break
        recorder.close()
        closed = True
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise RuntimeError("No video written; check EGL rendering and FFmpeg output")
        report = dict(checkpoint=str(args.checkpoint.resolve()), video=str(video_path),
            requested_seconds=args.seconds, simulated_seconds=(index + 1) * env.step_dt,
            seed=args.seed, standing_start_only=True, ended_early=ended_early,
            observation_noise=False, physics_randomization=True)
        video_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Video saved: {video_path}")
    finally:
        # close() also writes a partial clip if a rollout ends early.
        if not closed:
            if recorder is not None:
                recorder.close()
            else:
                env.close()


if __name__ == "__main__":
    main()
