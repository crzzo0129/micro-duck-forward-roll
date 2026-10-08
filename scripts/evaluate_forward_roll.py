"""Headless standing-start forward-roll evaluation with CSV/JSON reports."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

from roll_metrics import RollTracker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--agent", choices=("trained", "zero"), default="trained")
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("outputs"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.num_envs < 1:
        p.error("--num-envs must be positive")
    if args.agent == "trained" and (args.checkpoint is None or not args.checkpoint.is_file()):
        p.error("A trained evaluation requires an existing --checkpoint")
    if os.name != "nt":
        os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab_microduck.tasks.mdp import _sensor_any_contact

    task = "Mjlab-ForwardRoll-Flat-MicroDuck"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    # Horizon is managed here so timeout auto-reset does not erase landing state.
    cfg.terminations.pop("time_out", None)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        agent_cfg = load_rl_cfg(task)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if args.agent == "trained":
            runner = load_runner_cls(task)(wrapped, asdict(agent_cfg), device=args.device)
            runner.load(str(args.checkpoint.resolve()), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
        else:
            policy = lambda obs: torch.zeros((args.num_envs, 14), device=args.device)
        obs = wrapped.get_observations()
        if obs["actor"].shape[-1] != 61:
            raise RuntimeError(f"Actor observation contract changed: {obs['actor'].shape}")
        trackers = [RollTracker(env.step_dt) for _ in range(args.num_envs)]
        robot = env.scene["robot"]
        steps = round(cfg.episode_length_s / env.step_dt)
        last = None
        with torch.inference_mode():
            for _ in range(steps):
                actions = policy(obs)
                obs, rewards, dones, extras = wrapped.step(actions)
                q = robot.data.root_link_quat_w
                # z component of trunk's up axis; invariant to random initial yaw.
                tilt = torch.acos((1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(-1, 1))
                height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
                omega = torch.linalg.vector_norm(robot.data.root_link_ang_vel_b, dim=-1)
                finite = (torch.isfinite(robot.data.joint_pos).all(-1)
                          & torch.isfinite(q).all(-1) & torch.isfinite(height)
                          & torch.isfinite(omega) & torch.isfinite(actions).all(-1)
                          & torch.isfinite(rewards))
                left = _sensor_any_contact(env, "left_foot_ground_contact")
                right = _sensor_any_contact(env, "right_foot_ground_contact")
                last = torch.stack((env._roulade_max, env._roulade_head_latch.float(),
                    height, tilt, omega, left.float(), right.float(), finite.float(),
                    dones.float()), dim=1).cpu().tolist()
                for tracker, values in zip(trackers, last):
                    tracker.update(rotation=values[0], head_latch=values[1],
                        height=values[2], tilt=values[3], angular_speed=values[4],
                        left_contact=values[5], right_contact=values[6],
                        finite=bool(values[7]), reset=bool(values[8]))
        rows = [{"environment": i, "success": t.success, "failed": t.failed,
                 "head_top_contact": t.head_seen,
                 "max_supported_rotation_deg": math.degrees(t.max_rotation),
                 "terminal_stable_seconds": t.stable_steps * t.dt,
                 "peak_angular_speed_rad_s": t.peak_angular_speed,
                 "final_height_m": values[2], "final_tilt_deg": math.degrees(values[3])}
                for i, (t, values) in enumerate(zip(trackers, last))]
        report = {"task": task, "agent": args.agent, "seed": args.seed,
                  "num_envs": args.num_envs, "seconds": cfg.episode_length_s,
                  "standing_start_only": True, "observation_noise": False,
                  "physics_randomization": True,
                  "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                  "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
                      if args.checkpoint else None,
                  "success_rate": sum(r["success"] for r in rows) / len(rows),
                  "head_top_contact_rate": sum(t.head_seen for t in trackers) / len(trackers),
                  "failed_rate": sum(t.failed for t in trackers) / len(trackers),
                  "criteria": {"rotation_deg_min": 330, "head_top_latch_required": True,
                      "height_m": 0.115, "height_tolerance_m": 0.025,
                      "tilt_deg_max": 20, "angular_speed_rad_s_max": 1,
                      "both_feet_contact": True, "terminal_hold_s_min": 0.5}}
        args.output.mkdir(parents=True, exist_ok=True)
        stem = "evaluation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        (args.output / f"{stem}.json").write_text(json.dumps(report, indent=2))
        with (args.output / f"{stem}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(report, indent=2))
        print(f"Reports saved to {args.output / stem}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
