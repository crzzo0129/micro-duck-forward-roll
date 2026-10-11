"""Reproducible cloud workflow. Run from the repository with `uv run python`."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TASK = "Mjlab-ForwardRoll-Flat-MicroDuck"
EXPERIMENT = "microduck_forward_roll"
RECEIPT = ROOT / ".cache" / "smoke.json"


def fingerprint() -> str:
    digest = hashlib.sha256()
    paths = [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    paths += list((ROOT / "src").rglob("*.py"))
    paths += list((ROOT / "src").rglob("*.xml"))
    paths += list((ROOT / "src").rglob("*.stl"))
    paths += list((ROOT / "scripts").glob("*.py"))
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_info() -> dict:
    import torch
    return {
        "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name) for name in
                     ("mjlab", "torch", "mujoco", "mujoco-warp", "warp-lang", "better-actuator-models")},
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "visible_gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def run(command: list[str], dry_run: bool = False) -> None:
    print("Executing:", subprocess.list2cmdline(command), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    if sys.platform == "linux":
        env.setdefault("MUJOCO_GL", "egl")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def checkpoint_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Checkpoint not found: {path}")
    return path


def newest_checkpoint(root: Path) -> Path:
    checkpoints = list(root.glob("*/model_*.pt"))
    if not checkpoints:
        raise ValueError(f"No checkpoint found under {root}")
    newest_run = max({p.parent for p in checkpoints}, key=lambda p: p.stat().st_mtime_ns)
    return max((p for p in checkpoints if p.parent == newest_run),
               key=lambda p: int(re.fullmatch(r"model_(\d+)\.pt", p.name).group(1)))


def validate_onnx(path: Path) -> None:
    import numpy as np
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 61] or outputs[0].shape != [1, 14]:
        raise RuntimeError("ONNX contract must be [1,61] -> [1,14]")
    result = session.run(None, {inputs[0].name: np.zeros((1, 61), dtype=np.float32)})[0]
    if not np.isfinite(result).all():
        raise RuntimeError("ONNX returned non-finite actions")
    print(f"ONNX shape and finite-output check passed: {path}")


def doctor() -> dict:
    import torch
    import warp as wp
    import mjlab.tasks  # noqa: F401 — loads installed task plugins
    from mjlab.tasks.registry import load_env_cfg
    info = runtime_info()
    print(json.dumps(info, indent=2))
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Python 3.12 is required; use uv sync --frozen --python 3.12")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check GPU allocation, driver and CUDA-enabled PyTorch.")
    wp.init()
    if not wp.get_device("cuda:0").is_cuda:
        raise RuntimeError("Warp CUDA device unavailable")
    cfg = load_env_cfg(TASK)
    model = cfg.scene.entities["robot"].spec_fn().compile()
    if model.nu != 14:
        raise RuntimeError(f"Expected 14 actuators; got {model.nu}")
    required = {"head_ground_contact", "robot_ground_contact",
                "left_foot_ground_contact", "right_foot_ground_contact"}
    if not required <= {s.name for s in cfg.scene.sensors}:
        raise RuntimeError("Forward-roll contact sensors missing")
    print("Doctor passed: CUDA, Warp, registered forward-roll task and 14 actuators.")
    return info


def gpu_selection(values: list[str]) -> list[str]:
    """mjlab IDs index the GPUs visible to this process, not host physical IDs."""
    if values == ["all"]:
        return values
    if not values or any(not value.isascii() or not value.isdigit() for value in values):
        raise ValueError("--gpu-ids expects 'all' or non-negative GPU indices, e.g. 0 1 2 3")
    normalized = [str(int(value)) for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError("--gpu-ids must not contain duplicate indices")
    return normalized


def training_command(args, smoke: bool = False) -> list[str]:
    command = [sys.executable, "-m", "mjlab_microduck.train_cli", TASK,
               "--env.scene.num-envs", str(64 if smoke else args.num_envs),
               "--agent.max-iterations", str(5 if smoke else args.iterations),
               "--agent.logger", "tensorboard", "--agent.seed", str(args.seed),
               "--agent.run-name", "smoke" if smoke else args.run_name,
               "--enable-nan-guard", "True"]
    command += ["--gpu-ids", *gpu_selection(getattr(args, "gpu_ids", ["0"]))]
    if smoke:
        command += ["--agent.experiment-name", "microduck_forward_roll_smoke",
                    "--agent.save-interval", "1"]
    elif args.checkpoint:
        checkpoint = checkpoint_path(args.checkpoint)
        # mjlab matches run names against a regex beneath this experiment root.
        # Restrict to a single exact run rather than silently selecting another run.
        log_root = (ROOT / "logs/rsl_rl" / EXPERIMENT).resolve()
        if checkpoint.parent.parent != log_root:
            raise ValueError(f"For resume, put the checkpoint in {log_root}/<run-name>/ first. "
                             "play/export/evaluate can load checkpoints from any directory.")
        command += ["--agent.resume", "True", "--agent.load-run", "^" + re.escape(checkpoint.parent.name) + "$",
                    "--agent.load-checkpoint", "^" + re.escape(checkpoint.name) + "$"]
    return command


def require_smoke(gpu_ids: list[str] | None = None) -> None:
    if not RECEIPT.is_file():
        raise RuntimeError("Run `uv run python scripts/cloud.py smoke` before a long training run.")
    receipt = json.loads(RECEIPT.read_text())
    if receipt["fingerprint"] != fingerprint() or receipt["runtime"] != runtime_info():
        raise RuntimeError("Source, dependencies or GPU changed. Run smoke again first.")
    if receipt.get("gpu_ids", ["0"]) != gpu_selection(gpu_ids or ["0"]):
        raise RuntimeError("GPU selection changed. Run smoke with the same --gpu-ids first.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    for name in ("smoke", "train"):
        p = sub.add_parser(name)
        p.add_argument("--num-envs", type=int, default=1024,
                       help="Environments PER GPU in train; smoke always uses 64 per GPU")
        p.add_argument("--gpu-ids", nargs="+", default=["0"],
                       help="Visible GPU indices, e.g. 0 1 2 3, or all; default: 0")
        p.add_argument("--iterations", type=int, default=10000,
                       help="PPO updates in this invocation (additional updates when resuming)")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--run-name", default="forward_roll")
        p.add_argument("--checkpoint")
        p.add_argument("--dry-run", action="store_true")
    for name in ("play", "export", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--num-envs", type=int, default=128 if name == "evaluate" else 1)
        p.add_argument("--dry-run", action="store_true")
        if name == "play":
            p.add_argument("--video", action="store_true")
        if name == "export":
            p.add_argument("--output", default="outputs/forward_roll.onnx")
        if name == "evaluate":
            p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if hasattr(args, "gpu_ids"):
        try:
            args.gpu_ids = gpu_selection(args.gpu_ids)
        except ValueError as exc:
            parser.error(str(exc))
    if hasattr(args, "num_envs") and args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if hasattr(args, "iterations") and args.iterations < 1:
        parser.error("--iterations must be positive")
    os.chdir(ROOT)
    if args.command == "doctor":
        doctor()
    elif args.command == "smoke":
        if args.dry_run:
            run(training_command(args, smoke=True), True)
            return
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT.unlink(missing_ok=True)  # failed reruns must never retain a passing receipt
        info = doctor()
        run(training_command(args, smoke=True))
        checkpoint = newest_checkpoint(ROOT / "logs/rsl_rl/microduck_forward_roll_smoke")
        onnx = ROOT / ".cache/smoke.onnx"
        run([sys.executable, "scripts/export.py", TASK, "--checkpoint-file", str(checkpoint),
             "--num-envs", "1", "--onnx-file", str(onnx)])
        validate_onnx(onnx)
        RECEIPT.write_text(json.dumps({"fingerprint": fingerprint(), "runtime": info,
            "gpu_ids": args.gpu_ids,
            "checkpoint": str(checkpoint), "passed_at": datetime.now(timezone.utc).isoformat()}, indent=2))
        print("Smoke passed: PPO updates, NaN guard, ONNX shape and finite outputs.")
    elif args.command == "train":
        if not args.dry_run:
            require_smoke(args.gpu_ids)
        run(training_command(args), args.dry_run)
    else:
        checkpoint = checkpoint_path(args.checkpoint)
        if args.command == "play":
            command = [sys.executable, "-m", "mjlab.scripts.play", TASK,
                       "--checkpoint-file", str(checkpoint), "--num-envs", str(args.num_envs),
                       "--viewer", "viser"]
            if args.video:
                command += ["--video", "True", "--video-length", "250"]
        elif args.command == "export":
            output = Path(args.output).resolve()
            if not args.dry_run:
                output.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "scripts/export.py", TASK,
                       "--checkpoint-file", str(checkpoint), "--num-envs", "1",
                       "--onnx-file", str(output)]
        else:
            command = [sys.executable, "scripts/evaluate_forward_roll.py",
                       "--checkpoint", str(checkpoint), "--num-envs", str(args.num_envs),
                       "--seed", str(args.seed)]
        run(command, args.dry_run)
        if args.command == "export" and not args.dry_run:
            validate_onnx(output)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
