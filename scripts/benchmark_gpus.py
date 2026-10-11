"""Cloud-only PPO throughput comparison at a fixed TOTAL environment count.

Each GPU topology first passes the existing smoke/export gate. --dry-run never
imports the GPU runtime or launches training. No dependency installation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import statistics
import subprocess
import sys
import time
import uuid

import cloud


def configurations(total_envs: int, counts: list[int]) -> list[dict]:
    if total_envs <= 0 or not counts or len(set(counts)) != len(counts):
        raise ValueError("Use a positive total environment count and distinct GPU counts")
    if any(count <= 0 or total_envs % count for count in counts):
        raise ValueError("Total environments must be divisible by every positive GPU count")
    return [dict(gpus=count, per_gpu_envs=total_envs // count,
                 total_envs=total_envs, gpu_ids=[str(i) for i in range(count)])
            for count in counts]


def summarize(collection: list, learning: list, warmup: int) -> dict:
    """Scalar events expose step/value/wall_time; retain only matching iterations."""
    collect = {event.step: event for event in collection if event.step >= warmup}
    learn = {event.step: event for event in learning if event.step >= warmup}
    steps = sorted(collect.keys() & learn.keys())
    if len(steps) < 5:
        raise ValueError("Not enough post-warmup timing samples; increase --iterations")
    intervals = [(collect[b].wall_time - collect[a].wall_time) / (b - a)
                 for a, b in zip(steps, steps[1:])]
    if any(value <= 0 for value in intervals):
        raise ValueError("Non-increasing timing timestamps")
    return {
        "measured_iterations": len(steps),
        "median_collection_s": statistics.median(collect[s].value for s in steps),
        "median_learning_s": statistics.median(learn[s].value for s in steps),
        # Event-to-event timing also includes logging and checkpoint overhead.
        "median_wall_s_per_iteration": statistics.median(intervals),
        "mean_wall_s_per_iteration": sum(intervals) / len(intervals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-envs", type=int, default=1024)
    parser.add_argument("--gpu-counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--checkpoint", help="Optional same starting checkpoint for every case")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        cases = configurations(args.total_envs, args.gpu_counts)
        if args.warmup < 0 or args.iterations - args.warmup < 10:
            raise ValueError("Use non-negative warmup and at least 10 measured iterations")
        checkpoint = str(cloud.checkpoint_path(args.checkpoint)) if args.checkpoint else None
    except ValueError as exc:
        parser.error(str(exc))
    if not args.dry_run:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < max(args.gpu_counts):
            parser.error("Requested GPU count exceeds CUDA GPUs visible to this process")

    token = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    report_path = cloud.ROOT / "outputs" / f"gpu_benchmark_{token}.json"
    report = {"total_envs": args.total_envs, "iterations": args.iterations,
              "warmup": args.warmup, "checkpoint": checkpoint, "results": []}
    if not args.dry_run:
        report["runtime"] = cloud.runtime_info()
    for case in cases:
        print(f"\n{case['gpus']} GPUs x {case['per_gpu_envs']} envs = {args.total_envs} total", flush=True)
        base = [sys.executable, "scripts/cloud.py"]
        cloud.run(base + ["smoke", "--gpu-ids", *case["gpu_ids"]], args.dry_run)
        run_name = f"benchmark_{token}_{case['gpus']}gpu"
        command = base + ["train", "--gpu-ids", *case["gpu_ids"],
            "--num-envs", str(case["per_gpu_envs"]), "--iterations", str(args.iterations),
            "--run-name", run_name]
        if checkpoint:
            command += ["--checkpoint", checkpoint]
        started = time.monotonic()
        cloud.run(command, args.dry_run)
        if args.dry_run:
            continue
        runs = list((cloud.ROOT / "logs/rsl_rl" / cloud.EXPERIMENT).glob(f"*_{run_name}"))
        if len(runs) != 1:
            raise ValueError(f"Expected one benchmark run, found {len(runs)}")
        # A resumed runner starts logging at the saved iteration, not zero.
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(str(runs[0]), size_guidance={"scalars": 0})
        events.Reload()
        collection = events.Scalars("Perf/collection_time")
        learning = events.Scalars("Perf/learning_time")
        start_step = min(event.step for event in collection)
        metrics = summarize(collection, learning, start_step + args.warmup)
        row = dict(case, run_dir=str(runs[0]), process_wall_s=time.monotonic() - started, **metrics)
        row["speedup_vs_first"] = (report["results"][0]["mean_wall_s_per_iteration"] /
                                   row["mean_wall_s_per_iteration"]) if report["results"] else 1.0
        report["results"].append(row)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(row, indent=2), flush=True)
    if report["results"]:
        fastest = min(report["results"], key=lambda row: row["mean_wall_s_per_iteration"])
        print(f"\nFastest measured: {fastest['gpus']} GPUs, {fastest['per_gpu_envs']} envs/GPU")
        print(f"Report: {report_path}")
        print("Throughput only; this does not measure time to learn a successful roll.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
