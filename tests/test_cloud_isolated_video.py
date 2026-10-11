"""Isolated recording checks without simulation, CUDA or OpenGL."""

import ast
from contextlib import nullcontext
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cloud
from record_isolated import capture_trajectory, render_command
from render_rollout import update_pose


class Array(list):
    def cpu(self):
        return self
    def numpy(self):
        return self


class IsolatedVideoTests(unittest.TestCase):
    def test_gpu_capture_preserves_observed_poses_and_stops_on_done(self):
        for stop in (None, 2):
            env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=[Array([0, 0])])) )
            state = SimpleNamespace(index=0)
            obs = {"actor": SimpleNamespace(shape=(1, 61))}
            def step(actions):
                state.index += 1
                env.sim.data.qpos[0][:] = [state.index, state.index / 10]
                return obs, None, [state.index == stop], {}
            wrapped = SimpleNamespace(get_observations=lambda: obs, step=step)
            numeric = SimpleNamespace(isfinite=lambda values:
                SimpleNamespace(all=lambda: all(math.isfinite(v) for v in values)), stack=list)
            torch = SimpleNamespace(inference_mode=nullcontext, isfinite=numeric.isfinite)
            with patch("builtins.print"):
                poses, ended = capture_trajectory(env, wrapped, lambda obs: [0], 3, torch, numeric)
            self.assertEqual(len(poses), stop or 3)
            self.assertEqual(poses[0], [1, 0.1])
            self.assertEqual(ended, stop is not None)

    def test_renderer_runs_as_an_independent_script_with_no_gpu_framework_imports(self):
        command = render_command(Path("trace"), Path("video.mp4"), "egl")
        self.assertEqual(command[1], "-u")
        self.assertTrue(command[2].endswith("render_rollout.py"))
        tree = ast.parse((ROOT / "scripts/render_rollout.py").read_text(encoding="utf-8"))
        imports = {alias.name.split(".")[0] for node in ast.walk(tree)
                   if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse(imports & {"torch", "warp", "mjlab"})

    def test_pose_replay_does_not_step_physics(self):
        calls = []
        api = SimpleNamespace(mj_kinematics=lambda *a: calls.append("kinematics"),
            mj_comPos=lambda *a: calls.append("com"), mj_camlight=lambda *a: calls.append("camera"))
        data = SimpleNamespace(qpos=[0, 0])
        update_pose(object(), data, [1, 2], api)
        self.assertEqual(data.qpos, [1, 2])
        self.assertEqual(calls, ["kinematics", "com", "camera"])

    def test_cloud_forwards_isolation_and_render_backend(self):
        argv = ["cloud", "record", "--checkpoint", "mock.pt", "--isolated-render",
                "--render-backend", "osmesa", "--dry-run"]
        with patch.object(sys, "argv", argv), \
             patch.object(cloud, "checkpoint_path", return_value=Path("mock.pt")), \
             patch.object(cloud, "run") as run:
            cloud.main()
        command, dry_run = run.call_args.args
        self.assertEqual(command[2], "scripts/record_isolated.py")
        self.assertEqual(command[-2:], ["--render-backend", "osmesa"])
        self.assertTrue(dry_run)


if __name__ == "__main__":
    unittest.main()
