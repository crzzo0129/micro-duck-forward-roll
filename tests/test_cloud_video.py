"""Recording and zero-command GUI regressions without simulation or dependencies."""

import ast
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import record_forward_roll as record
import cloud


class VideoTests(unittest.TestCase):
    def test_cloud_record_uses_finite_recorder_instead_of_browser(self):
        with patch.object(sys, "argv", ["cloud", "record", "--checkpoint", "mock.pt",
                                        "--seconds", "8", "--dry-run"]), \
             patch.object(cloud, "checkpoint_path", return_value=Path("mock.pt")), \
             patch.object(cloud, "run") as run:
            cloud.main()
        command, dry_run = run.call_args.args
        self.assertEqual(command[1], "scripts/record_forward_roll.py")
        self.assertEqual(command[command.index("--seconds") + 1], "8.0")
        self.assertNotIn("--viewer", command)
        self.assertTrue(dry_run)

    def test_duration_matches_50hz_and_rejects_invalid_values(self):
        self.assertEqual(record.recording_steps(5, 0.02), 250)
        self.assertEqual(record.recording_steps(0.001, 0.02), 1)
        for seconds, dt in ((0, 0.02), (-1, 0.02), (float("nan"), 0.02),
                            (float("inf"), 0.02), (5, 0)):
            with self.assertRaises(ValueError):
                record.recording_steps(seconds, dt)

    def test_play_gui_is_read_only_and_has_no_invalid_sliders(self):
        # Execute the production GUI method without importing the physics stack.
        tree = ast.parse((ROOT / "src/mjlab_microduck/tasks/mdp.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "ForwardRollPlayCommand")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "create_gui")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "mdp.py", "exec"), namespace)
        messages = []
        def forbid_slider(*args, **kwargs):
            self.fail("Fixed zero command must not create velocity sliders")
        gui = SimpleNamespace(add_folder=lambda name: nullcontext(),
                              add_markdown=messages.append, add_slider=forbid_slider)
        state = SimpleNamespace(command=[0, 0, 0])
        namespace["create_gui"](state, "twist", SimpleNamespace(gui=gui), lambda: 0)
        self.assertEqual(state.command, [0, 0, 0])
        self.assertEqual(len(messages), 1)

    def test_recording_is_bounded_stops_on_failure_and_closes_once(self):
        for stop_after in (None, 3):
            with self.subTest(stop_after=stop_after), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                checkpoint = output / "model_100.pt"
                checkpoint.write_bytes(b"mock checkpoint")
                cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=99), seed=None,
                                      terminations={"time_out": object()})
                state = SimpleNamespace(steps=0, closes=0, video=None)
                obs = {"actor": SimpleNamespace(shape=(1, 61))}
                class Env:
                    step_dt = 0.02
                    def __init__(self, cfg, device, render_mode):
                        self.render_mode = render_mode
                    def close(self):
                        state.closes += 1
                class Recorder:
                    def __init__(self, env, video_folder, name_prefix, **kwargs):
                        self.env = env
                        self.is_recording = False
                        self.path = video_folder / f"{name_prefix}-step-0.mp4"
                    def close(self):
                        self.path.write_bytes(b"mock video")
                        state.video = self.path
                        self.env.close()
                class Wrapper:
                    def __init__(self, recorder, **kwargs):
                        pass
                    def get_observations(self):
                        return obs
                    def step(self, actions):
                        state.steps += 1
                        return obs, None, [state.steps == stop_after], {}
                class Runner:
                    def __init__(self, *args, **kwargs):
                        pass
                    def load(self, *args, **kwargs):
                        pass
                    def get_inference_policy(self, **kwargs):
                        return lambda obs: object()
                @dataclass
                class RunnerCfg:
                    clip_actions: float = 1.0
                tasks = SimpleNamespace()
                modules = {"torch": SimpleNamespace(inference_mode=nullcontext,
                    isfinite=lambda actions: SimpleNamespace(all=lambda: True)),
                    "mjlab": SimpleNamespace(tasks=tasks), "mjlab.tasks": tasks,
                    "mjlab.envs": SimpleNamespace(ManagerBasedRlEnv=Env),
                    "mjlab.rl": SimpleNamespace(RslRlVecEnvWrapper=Wrapper),
                    "mjlab.tasks.registry": SimpleNamespace(load_env_cfg=lambda *a, **k: cfg,
                        load_rl_cfg=lambda *a: RunnerCfg(), load_runner_cls=lambda *a: Runner),
                    "mjlab.utils.wrappers": SimpleNamespace(VideoRecorder=Recorder)}
                argv = ["record", "--checkpoint", str(checkpoint), "--seconds", "0.1",
                        "--output-dir", str(output)]
                with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
                     patch("builtins.print"):
                    record.main()
                report = json.loads(state.video.with_suffix(".json").read_text())
                self.assertEqual(state.steps, 3 if stop_after else 5)
                self.assertEqual(state.closes, 1)
                self.assertEqual(cfg.scene.num_envs, 1)
                self.assertNotIn("time_out", cfg.terminations)
                self.assertEqual(report["ended_early"], stop_after is not None)


if __name__ == "__main__":
    unittest.main()
