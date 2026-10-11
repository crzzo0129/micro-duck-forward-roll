"""Standard-library tests: checkpoint choice, resume quoting, skill criteria."""

import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import cloud
from roll_metrics import RollTracker


class WorkflowTests(unittest.TestCase):
    def test_gpu_selection_rejects_ambiguous_or_invalid_ids(self):
        for values in (["all", "0"], ["-1"], ["0", "00"], [], ["bad"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                cloud.gpu_selection(values)
        self.assertEqual(cloud.gpu_selection(["00", "1"]), ["0", "1"])

    def test_four_gpu_training_and_smoke_keep_per_gpu_environment_count(self):
        args = SimpleNamespace(num_envs=4096, iterations=1000, seed=42,
            run_name="four_gpu", checkpoint=None, gpu_ids=["0", "1", "2", "3"])
        train = cloud.training_command(args)
        smoke = cloud.training_command(args, smoke=True)
        for command in (train, smoke):
            start = command.index("--gpu-ids")
            self.assertEqual(command[start + 1:start + 5], ["0", "1", "2", "3"])
        self.assertEqual(train[train.index("--env.scene.num-envs") + 1], "4096")
        self.assertEqual(smoke[smoke.index("--env.scene.num-envs") + 1], "64")
        self.assertEqual(smoke[smoke.index("--agent.max-iterations") + 1], "5")

    def test_all_gpu_selection_and_single_gpu_default_forwarded(self):
        args = SimpleNamespace(num_envs=1024, iterations=10, seed=42,
            run_name="test", checkpoint=None)
        self.assertEqual(cloud.training_command(args)[-2:], ["--gpu-ids", "0"])
        args.gpu_ids = ["all"]
        self.assertEqual(cloud.training_command(args)[-2:], ["--gpu-ids", "all"])

    def test_single_gpu_smoke_does_not_authorize_multi_gpu_training(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "smoke.json"
            receipt.write_text('{"fingerprint":"test","runtime":{},"gpu_ids":["0"]}')
            with patch.object(cloud, "RECEIPT", receipt), \
                 patch.object(cloud, "fingerprint", return_value="test"), \
                 patch.object(cloud, "runtime_info", return_value={}):
                cloud.require_smoke(["0"])
                with self.assertRaisesRegex(RuntimeError, "GPU selection changed"):
                    cloud.require_smoke(["0", "1", "2", "3"])

    def test_checkpoint_iteration_sorting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run").mkdir()
            for n in (9, 100, 10):
                (root / "run" / f"model_{n}.pt").write_bytes(b"test")
            self.assertEqual(cloud.newest_checkpoint(root).name, "model_100.pt")

    def test_resume_uses_exact_run_and_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "logs/rsl_rl/microduck_forward_roll/run.v1/model_3000.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"test")
            args = SimpleNamespace(num_envs=1024, iterations=500, seed=42,
                run_name="resume", checkpoint=str(checkpoint))
            with patch.object(cloud, "ROOT", Path(directory)):
                cmd = cloud.training_command(args)
            self.assertEqual(cmd[cmd.index("--agent.load-run") + 1], r"^run\.v1$")
            self.assertEqual(cmd[cmd.index("--agent.load-checkpoint") + 1], r"^model_3000\.pt$")
            self.assertEqual(cmd[cmd.index("--agent.logger") + 1], "tensorboard")

    def test_missing_checkpoint_refused(self):
        with self.assertRaises(ValueError):
            cloud.checkpoint_path("/nonexistent/microduck_model.pt")


class SuccessTests(unittest.TestCase):
    def update(self, tracker, **changes):
        state = dict(rotation=2 * math.pi, head_latch=True, height=0.115,
                     tilt=0, angular_speed=0.1, left_contact=True, right_contact=True)
        state.update(changes)
        tracker.update(**state)

    def test_no_motion_cannot_pass(self):
        t = RollTracker(0.02)
        for _ in range(250):
            self.update(t, rotation=0, head_latch=False)
        self.assertFalse(t.success)

    def test_air_flip_or_shoulder_without_head_cannot_pass(self):
        t = RollTracker(0.02)
        for _ in range(250):
            self.update(t, head_latch=False)
        self.assertFalse(t.success)

    def test_both_feet_required(self):
        t = RollTracker(0.02)
        for _ in range(250):
            self.update(t, right_contact=False)
        self.assertFalse(t.success)

    def test_must_hold_and_remain_stable_at_end(self):
        t = RollTracker(0.02)
        for _ in range(24):
            self.update(t)
        self.assertFalse(t.success)
        self.update(t)
        self.assertTrue(t.success)
        self.update(t, tilt=math.radians(45))
        self.assertFalse(t.success)

    def test_nan_or_reset_is_permanent_failure(self):
        for changes in (dict(finite=False), dict(reset=True)):
            t = RollTracker(0.02)
            self.update(t, **changes)
            for _ in range(250):
                self.update(t)
            self.assertFalse(t.success)


if __name__ == "__main__":
    unittest.main()
