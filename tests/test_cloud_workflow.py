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
