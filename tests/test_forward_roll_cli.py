"""Parse the exact cloud commands without building an env or training a policy."""

import sys
from unittest.mock import patch

from mjlab.scripts import train


def test_forward_roll_cli_preserves_symmetry_and_accepts_smoke_flags():
    argv = ["train", "Mjlab-ForwardRoll-Flat-MicroDuck",
            "--env.scene.num-envs", "64", "--agent.max-iterations", "5",
            "--agent.logger", "tensorboard", "--agent.seed", "42",
            "--agent.run-name", "smoke", "--enable-nan-guard", "True",
            "--agent.experiment-name", "microduck_forward_roll_smoke",
            "--agent.save-interval", "1"]
    with patch.object(sys, "argv", argv), patch.object(train, "launch_training") as launch:
        train.main()
    assert launch.call_count == 1
    cfg = launch.call_args.kwargs["args"]
    assert cfg.env.scene.num_envs == 64
    assert cfg.agent.max_iterations == 5
    assert cfg.agent.logger == "tensorboard"
    assert cfg.agent.algorithm.symmetry_cfg["use_mirror_loss"] is True
    assert cfg.enable_nan_guard is True


def test_mjlab_accepts_multi_gpu_flags_without_launching_workers():
    for values, expected in ((["all"], "all"), (["0", "1", "2", "3"], [0, 1, 2, 3])):
        argv = ["train", "Mjlab-ForwardRoll-Flat-MicroDuck",
                "--env.scene.num-envs", "4096", "--gpu-ids", *values,
                "--agent.logger", "tensorboard"]
        with patch.object(sys, "argv", argv), patch.object(train, "launch_training") as launch:
            train.main()
        cfg = launch.call_args.kwargs["args"]
        assert cfg.gpu_ids == expected
        assert cfg.env.scene.num_envs == 4096
