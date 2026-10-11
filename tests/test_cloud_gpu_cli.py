"""GPU parser regression: no MJlab import, simulation, CUDA, or training."""

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
from typing import Literal
from unittest import TestCase, main, skipUnless
from unittest.mock import patch
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("gpu_train_cli",
    Path(__file__).resolve().parents[1] / "src/mjlab_microduck/train_cli.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
try:
    import tyro
except ImportError:
    tyro = None


@dataclass
class GpuConfig:
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])


class GpuCliTests(TestCase):
    def test_normalization_preserves_other_flags(self):
        argv = ["task", "--gpu-ids", "0", "1", "--agent.max-iterations", "5"]
        self.assertEqual(cli.normalize_gpu_argv(argv),
                         ["task", "--gpu-ids", "[0, 1]", "--agent.max-iterations", "5"])
        self.assertEqual(cli.normalize_gpu_argv(["--gpu-ids", "0"]), ["--gpu-ids", "[0]"])
        self.assertEqual(cli.normalize_gpu_argv(["--gpu-ids=[0,1]"]), ["--gpu-ids", "[0, 1]"])

    def test_invalid_gpu_values_refused(self):
        for values in ([], ["-1"], ["0", "0"], ["[True]"], ["[]"], ["[0.5]"],
                       ["all", "0"], ["None", "0"], ["0", "--gpu-ids", "1"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                cli.normalize_gpu_argv(["--gpu-ids", *values])

    def test_wrapper_forwards_normalized_args_and_restores_argv_on_error(self):
        original = ["train", "task", "--gpu-ids", "0"]
        observed = []
        def fake_main():
            observed.extend(sys.argv)
            raise SystemExit(2)
        with patch.object(sys, "argv", original), \
             patch.dict(sys.modules, {"mjlab.scripts.train": SimpleNamespace(main=fake_main)}):
            with self.assertRaises(SystemExit):
                cli.main()
            self.assertIs(sys.argv, original)
        self.assertEqual(observed, ["train", "task", "--gpu-ids", "[0]"])

    @skipUnless(tyro is not None, "tyro unavailable; run this file in the installed environment")
    def test_real_tyro_gpu_union_with_mjlab_collection_flags(self):
        flags = (tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff,
                 tyro.conf.UsePythonSyntaxForLiteralCollections)
        for values, expected in ((["0"], [0]), (["0", "1", "2", "3"], [0, 1, 2, 3]),
                                 (["all"], "all"), (["None"], None)):
            with self.subTest(values=values):
                cfg = tyro.cli(GpuConfig, args=cli.normalize_gpu_argv(["--gpu-ids", *values]),
                               config=flags)
                self.assertEqual(cfg.gpu_ids, expected)


if __name__ == "__main__":
    main()
