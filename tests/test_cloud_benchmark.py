"""Benchmark accounting tests; no GPU, TensorBoard, simulation or training."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_gpus import configurations, summarize


class BenchmarkTests(unittest.TestCase):
    def test_total_batch_preserved_across_gpu_counts(self):
        cases = configurations(1024, [1, 2, 4])
        self.assertEqual([case["per_gpu_envs"] for case in cases], [1024, 512, 256])
        for case in cases:
            self.assertEqual(case["gpus"] * case["per_gpu_envs"], 1024)
            self.assertEqual(len(case["gpu_ids"]), case["gpus"])

    def test_invalid_or_nondivisible_gpu_counts_refused(self):
        for total, counts in ((0, [1]), (1023, [2]), (1024, [0]),
                              (1024, []), (1024, [1, 1])):
            with self.subTest(total=total, counts=counts), self.assertRaises(ValueError):
                configurations(total, counts)

    def test_warmup_excluded_and_logging_overhead_counted(self):
        # Reported compute time is 0.75 s but real event spacing is 2 s.
        collection = [SimpleNamespace(step=i, value=99 if i < 20 else 0.5,
                                      wall_time=100 + 2 * i) for i in range(30)]
        learning = [SimpleNamespace(step=i, value=99 if i < 20 else 0.25,
                                    wall_time=100 + 2 * i) for i in range(30)]
        result = summarize(collection, learning, 20)
        self.assertEqual(result["measured_iterations"], 10)
        self.assertEqual(result["median_collection_s"], 0.5)
        self.assertEqual(result["median_learning_s"], 0.25)
        self.assertEqual(result["mean_wall_s_per_iteration"], 2.0)

    def test_resumed_iterations_and_unmatched_events(self):
        collection = [SimpleNamespace(step=i, value=0.5, wall_time=2 * i)
                      for i in range(3000, 3030)]
        learning = [SimpleNamespace(step=i, value=0.25, wall_time=2 * i)
                    for i in range(3000, 3029)]
        self.assertEqual(summarize(collection, learning, 3020)["measured_iterations"], 9)
        with self.assertRaisesRegex(ValueError, "Not enough"):
            summarize(collection, learning, 3027)


if __name__ == "__main__":
    unittest.main()
