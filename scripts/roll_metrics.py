"""Explicit success criteria shared by the evaluator and negative-control tests."""

import math
from dataclasses import dataclass


@dataclass
class RollTracker:
    dt: float
    stable_steps: int = 0
    failed: bool = False
    head_seen: bool = False
    max_rotation: float = 0.0
    peak_angular_speed: float = 0.0

    def update(self, *, rotation, head_latch, height, tilt, angular_speed,
               left_contact, right_contact, finite=True, reset=False):
        self.failed |= not finite or reset
        if not finite:
            self.stable_steps = 0
            return
        self.head_seen |= bool(head_latch)
        self.max_rotation = max(self.max_rotation, rotation)
        self.peak_angular_speed = max(self.peak_angular_speed, angular_speed)
        eligible = (not self.failed and self.head_seen
                    and self.max_rotation >= math.radians(330)
                    and abs(height - 0.115) <= 0.025
                    and tilt <= math.radians(20)
                    and angular_speed <= 1.0
                    and left_contact and right_contact)
        self.stable_steps = self.stable_steps + 1 if eligible else 0

    @property
    def success(self):
        # Require stable landing at the END of the episode, not briefly in transit.
        return not self.failed and self.stable_steps * self.dt >= 0.5 - 1e-9
