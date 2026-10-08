"""Regression cases for the actual upstream support and head-top gates."""

from types import SimpleNamespace
import math
import torch
from mjlab_microduck.tasks import mdp


class Scene(dict):
    pass


def fake_env(supported=True, head=False, sideways=False):
    def sensor(value):
        return SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[int(value)]])))
    q = torch.tensor([[1., 0., 0., 0.]])
    if sideways:
        q = torch.tensor([[math.sqrt(0.5), math.sqrt(0.5), 0., 0.]])
    data = SimpleNamespace(root_link_ang_vel_b=torch.tensor([[0., 2., 0.]]),
                           root_link_quat_w=q, body_link_quat_w=q.unsqueeze(1))
    asset = SimpleNamespace(data=data, find_bodies=lambda name: ([0], ["jaw_soft"]))
    scene = Scene(robot=asset)
    scene.sensors = {"robot_ground_contact": sensor(supported),
                     "head_ground_contact": sensor(head)}
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=0.02,
                          common_step_counter=1, scene=scene)
    return env, asset


def test_airborne_rotation_does_not_count():
    env, asset = fake_env(supported=False)
    mdp._update_roulade_accum(env, asset)
    assert env._roulade_max.item() == 0


def test_sideways_rotation_does_not_count():
    env, asset = fake_env(sideways=True)
    mdp._update_roulade_accum(env, asset)
    assert env._roulade_max.item() == 0


def test_supported_forward_rotation_is_integrated_once_per_step():
    env, asset = fake_env()
    mdp._update_roulade_accum(env, asset)
    first = env._roulade_max.clone()
    assert first.item() > 0
    mdp._update_roulade_accum(env, asset)
    assert torch.equal(env._roulade_max, first)


def test_landing_gate_requires_head_latch():
    env, asset = fake_env()
    mdp._roulade_state(env)
    env._roulade_max[:] = 2 * math.pi
    assert mdp._roulade_completion_gate(env, 4.5, 5.8, require_head=True).item() == 0
    env._roulade_head_latch[:] = True
    assert mdp._roulade_completion_gate(env, 4.5, 5.8, require_head=True).item() == 1


def test_head_shell_contact_without_top_down_does_not_latch():
    env, asset = fake_env(head=True)
    mdp._roulade_state(env)
    env._roulade_accum[:] = math.pi / 2
    mdp._update_roulade_accum(env, asset)
    assert not env._roulade_head_latch.item()


def test_head_top_down_contact_in_roll_window_latches():
    env, asset = fake_env(head=True)
    asset.data.body_link_quat_w = torch.tensor([[[0., 0., 1., 0.]]])
    mdp._roulade_state(env)
    env._roulade_accum[:] = math.pi / 2
    mdp._update_roulade_accum(env, asset)
    assert env._roulade_head_latch.item()
