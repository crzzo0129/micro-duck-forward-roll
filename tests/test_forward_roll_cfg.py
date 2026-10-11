import mujoco
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
import mjlab.tasks  # loads plugin

TASK = "Mjlab-ForwardRoll-Flat-MicroDuck"


def test_forward_roll_contacts_and_model():
    cfg = load_env_cfg(TASK)
    model = cfg.scene.entities["robot"].spec_fn().compile()
    assert model.nu == 14
    assert model.nq == 21
    for name in ("left_foot_collision", "right_foot_collision"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft") >= 0
    assert {s.name for s in cfg.scene.sensors} >= {
        "head_ground_contact", "robot_ground_contact",
        "left_foot_ground_contact", "right_foot_ground_contact"}
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations
    assert cfg.terminations["nan_state"].params["sensor_names"] == ("feet_ground_contact",)


def test_eval_cannot_spawn_halfway_or_override_standing():
    cfg = load_env_cfg(TASK, play=True)
    assert cfg.events["set_roulade_state"].params["midroll_prob"] == 0
    assert cfg.events["set_roulade_state"].params["standing_prob"] == 1
    assert cfg.curriculum == {}
    assert cfg.observations["actor"].enable_corruption is False
    assert cfg.commands["twist"].ranges.lin_vel_x == (0, 0)
    assert cfg.commands["twist"].ranges.lin_vel_y == (0, 0)
    assert cfg.commands["twist"].ranges.ang_vel_z == (0, 0)
    assert type(cfg.commands["twist"]).__name__ == "ForwardRollPlayCommandCfg"
    assert cfg.events["set_roulade_state"].params["forward_vel_range"] == (0, 0)


def test_reward_sign_and_training_curriculum():
    cfg = load_env_cfg(TASK)
    assert cfg.rewards["gentle_landing"].weight > 0  # returns a negative value
    assert cfg.rewards["roulade_stand_tax"].weight > 0  # returns a negative value
    assert cfg.rewards["roulade_overspeed"].weight < 0
    assert "upright" not in cfg.rewards  # standing cannot prevent skill discovery
    assert "roulade_spawn_mix" in cfg.curriculum
    assert cfg.events["set_roulade_state"].params["midroll_prob"] > 0
    assert load_rl_cfg(TASK).logger == "tensorboard"
    assert load_rl_cfg(TASK).experiment_name == "microduck_forward_roll"


def test_observation_layout_keeps_runtime_contract():
    actor = load_env_cfg(TASK).observations["actor"].terms
    assert list(actor) == ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
                          "actions", "command", "head_command", "body_command"]
    assert actor["head_command"].params["dim"] == 4
    assert actor["body_command"].params["dim"] == 6
