"""Simulation-first forward-roll experiment based on the official roulade recipe.

Keep the proven contact gates, BAM physics and reverse curriculum. Evaluation
starts from standing only; a curriculum spawn is never a full-roll success.
"""

from copy import deepcopy

from mjlab.sensor import ContactMatch, ContactSensorCfg
from . import mdp

from .microduck_roulade_env_cfg import (
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
)


def make_microduck_forward_roll_env_cfg(play: bool = False):
    cfg = make_microduck_roulade_env_cfg(play=play)
    # Match the walking recipe's contact-force NaN protection. A bad force can
    # reach critic observations before root/joint state itself becomes non-finite.
    cfg.terminations["nan_state"].params["sensor_names"] = ("feet_ground_contact",)
    for term, safe_func in (
        ("foot_contact_forces", mdp.foot_contact_forces_safe),
        ("foot_air_time", mdp.foot_air_time_safe),
    ):
        if term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[term].func = safe_func
    # Separate feet are necessary to distinguish landing from a one-foot tripod.
    cfg.scene.sensors = (*cfg.scene.sensors, *(
        ContactSensorCfg(
            name=f"{side}_foot_ground_contact",
            primary=ContactMatch(
                mode="geom", pattern=f"{side}_foot_collision", entity="robot"
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",), reduce="none", num_slots=1,
        ) for side in ("left", "right")
    ))
    if play:
        cfg.curriculum = {}
        cfg.scene.num_envs = 1
        cfg.observations["actor"].enable_corruption = False
        spawn = cfg.events["set_roulade_state"].params
        spawn.update(standing_prob=1.0, midroll_prob=0.0,
                     forward_vel_range=(0.0, 0.0))
        # The command contract stays 61D; evaluate the actual idle command.
        for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            setattr(cfg.commands["twist"].ranges, name, (0.0, 0.0))
        # Zero is outside the upstream viewer's Max slider range [0.1, 10].
        # A read-only play command preserves the idle observation contract.
        cfg.commands["twist"] = mdp.ForwardRollPlayCommandCfg(**vars(cfg.commands["twist"]))
    return cfg


MicroduckForwardRollRlCfg = deepcopy(MicroduckRouladeRlCfg)
MicroduckForwardRollRlCfg.experiment_name = "microduck_forward_roll"
MicroduckForwardRollRlCfg.run_name = "forward_roll"
MicroduckForwardRollRlCfg.logger = "tensorboard"
