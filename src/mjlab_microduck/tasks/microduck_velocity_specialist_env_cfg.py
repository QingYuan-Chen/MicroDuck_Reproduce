"""Sampling-only experiment. The original velocity task is left unchanged."""
from copy import deepcopy
from dataclasses import fields

from .microduck_velocity_env_cfg import make_microduck_velocity_env_cfg, MicroduckRlCfg
from .mdp import SpecialistVelocityCommandCfg


def make_microduck_velocity_specialist_env_cfg(play=False):
    cfg = deepcopy(make_microduck_velocity_env_cfg(play=play))
    source = cfg.commands["twist"]
    cfg.commands["twist"] = SpecialistVelocityCommandCfg(
        **{field.name: deepcopy(getattr(source, field.name)) for field in fields(source)}
    )
    return cfg


MicroduckSpecialistRlCfg = deepcopy(MicroduckRlCfg)
MicroduckSpecialistRlCfg.experiment_name = "velocity_specialist"
MicroduckSpecialistRlCfg.run_name = "velocity_specialist"
