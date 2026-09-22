"""Sampling-only intervention: distributions, direction signs and config isolation."""
from dataclasses import asdict, fields
from types import SimpleNamespace
from unittest.mock import patch

import unittest
import torch

from mjlab_microduck.tasks.mdp import SpecialistVelocityCommand, VelocityCommandCommandOnly
from mjlab_microduck.tasks.microduck_velocity_specialist_env_cfg import make_microduck_velocity_specialist_env_cfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg


def test_only_twist_sampling_changes():
    original = make_microduck_velocity_env_cfg()
    variant = make_microduck_velocity_specialist_env_cfg()
    a, b = asdict(original), asdict(variant)
    a["commands"].pop("twist")
    b["commands"].pop("twist")
    # Factory creates a new equivalent terrain lambda on every invocation.
    fa = a["scene"]["terrain"].pop("spec_fn")
    fb = b["scene"]["terrain"].pop("spec_fn")
    assert fa.__code__ is fb.__code__
    assert a == b
    for f in fields(original.commands["twist"]):
        assert getattr(original.commands["twist"], f.name) == getattr(variant.commands["twist"], f.name)
    variant.rewards["track_linear_velocity"].weight = 999
    assert original.rewards["track_linear_velocity"].weight == 2.


def make_sampler(n=20000):
    obj = object.__new__(SpecialistVelocityCommand)
    obj._env = SimpleNamespace(device="cpu", num_envs=n)
    obj.cfg = make_microduck_velocity_specialist_env_cfg().commands["twist"]
    obj.vel_command_b = torch.zeros(n, 3)
    obj.vel_command_w = torch.zeros(n, 3)
    for attr in ("is_standing_env", "is_heading_env", "is_world_env", "is_forward_env"):
        setattr(obj, attr, torch.ones(n, dtype=torch.bool))
    obj.sampling_bucket_counts = torch.zeros(4, dtype=torch.long)
    return obj


def parent_sample(self, ids):
    self.vel_command_b[ids] = torch.tensor([.1, .2, .3])
    self.vel_command_w[ids] = self.vel_command_b[ids]


def test_sampling_fractions_signs_and_flags():
    torch.manual_seed(71)
    obj = make_sampler()
    with patch.object(VelocityCommandCommandOnly, "_resample_command", parent_sample):
        obj._resample_command(torch.arange(obj.num_envs))
    c = obj.vel_command_b
    mixed = c[:, 1] != 0
    backward = c[:, 0] < 0
    left = (~mixed) & (c[:, 2] > 0)
    right = c[:, 2] < 0
    for mask, probability in zip((mixed, backward, left, right), obj.cfg.bucket_probabilities):
        assert abs(mask.float().mean().item() - probability) < .015
    assert torch.all((c[backward, 0] >= -.2) & (c[backward, 0] <= -.05))
    assert torch.all(c[backward, 1:] == 0)
    assert torch.all(c[left | right, :2] == 0)
    assert torch.all((c[left | right, 2].abs() >= .15) & (c[left | right, 2].abs() <= .6))
    for attr in ("is_standing_env", "is_heading_env", "is_world_env", "is_forward_env"):
        assert not getattr(obj, attr)[~mixed].any()
        assert getattr(obj, attr)[mixed].all()
    assert torch.equal(obj.vel_command_w, obj.vel_command_b)
    assert obj.sampling_bucket_counts.sum() == obj.num_envs


def test_subset_does_not_modify_other_worlds():
    obj = make_sampler(12)
    with patch.object(VelocityCommandCommandOnly, "_resample_command", parent_sample):
        obj._resample_command(torch.tensor([1, 5, 9]))
    assert torch.all(obj.vel_command_b[[0, 2, 3, 4, 6, 7, 8, 10, 11]] == 0)
    assert obj.sampling_bucket_counts.sum() == 3


def test_invalid_bucket_configuration_rejected():
    cfg = make_microduck_velocity_specialist_env_cfg().commands["twist"]
    cfg.bucket_probabilities = (.5, .5, .5, .5)
    with unittest.TestCase().assertRaises(ValueError):
        cfg.__post_init__()


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in list(globals().items()) if name.startswith("test_"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
