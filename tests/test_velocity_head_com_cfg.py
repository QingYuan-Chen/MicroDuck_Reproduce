"""Head CoM randomization must not silently target a leg link."""

import re

import mujoco
import pytest

from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


@pytest.mark.parametrize("play", [False, True])
@pytest.mark.parametrize("rough", [False, True])
def test_head_com_targets_only_head_bodies(play, rough):
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    event = cfg.events["randomize_head_com"]
    patterns = event.params["asset_cfg"].body_names
    model = cfg.scene.entities["robot"].spec_fn().compile()
    matched = {
        i for i in range(1, model.nbody)
        if any(re.fullmatch(p, model.body(i).name) for p in patterns)
    }
    assert {model.body(i).name for i in matched} == {
        "neck", "neck_pitch", "yaw_roll_motion", "jaw_soft",
    }
    neck_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "neck")
    for body_id in matched:
        ancestor = body_id
        while ancestor not in (0, neck_id):
            ancestor = int(model.body_parentid[ancestor])
        assert ancestor == neck_id

    # Keep the recipe unchanged apart from the mistaken body selection.
    assert event.mode == "reset"
    assert event.params["operation"] == "add"
    assert event.params["ranges"] == (-0.003, 0.003)
    assert cfg.curriculum["head_com_range"].params["range_stages"] == [
        {"step": 0, "range": 0.003},
        {"step": 500 * 24, "range": 0.005},
        {"step": 1000 * 24, "range": 0.01},
    ]
