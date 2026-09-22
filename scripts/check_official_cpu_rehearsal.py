"""Headless rehearsal using infer_policy's real observation/action/BAM code.

Nominal fixed physics, no randomization/pushes: not a paired Warp comparison.
"""
import json
from pathlib import Path
import math
import numpy as np
import mujoco
from infer_policy import (PolicyInference, load_bam_model, load_mujoco_with_bam,
                          MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN)


def main():
    output = Path("logs/official_comparison_20260922/cpu_rehearsal.json")
    if output.exists():
        raise FileExistsError(output)
    conditions = {"stand": (0, 0, 0), "forward_0.1": (.1, 0, 0),
        "forward_0.3": (.3, 0, 0), "backward_0.1": (-.1, 0, 0),
        "backward_0.2": (-.2, 0, 0), "turn_left_0.3": (0, 0, .3),
        "turn_right_0.3": (0, 0, -.3)}
    rows = []
    for label, cmd in conditions.items():
        bam_model = load_bam_model(BAM_KP_FW, 7.4, None)
        model, data, bam, _ = load_mujoco_with_bam(MICRODUCK_XML, bam_model, .005, .1, BAM_VIN_MIN)
        policy = PolicyInference(model, data, walking_onnx_path="alpha_walking.onnx",
            bam_ctrl=bam, use_projected_gravity=True, new_cmd_obs=True)
        policy.set_vel_cmd(*cmd)
        free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        qadr, vadr = model.jnt_qposadr[free], model.jnt_dofadr[free]
        data.qpos[qadr:qadr+3] = [0, 0, .125]
        data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]
        data.qpos[policy.joint_qpos_indices] = policy.default_pose
        bam.reset(data.qpos)
        policy.set_position_targets(policy.default_pose)
        mujoco.mj_forward(model, data)
        values = []
        fell = False
        for step in range(1000):
            action = policy.infer()
            policy.apply_action(action)
            for _ in range(4):
                bam.update()
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            quat = data.qpos[qadr+3:qadr+7].astype(np.float32)
            gravity = policy.quat_rotate_inverse(quat, np.array([0, 0, -1], dtype=np.float32))
            tilt = math.acos(float(np.clip(-gravity[2], -1, 1)))
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("Nonfinite CPU state")
            if step >= 50:
                v = policy.quat_rotate_inverse(quat, data.qvel[vadr:vadr+3].astype(np.float32))
                values.append([float(v[0]), float(v[1]), float(data.qvel[vadr+5])])
            if tilt > math.radians(70):
                fell = True
                break
        row = dict(condition=label, command=cmd, duration_s=(step+1)*.02, fell=fell,
                   mean_velocity=np.mean(values, axis=0).tolist() if values else None)
        rows.append(row)
        print("CPU_RESULT", json.dumps(row), flush=True)
    output.write_text(json.dumps(dict(scene=MICRODUCK_XML, source="infer_policy.PolicyInference + BAM controller", vin=7.4, vin_drop_gain=.1, randomized=False, trials=rows), indent=2))


if __name__ == "__main__":
    main()
