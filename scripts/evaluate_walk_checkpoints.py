"""Headless, first-episode policy comparison. Does not train or modify checkpoints.

All metrics are measured directly, NOT reward-manager normalized statistics.
Frozen final-stage DR; fixed head/body commands; first episode only per environment.
Push intervals and amplitudes use a separate RNG, shared across checkpoints.
"""
import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

COMMANDS = {
    "stand": (0., 0., 0.),
    "forward_0.1": (.1, 0., 0.),
    "forward_0.3": (.3, 0., 0.),
    "backward_0.1": (-.1, 0., 0.),
    "lateral_0.1": (0., .1, 0.),
    "turn_left_0.3": (0., 0., .3),
    "turn_right_0.3": (0., 0., -.3),
}

SCAN_COMMANDS = {"stand": (0., 0., 0.)}
for _speed in (.05, .1, .2):
    SCAN_COMMANDS[f"forward_{_speed}"] = (_speed, 0., 0.)
    SCAN_COMMANDS[f"backward_{_speed}"] = (-_speed, 0., 0.)
for _speed in (.15, .3, .6):
    SCAN_COMMANDS[f"turn_left_{_speed}"] = (0., 0., _speed)
    SCAN_COMMANDS[f"turn_right_{_speed}"] = (0., 0., -_speed)
# Retain checks outside the specialist buckets as regression guardrails.
SCAN_COMMANDS.update(forward_0_3=(.3, 0., 0.), lateral_0_1=(0., .1, 0.))


class OnnxEvaluationPolicy:
    """Run the embedded normalizer exactly once; batch metadata only is relaxed.

    Verify batched inference against the original singleton graph before use.
    Original ONNX file is never changed.
    """
    def __init__(self, path, env):
        import onnx
        import onnxruntime as ort
        graph = onnx.load(str(path))
        metadata = {p.key: p.value for p in graph.metadata_props}
        action = env.action_manager.get_term("joint_pos")
        assert metadata["joint_names"].split(",") == action.target_names
        assert metadata["observation_names"].split(",") == list(env.observation_manager.active_terms["actor"])
        assert metadata["command_names"].split(",") == list(env.command_manager.active_terms)
        defaults = env.scene["robot"].data.default_joint_pos[0, action.target_ids].cpu().numpy()
        assert np.allclose(np.fromstring(metadata["default_joint_pos"], sep=","), defaults, atol=.0006)
        assert float(metadata["action_scale"]) == action.scale == 1.
        assert graph.graph.input[0].type.tensor_type.shape.dim[1].dim_value == 61
        assert graph.graph.output[0].type.tensor_type.shape.dim[1].dim_value == 14
        assert [n.op_type for n in graph.graph.node[:2]] == ["Sub", "Div"]
        # Only elementwise/affine nodes are supported by this batch adapter.
        assert set(n.op_type for n in graph.graph.node) <= {"Sub", "Div", "Gemm", "Elu", "Identity"}
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        original = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        for value in [*graph.graph.input, *graph.graph.output]:
            value.type.tensor_type.shape.dim[0].dim_param = "batch"
        self.session = ort.InferenceSession(graph.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = original.get_inputs()[0].name
        sample = np.random.default_rng(412).normal(0, .2, (16, 61)).astype(np.float32)
        reference = np.concatenate([original.run(None, {self.input_name: x[None]})[0] for x in sample])
        result = self.session.run(None, {self.input_name: sample})[0]
        np.testing.assert_allclose(result, reference, rtol=2e-5, atol=2e-5)
        self.audit = dict(metadata=metadata, batch_max_abs_error=float(np.max(np.abs(result-reference))),
                          normalization="embedded ONNX Sub/Div only", backend="onnxruntime CPU",
                          source_file_unchanged=True)
        print("ONNX_AUDIT", json.dumps(self.audit), flush=True)

    def __call__(self, obs):
        raw = obs["actor"]
        values = self.session.run(None, {self.input_name: raw.cpu().numpy()})[0]
        return torch.as_tensor(values, device=raw.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--push", choices=["none", "train", "play"], required=True)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["basic", "scan"], default="basic")
    args = parser.parse_args()
    command_table = COMMANDS if args.suite == "basic" else SCAN_COMMANDS
    if args.output.exists():
        raise FileExistsError(args.output)
    configure_torch_backends()
    task = "Mjlab-Velocity-Flat-MicroDuck"
    cfg = load_env_cfg(task, play=False)
    cfg.seed = args.seed
    cfg.scene.num_envs = len(command_table) * args.trials
    cfg.episode_length_s = args.seconds
    cfg.auto_reset = False
    # Fixed final-stage domain randomization; no curriculum changing the test.
    cfg.curriculum = {}
    cfg.events.pop("push_robot", None)
    cfg.events["randomize_com"].params["ranges"] = (-.015, .015)
    cfg.events["randomize_head_com"].params["ranges"] = (-.01, .01)
    twist_cfg = cfg.commands["twist"]
    for attr in ("rel_standing_envs", "rel_heading_envs", "rel_world_envs",
                 "rel_forward_envs", "rel_turn_in_place_envs", "init_velocity_prob"):
        setattr(twist_cfg, attr, 0.)
    twist_cfg.heading_command = False
    twist_cfg.ranges.heading = None
    for name, dims in (("head_pose", 4), ("body_pose", 6)):
        cfg.commands[name].ranges = tuple((0., 0.) for _ in range(dims))
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        commands = torch.tensor(list(command_table.values()), device=env.device).repeat_interleave(args.trials, 0)
        twist = env.command_manager.get_term("twist")

        def fixed_resample(self, ids):
            self.vel_command_b[ids] = commands[ids]
            self.vel_command_w[ids] = commands[ids]
            self.is_standing_env[ids] = False
            self.is_heading_env[ids] = False
            self.is_world_env[ids] = False
            self.is_forward_env[ids] = False

        twist._resample_command = MethodType(fixed_resample, twist)
        wrapper = RslRlVecEnvWrapper(env)
        runner = load_runner_cls(task)(wrapper, asdict(load_rl_cfg(task)), device="cuda:0")
        if args.checkpoint.suffix == ".onnx":
            policy = OnnxEvaluationPolicy(args.checkpoint, env)
        else:
            runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location="cuda:0")
            policy = runner.get_inference_policy(device="cuda:0")
        obs, _ = wrapper.reset()
        robot = env.scene["robot"]
        ids, names = robot.find_joints_by_actuator_names((r".*neck.*", r".*head.*"))
        assert len(ids) == 4, names
        action_term = env.action_manager.get_term("joint_pos")
        servo_ids = action_term.target_ids
        joint_names = action_term.target_names
        n = env.num_envs
        steps = round(args.seconds / env.step_dt)
        alive = torch.ones(n, dtype=torch.bool, device=env.device)
        duration = torch.full((n,), args.seconds, device=env.device)
        outcome = ["horizon"] * n
        # Metrics exclude the first second and samples after the first terminal.
        sums = torch.zeros((n, 8), device=env.device)
        head_sum = torch.zeros((n, 4), device=env.device)
        head_sq = torch.zeros_like(head_sum)
        counts = torch.zeros(n, device=env.device)
        joint_error_sum = torch.zeros((n, len(joint_names)), device=env.device)
        joint_soft_hits = torch.zeros_like(joint_error_sum)
        joint_hard_hits = torch.zeros_like(joint_error_sum)
        joint_q_sum = torch.zeros_like(joint_error_sum)
        joint_target_sum = torch.zeros_like(joint_error_sum)
        prev_action = prev_prev_action = None
        rng = np.random.default_rng(args.seed + 7)
        interval = {"train": (3., 6.), "play": (.5, 1.), "none": (1e6, 1e6)}[args.push]
        next_push = rng.uniform(*interval, n)
        pushes = np.zeros(n, dtype=int)
        initial_qpos = robot.data.joint_pos.detach().cpu().tolist()
        with torch.inference_mode():
            for step in range(steps):
                action = policy(obs)
                if not torch.isfinite(action).all():
                    raise RuntimeError("Nonfinite policy action")
                assert torch.allclose(env.command_manager.get_command("twist"), commands)
                obs_dict, _, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                data = robot.data
                head = data.joint_pos[:, ids] - data.default_joint_pos[:, ids]
                tilt = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1., 1.))
                vel = data.root_link_lin_vel_b
                yaw = data.root_link_ang_vel_b[:, 2]
                acc = torch.zeros(n, device=env.device) if prev_prev_action is None else (action - 2 * prev_action + prev_prev_action).abs().mean(-1)
                values = torch.stack(((vel[:, :2] - commands[:, :2]).norm(dim=-1),
                    (yaw - commands[:, 2]).abs(), vel[:, 0], vel[:, 1], yaw,
                    tilt, data.joint_vel[:, ids].square().mean(-1), acc), -1)
                finite = torch.isfinite(values).all(-1) & torch.isfinite(head).all(-1)
                if torch.any(alive & ~finite):
                    raise RuntimeError("Nonfinite state in a scored episode")
                valid = alive & (step >= round(1. / env.step_dt))
                sums[valid] += values[valid]
                head_sum[valid] += head[valid]
                head_sq[valid] += head[valid].square()
                counts[valid] += 1
                q = data.joint_pos[:, servo_ids]
                limits = data.joint_pos_limits[:, servo_ids]
                soft = data.soft_joint_pos_limits[:, servo_ids]
                margin = .02 * (limits[..., 1] - limits[..., 0])
                joint_error_sum[valid] += (data.joint_pos_target[:, servo_ids] - q).abs()[valid]
                joint_q_sum[valid] += q[valid]
                joint_target_sum[valid] += data.joint_pos_target[:, servo_ids][valid]
                joint_soft_hits[valid] += ((q < soft[..., 0]) | (q > soft[..., 1]))[valid]
                joint_hard_hits[valid] += ((q < limits[..., 0] + margin) | (q > limits[..., 1] - margin))[valid]
                ended = alive & done
                for idx in ended.nonzero().flatten().tolist():
                    duration[idx] = (step + 1) * env.step_dt
                    outcome[idx] = "fall" if env.termination_manager.get_term("fell_over")[idx] else ("nan" if env.termination_manager.get_term("nan_state")[idx] else ("horizon" if truncated[idx] else "other"))
                alive &= ~done
                # Reset terminated worlds only to keep the simulator valid; never score them again.
                reset_ids = done.nonzero().flatten()
                if len(reset_ids):
                    obs_dict, _ = env.reset(env_ids=reset_ids)
                # Matched external push schedule; same additive world XY velocity change as training.
                hit = np.flatnonzero(next_push <= (step + 1) * env.step_dt)
                if len(hit):
                    deltas = rng.uniform(-.3, .3, (len(hit), 2))
                    next_push[hit] += rng.uniform(*interval, len(hit))
                    active_hit = alive[torch.as_tensor(hit, device=env.device)].cpu().numpy()
                    hit, deltas = hit[active_hit], deltas[active_hit]
                    if len(hit):
                        ti = torch.as_tensor(hit, device=env.device)
                        velocity = data.root_link_vel_w[ti].clone()
                        velocity[:, :2] += torch.tensor(deltas, device=env.device, dtype=velocity.dtype)
                        robot.write_root_link_velocity_to_sim(velocity, env_ids=ti)
                        pushes[hit] += 1
                        env.sim.forward()
                        env.sim.sense()
                        obs_dict = env.observation_manager.compute(update_history=False)
                obs = TensorDict(obs_dict, batch_size=[n])
                prev_prev_action, prev_action = prev_action, action.clone()
                if (step + 1) % 250 == 0:
                    print(f"EVAL {args.checkpoint.stem} {args.push} {step+1}/{steps} alive={alive.sum().item()}/{n}", flush=True)
        count = counts.clamp(min=1)
        averages = sums / count[:, None]
        head_bias = head_sum / count[:, None]
        head_variance = (head_sq / count[:, None] - head_bias.square()).clamp(min=0)
        rows = []
        for i in range(n):
            v = averages[i].cpu().tolist()
            rows.append(dict(condition=list(command_table)[i // args.trials], trial=i % args.trials,
                outcome=outcome[i], duration_s=duration[i].item(), metric_steps=int(counts[i]),
                xy_error_m_s=v[0], yaw_error_rad_s=v[1], actual_vx=v[2], actual_vy=v[3], actual_yaw=v[4],
                tilt_deg=math.degrees(v[5]), head_speed_rms_rad_s=math.sqrt(v[6]), action_second_diff=v[7],
                head_bias_deg=(head_bias[i] * 180/math.pi).cpu().tolist(),
                head_temporal_std_deg=(head_variance[i].sqrt() * 180/math.pi).cpu().tolist(), pushes=int(pushes[i]),
                joint_target_mae_rad=(joint_error_sum[i] / count[i]).cpu().tolist(),
                joint_mean_rad=(joint_q_sum[i] / count[i]).cpu().tolist(),
                joint_target_mean_rad=(joint_target_sum[i] / count[i]).cpu().tolist(),
                joint_soft_limit_fraction=(joint_soft_hits[i] / count[i]).cpu().tolist(),
                joint_hard_margin_fraction=(joint_hard_hits[i] / count[i]).cpu().tolist()))
        result = dict(checkpoint=str(args.checkpoint.resolve()), push=args.push, seed=args.seed,
            onnx_audit=getattr(policy, "audit", None),
            trials_per_condition=args.trials, horizon_s=args.seconds, commands=command_table,
            suite=args.suite, joint_names=joint_names,
            joint_limits_rad=robot.data.joint_pos_limits[0, servo_ids].cpu().tolist(),
            head_joints=names, head_target="zero delta from HOME", metrics_window="after 1s, before first terminal; survivors only at each time",
            domain_randomization="frozen final-stage CoM +/-15mm, head CoM +/-10mm; remaining recipe defaults and observation noise retained",
            initial_joint_positions=initial_qpos, trials=rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
        print(f"RESULT {args.output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
