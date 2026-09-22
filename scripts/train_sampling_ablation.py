"""Bounded same-checkpoint control/specialist training, with isolated output."""
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ["MICRODUCK_WARM_START"] = "0"
os.environ["WANDB_MODE"] = "offline"

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["control", "specialist"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=250)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    task = "Mjlab-Velocity-Flat-MicroDuck" if args.arm == "control" else "Mjlab-Velocity-Specialist-Flat-MicroDuck"
    cfg = load_env_cfg(task)
    agent = load_rl_cfg(task)
    cfg.seed = agent.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    agent.logger = "tensorboard"
    agent.upload_model = False
    agent.save_interval = 50
    agent.max_iterations = args.iterations
    agent.run_name = f"sampling_ab_{args.arm}"
    configure_torch_backends()
    dump_yaml(args.output / "params/env.yaml", asdict(cfg))
    dump_yaml(args.output / "params/agent.yaml", asdict(agent))
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        wrapper = RslRlVecEnvWrapper(env)
        runner = load_runner_cls(task)(wrapper, asdict(agent), str(args.output), device="cuda:0")
        runner.load(str(args.checkpoint.resolve()), map_location="cuda:0")
        restored_step = env.common_step_counter
        restored_iter = runner.current_learning_iteration
        # The library restores optimizer lr but not PPO's adaptive-lr scalar.
        # Restore that scalar as well IN BOTH ARMS to avoid a startup lr jump.
        optimizer_lr = runner.alg.optimizer.param_groups[0]["lr"]
        runner.alg.learning_rate = optimizer_lr
        runner.add_git_repo_to_log(__file__)
        runner.learn(args.iterations, init_at_random_ep_len=True)
        command = env.command_manager.get_term("twist")
        result = dict(arm=args.arm, source=str(args.checkpoint.resolve()), seed=args.seed,
            num_envs=args.num_envs, iterations=args.iterations,
            restored_iteration=restored_iter, restored_common_step_counter=restored_step,
            restored_learning_rate=optimizer_lr, final_iteration=runner.current_learning_iteration,
            final_common_step_counter=env.common_step_counter,
            bucket_counts=command.sampling_bucket_counts.cpu().tolist() if hasattr(command, "sampling_bucket_counts") else None)
        (args.output / "experiment.json").write_text(json.dumps(result, indent=2))
        print("EXPERIMENT_RESULT", json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
