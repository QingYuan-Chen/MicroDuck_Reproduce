"""Aggregate direct evaluation metrics; equal weight to each fixed command."""
import argparse
import json
from pathlib import Path
from statistics import mean


def summarize(path):
    data = json.loads(path.read_text())
    result = {}
    for condition, command in data["commands"].items():
        rows = [r for r in data["trials"] if r["condition"] == condition]
        axis = "actual_yaw" if "turn" in condition else ("actual_vy" if "lateral" in condition else "actual_vx")
        target = command[2] if "turn" in condition else (command[1] if "lateral" in condition else command[0])
        result[condition] = dict(target=target, actual=mean(r[axis] for r in rows),
            xy_mae=mean(r["xy_error_m_s"] for r in rows), yaw_mae=mean(r["yaw_error_rad_s"] for r in rows),
            survival=sum(r["outcome"] == "horizon" for r in rows) / len(rows),
            duration=mean(r["duration_s"] for r in rows),
            action_second_diff=mean(r["action_second_diff"] for r in rows),
            head_yaw_std_deg=mean(r["head_temporal_std_deg"][2] for r in rows))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    results = {p.stem: summarize(p) for p in args.paths}
    print("| Command | " + " | ".join(results) + " |")
    print("|---|" + "---:|" * len(results))
    for condition in next(iter(results.values())):
        print("| " + condition + " | " + " | ".join(f"{r[condition]['actual']:.4f} ({100*r[condition]['survival']:.0f}%)" for r in results.values()) + " |")
    print("\nActual signed velocity (m/s or rad/s), followed by full-horizon survival.")
    for name, result in results.items():
        print(name)
        for prefix, metric in (("backward", "xy_mae"), ("turn", "yaw_mae"), ("forward", "xy_mae")):
            rows = [r for c, r in result.items() if c.startswith(prefix)]
            print(prefix, metric, mean(r[metric] for r in rows))
        print("survival", mean(r["survival"] for r in result.values()), "action_second_diff", mean(r["action_second_diff"] for r in result.values()))


if __name__ == "__main__":
    main()
