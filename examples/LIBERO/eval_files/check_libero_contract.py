"""Probe the installed LIBERO/Panda gripper using measured finger positions.

Run in the LIBERO environment, independently of a policy or StarVLA mapping:
    python -m examples.LIBERO.eval_files.check_libero_contract
"""

import json
import pathlib
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
import tyro


@dataclass
class Args:
    task_suite_name: str = "libero_object"
    task_id: int = 0
    seed: int = 7
    num_steps: int = 60


def main(args: Args) -> None:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(args.seed)
    init_state = suite.get_task_init_states(args.task_id)[0]
    result = {"robosuite_version": version("robosuite"), "commands": {}}
    try:
        robot = env.env.robots[0]
        result["gripper_class"] = type(robot.gripper).__name__
        for command in (-1.0, 1.0):
            env.reset()
            obs = env.set_init_state(init_state)
            for _ in range(10):
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
            before = np.asarray(obs["robot0_gripper_qpos"]).copy()
            for _ in range(args.num_steps):
                obs, _, _, _ = env.step([0.0] * 6 + [command])
            after = np.asarray(obs["robot0_gripper_qpos"])
            result["commands"][str(command)] = {
                "qpos_before": before.tolist(), "qpos_after": after.tolist(),
                "finger_separation": float(abs(after[0] - after[1])),
            }
        print(json.dumps(result, indent=2))
        opened = result["commands"]["-1.0"]["finger_separation"]
        closed = result["commands"]["1.0"]["finger_separation"]
        if opened <= closed + 0.001:
            raise RuntimeError("Measured gripper does not confirm -1=open/+1=close; inspect the installed environment")
        print("Measured LIBERO contract: -1 opens, +1 closes.")
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
