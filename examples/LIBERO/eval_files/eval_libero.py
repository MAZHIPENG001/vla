"""LIBERO evaluation with the episode-local action queue used by openpi."""

from collections import deque
import dataclasses
import logging
import pathlib

import numpy as np
import tqdm

from examples.LIBERO.eval_files.libero_contract import (
    GripperEncoding,
    gripper_to_libero,
    libero_images,
    libero_state,
)
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    unnorm_key: str | None = None
    # None executes the full predicted chunk (8 for the released OFT model).
    replan_interval: int | None = None
    gripper_encoding: GripperEncoding = "binary_open"
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    max_tasks: int = -1  # Positive values limit the task count for smoke tests.
    video_out_path: str = "experiments/libero/logs"
    seed: int = 7


def _resolve_replan_interval(requested: int | None, horizon: int) -> int:
    interval = horizon if requested is None else requested
    if not 1 <= interval <= horizon:
        raise ValueError(f"replan_interval must be in [1, {horizon}], got {interval}")
    return interval


def _run_episode(env, client: ModelClient, init_state, task_description: str, args: Args, max_steps: int):
    """Infer when the queue is empty and execute its actions in prediction order."""
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    interval = _resolve_replan_interval(args.replan_interval, client.action_chunk_size)
    action_plan = deque()
    replay_images = []
    for _ in range(max_steps):
        images = libero_images(obs)
        replay_images.append(images[0])
        if not action_plan:
            example = {"image": images, "lang": task_description}
            if client.include_state:
                example["state"] = libero_state(obs)[None]
            actions = client.infer(example)
            # Shorter valid predictions are exhausted before the next inference.
            action_plan.extend(actions[:interval])

        action = action_plan.popleft().copy()
        action[6] = gripper_to_libero(action[6], args.gripper_encoding)[0]
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True, replay_images
    return False, replay_images


def eval_libero(args: Args) -> None:
    from libero.libero import benchmark
    import imageio

    if args.task_suite_name not in MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    if args.num_trials_per_task <= 0 or args.num_steps_wait < 0:
        raise ValueError("num_trials_per_task must be positive and num_steps_wait must be nonnegative")
    logging.info("Arguments: %s", dataclasses.asdict(args))
    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = task_suite.n_tasks if args.max_tasks <= 0 else min(args.max_tasks, task_suite.n_tasks)
    video_dir = pathlib.Path(args.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)

    client = ModelClient(args.host, args.port, args.unnorm_key)
    try:
        interval = _resolve_replan_interval(args.replan_interval, client.action_chunk_size)
        logging.info(
            "Checkpoint=%s; replan_interval=%d; include_state=%s; gripper_encoding=%s; "
            "cameras=primary,wrist; rotation=180; resize=224 bicubic",
            client.metadata.get("ckpt_path"), interval, client.include_state, args.gripper_encoding,
        )
        total_episodes, total_successes = 0, 0
        for task_id in tqdm.tqdm(range(n_tasks), desc="Tasks"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            if args.num_trials_per_task > len(initial_states):
                raise ValueError(f"Requested {args.num_trials_per_task} trials, but task {task_id} has {len(initial_states)} initial states")
            env, description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            task_successes = 0
            try:
                for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=description):
                    success, replay = _run_episode(
                        env, client, initial_states[episode_idx], str(description), args,
                        MAX_STEPS[args.task_suite_name],
                    )
                    task_successes += int(success)
                    total_successes += int(success)
                    total_episodes += 1
                    suffix = "success" if success else "failure"
                    segment = str(description).replace(" ", "_")
                    imageio.mimwrite(
                        video_dir / f"rollout_{segment}_episode{episode_idx}_{suffix}.mp4", replay, fps=10,
                    )
                    logging.info(
                        "Success: %s; episodes=%d; successes=%d (%.1f%%)",
                        success, total_episodes, total_successes, 100 * total_successes / total_episodes,
                    )
                logging.info("Current task success rate: %s", task_successes / args.num_trials_per_task)
                logging.info("Current total success rate: %s", total_successes / total_episodes)
            finally:
                env.close()
        logging.info("Total success rate: %s", total_successes / total_episodes)
        logging.info("Total episodes: %d", total_episodes)
    finally:
        client.close()


def _get_libero_env(task, resolution, seed):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env, task.language


if __name__ == "__main__":
    import tyro

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-8s | %(message)s", datefmt="%m/%d [%H:%M:%S]",
    )
    tyro.cli(eval_libero)
