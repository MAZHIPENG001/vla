import dataclasses
import json
import logging
import math
import os
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Literal

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim.
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = -1  # If > 0, limit the number of tasks evaluated (smoke / quick check). -1 = run all.
    render_gpu_device_id: int = -1  # Original EGL GPU index, before CUDA_VISIBLE_DEVICES renumbering; -1 = auto.
    realtime: bool = True  # Advance simulation while inference runs in a worker thread.
    control_freq: int = 20  # Simulation control frequency and replay FPS, in Hz.
    rtc_enabled: bool = False  # Preserve the existing rollout when disabled.
    rtc_stride: int | Literal["auto"] = "auto"  # Paper s_min; auto selects 1, actual s follows Algorithm 1.
    rtc_delay: int | Literal["auto"] = "auto"  # Measure d from control steps; int seeds initial estimate.
    rtc_beta: float = 5.0  # Paper Eq. 2 guidance clipping beta (Table 4).
    inference_timeout: float = 10.0  # Deployment transport bound, in wall seconds.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    post_process_action: bool = True

    job_name: str = "test"


def _make_model_example(obs, task_description):
    # Copy camera buffers before sending them to the inference thread. LIBERO's
    # renderer and all env.step calls stay on the main thread (OpenGL context).
    return {
        "image": [
            np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        ],
        "lang": str(task_description),
    }


def _decode_action(response):
    raw_action = response["raw_action"]
    world_vector = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
    rotation = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
    gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
    if (world_vector.size, rotation.size, gripper.size) != (3, 3, 1):
        raise ValueError(
            f"Invalid action sizes: world_vector={world_vector.shape}, "
            f"rotation_delta={rotation.shape}, gripper={gripper.shape}"
        )
    action = np.concatenate([world_vector, rotation, _binarize_gripper_open(gripper)])
    if not np.isfinite(action).all() or not np.isfinite(gripper).all():
        raise ValueError("Model returned a non-finite action")
    return action.tolist()


@lru_cache(maxsize=1)
def _video_status_badges():
    """Render English labels once with Pillow's bundled font."""
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Older Pillow versions have a fixed-size bundled font.
        font = ImageFont.load_default()

    badges = {}
    for waiting, text, color in (
        (True, "Waiting for inference", (255, 0, 0, 255)),
        (False, "Executing", (0, 255, 0, 255)),
    ):
        left, top, right, bottom = font.getbbox(text)
        padding = 5
        badge = Image.new("RGBA", (right - left + 2 * padding, bottom - top + 2 * padding), (0, 0, 0, 180))
        ImageDraw.Draw(badge).text((padding - left, padding - top), text, font=font, fill=color)
        badges[waiting] = badge
    return badges


def _annotate_video_frame(frame, waiting):
    """Overlay status on a copy; model observations retain their original pixels."""
    image = Image.fromarray(frame).copy()
    badge = _video_status_badges()[waiting]
    image.paste(badge, (8, 8), badge)
    return np.asarray(image)


def _run_episode(env, client_model, obs, task_description, args, max_steps):
    """Run one rollout, counting inference waits against the simulation budget.

    Policy indices advance only when a predicted action is consumed. Waiting
    does not skip cached actions. Only one request may be in flight, and it is
    drained before returning so it cannot mutate the next episode's cache.
    """
    if args.rtc_enabled:
        return _run_rtc_episode(env, client_model, obs, task_description, args, max_steps)
    _video_status_badges()  # Validate and load the font before starting the control clock.
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    replay_images = []
    replay_waiting = []
    policy_step = wait_steps = 0
    done = False
    pending = None
    hold_action = LIBERO_DUMMY_ACTION.copy()
    period = 1.0 / args.control_freq
    started = time.monotonic()
    next_tick = started

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="libero-inference") as executor:
        try:
            for _ in range(max_steps):
                if args.realtime:
                    delay = next_tick - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)

                action = None
                if pending is None:
                    example = _make_model_example(obs, task_description)
                    if args.realtime and client_model.needs_inference(policy_step):
                        pending = executor.submit(client_model.step, example=example, step=policy_step)
                    else:
                        # Cache hits must not incur an extra control tick.
                        action = _decode_action(client_model.step(example=example, step=policy_step))

                if pending is not None and pending.done():
                    response = pending.result()  # Propagate inference failures.
                    pending = None
                    action = _decode_action(response)

                waiting_for_inference = action is None
                if waiting_for_inference:
                    # OSC_POSE commands are deltas: zero arm deltas hold the
                    # current pose; retain only the last gripper command.
                    action = hold_action.copy()
                    wait_steps += 1
                else:
                    hold_action[-1] = action[-1]
                    policy_step += 1

                obs, _, done, _ = env.step(action)
                # Include every physics/control step, including terminal and
                # inference-wait steps, in the video.
                replay_images.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                replay_waiting.append(waiting_for_inference)
                next_tick += period
                if done:
                    break

            if args.realtime:
                # The final env.step advances one full control period too.
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            elapsed = time.monotonic() - started
        finally:
            if pending is not None:
                pending.result()

    sim_seconds = len(replay_images) * period
    logging.info(
        "Rollout timing: sim=%.3fs, wall=%.3fs, policy_actions=%d, inference_wait_steps=%d",
        sim_seconds, elapsed, policy_step, wait_steps,
    )
    if args.realtime and elapsed > sim_seconds * 1.1:
        logging.warning(
            "Simulation did not keep up with real time (%.2fx wall/sim); "
            "inference latency in simulated seconds may be underestimated.",
            elapsed / sim_seconds,
        )
    # Draw after the rollout so text rendering cannot affect control timing or
    # the camera images used for the next model request.
    replay_images = [
        _annotate_video_frame(frame, waiting)
        for frame, waiting in zip(replay_images, replay_waiting)
    ]
    return replay_images, done


def _run_rtc_episode(env, client_model, obs, task_description, args, max_steps):
    """Execute RTC at the simulation control clock; all env access stays here."""
    from examples.LIBERO.eval_files.rtc_controller import RTCConfig, RTCController

    if not args.realtime:
        raise ValueError("RTC requires realtime=True so inference delay is measured in control cycles")
    config = RTCConfig(
        stride=args.rtc_stride, delay=args.rtc_delay, beta=args.rtc_beta,
        timeout=args.inference_timeout,
    )
    config.validate(client_model.action_chunk_size)
    client_model.set_inference_timeout(config.timeout)
    controller = RTCController(client_model, config)
    replay_images, replay_waiting = [], []
    done = False
    hold_action = LIBERO_DUMMY_ACTION.copy()
    period = 1.0 / args.control_freq
    _video_status_badges()
    try:
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        started = next_tick = time.monotonic()
        for control_step in range(max_steps):
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            raw = controller.next_action(_make_model_example(obs, task_description), control_step)
            waiting = raw is None
            if waiting:
                action = hold_action.copy()
            else:
                action = _decode_action({"raw_action": {
                    "world_vector": raw[:3], "rotation_delta": raw[3:6], "open_gripper": raw[6:7],
                }})
                hold_action[-1] = action[-1]
            obs, _, done, _ = env.step(action)
            replay_images.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            replay_waiting.append(waiting)
            next_tick += period
            if done:
                break
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        elapsed = time.monotonic() - started
        sim_seconds = len(replay_images) * period
        logging.info("RTC rollout timing: sim=%.3fs wall=%.3fs", sim_seconds, elapsed)
        if sim_seconds and elapsed > sim_seconds * 1.1:
            logging.warning("RTC simulation slower than real time: %.2fx wall/sim", elapsed / sim_seconds)
    finally:
        try:
            controller.close()  # Bounded drain, including early success and exceptions.
        finally:
            logging.info("RTC episode statistics: %s", json.dumps(dataclasses.asdict(controller.stats)))
    return [
        _annotate_video_frame(frame, waiting)
        for frame, waiting in zip(replay_images, replay_waiting)
    ], done


def eval_libero(args: Args) -> None:
    from libero.libero import benchmark
    from examples.LIBERO.eval_files.model2libero_interface import ModelClient

    if args.control_freq <= 0:
        raise ValueError("control_freq must be positive")
    if args.num_steps_wait < 0:
        raise ValueError("num_steps_wait must be non-negative")
    if args.rtc_enabled and not args.realtime:
        raise ValueError("RTC requires realtime=True")
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":#220
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":#280
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":#300
        max_steps = 600  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":#520
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":#400
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
    )

    # Optional smoke-test cap (still useful for quick verification with -1 = full run).
    n_eval_tasks = num_tasks_in_suite if args.max_tasks <= 0 else min(args.max_tasks, num_tasks_in_suite)
    logging.info(f"Evaluating {n_eval_tasks} of {num_tasks_in_suite} tasks (max_tasks={args.max_tasks})")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    try:
        for task_id in tqdm.tqdm(range(n_eval_tasks)):
            # Get task
            task = task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = _get_libero_env(
                task, LIBERO_ENV_RESOLUTION, args.seed, args.control_freq, args.render_gpu_device_id
            )

            try:
                # Start episodes
                task_episodes, task_successes = 0, 0
                for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                    logging.info(f"\nTask: {task_description}")

                    # Reset environment
                    client_model.reset(task_description=task_description)  # Reset the client connection
                    env.reset()

                    # Set initial states
                    obs = env.set_init_state(initial_states[episode_idx])

                    logging.info(f"Starting episode {task_episodes + 1}...")
                    replay_images, done = _run_episode(
                        env, client_model, obs, task_description, args, max_steps
                    )
                    if done:
                        task_successes += 1
                        total_successes += 1

                    task_episodes += 1
                    total_episodes += 1

                    # Save a replay video of the episode
                    suffix = "success" if done else "failure"
                    task_segment = task_description.replace(" ", "_")
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=args.control_freq,
                    )

                    # Log current results
                    logging.info(f"Success: {done}")
                    logging.info(f"# episodes completed so far: {total_episodes}")
                    logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            finally:
                env.close()

            # Log final results
            logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
        logging.info(f"Total episodes: {total_episodes}")

    finally:
        client_model.close()


def _get_libero_env(task, resolution, seed, control_freq=20, render_gpu_device_id=-1):
    """Initializes and returns the LIBERO environment, along with the task description."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "control_freq": control_freq,
        "render_gpu_device_id": render_gpu_device_id,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    import tyro

    tyro.cli(eval_libero)
