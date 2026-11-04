"""Offline, read-only semantic annotation of LeRobot v2 episodes.

Run ``python -m data_process.semantic_segmentation --help``. Model dependencies
are imported only when inference is requested; an injected backend can implement
``generate(messages) -> str`` for evaluation or alternative model servers.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import tempfile
import uuid

import numpy as np
from PIL import Image

from .config import resolve_camera_keys
from .dataset import LeRobotReader


PROMPT_VERSION = "semantic-segmentation-v1"
SPECIAL_LABELS = ("等待", "重试／失败恢复", "无法判断")
RULES = """你正在标注机器人长程任务。输入是按时间排列的采样画面，同一 sample_id 的多路相机属于同一时刻。
根据实际可见行为判断阶段，使用“动作＋对象＋目标”的子任务名称。任务描述不是已经完成的证据。
允许等待、重试／失败恢复、无法判断，不得编造未观察到的动作或成功结果。
sample_id 是本次输入的局部编号；row_index 是 Episode 内行索引，frame_index 是原始帧标识。
只输出要求的 JSON 对象，不要 Markdown 或解释。边界表示首次进入后一个阶段的位置。
"""


@dataclass(frozen=True)
class SegmentationConfig:
    sample_fps: float = 2.0
    window_seconds: float = 20.0
    overlap_seconds: float = 4.0
    max_frames: int = 40
    overview_frames: int = 32
    image_max_side: int = 448
    refine_seconds: float = 1.5
    refine_frames: int = 24
    retries: int = 1

    def __post_init__(self):
        for name in ("sample_fps", "window_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为正的有限数")
        if not math.isfinite(self.overlap_seconds) or not 0 <= self.overlap_seconds < self.window_seconds:
            raise ValueError("overlap_seconds 必须在 [0, window_seconds) 内")
        if not math.isfinite(self.refine_seconds) or self.refine_seconds < 0:
            raise ValueError("refine_seconds 必须为非负有限数；0 表示不细化")
        for name in ("max_frames", "overview_frames", "refine_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 2:
                raise ValueError(f"{name} 必须为至少 2 的整数")
        if type(self.image_max_side) is not int or self.image_max_side < 32:
            raise ValueError("image_max_side 必须为至少 32 的整数")
        if type(self.retries) is not int or self.retries < 0:
            raise ValueError("retries 必须为非负整数")


class EpisodeSource:
    """Keep original identifiers and clocks; decode only sampled images."""

    def __init__(self, reader, episode, cameras=None, task=None):
        self.reader, self.episode = reader, episode
        self.length = reader.episodes[episode]["length"]
        if self.length <= 0:
            raise ValueError(f"Episode {episode} 没有帧")
        self.cameras = resolve_camera_keys(reader.camera_keys, cameras)
        columns = [key for key in ("timestamp", "frame_index", "task_index") if key in reader.features]
        values = reader.read_episode(episode, columns=columns) if columns else {}
        if "timestamp" in values:
            self.times = np.asarray(values["timestamp"], dtype=float).reshape(-1)
            self.timestamp_source = "timestamp"
        else:
            fps = float(reader.fps)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("缺少 timestamp 时需要有效的 FPS")
            self.times = np.arange(self.length, dtype=float) / fps
            self.timestamp_source = "estimated_from_fps"
        if (len(self.times) != self.length or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times) <= 0)):
            raise ValueError("timestamp 必须与帧数一致、有限且严格递增")
        identifiers = np.asarray(values.get("frame_index", np.arange(self.length))).reshape(-1)
        if len(identifiers) != self.length or identifiers.dtype.kind not in "iu":
            raise ValueError("frame_index 必须是与帧数一致的整数标识")
        self.frame_indices = identifiers
        if task is not None and not task.strip():
            raise ValueError("任务描述不能为空")
        tasks = reader.episodes[episode].get("tasks", reader.episodes[episode].get("task", [])) or []
        if isinstance(tasks, str):
            tasks = [tasks]
        if not tasks and "task_index" in values:
            tasks = [reader.tasks[int(index)] for index in np.unique(values["task_index"])]
        if task is None and len(tasks) != 1:
            raise ValueError("无法确定唯一完整任务，请通过 --task 指定")
        self.task = task.strip() if task else tasks[0]

    def sample(self, start, end, limit, fps=None):
        """Sample in time, include both ends, and never exceed the image budget."""
        count = limit if fps is None else min(limit, max(2, math.ceil(
            (self.times[end - 1] - self.times[start]) * fps) + 1))
        count = min(count, end - start)
        targets = np.linspace(self.times[start], self.times[end - 1], count)
        return np.unique(np.clip(np.searchsorted(self.times, targets), start, end - 1)).tolist()

    def windows(self, config):
        windows, start = [], 0
        while start < self.length:
            end = max(start + 1, int(np.searchsorted(
                self.times, self.times[start] + config.window_seconds)))
            end = min(end, self.length)
            windows.append((start, end))
            if end == self.length:
                break
            start = max(start + 1, int(np.searchsorted(
                self.times, self.times[start] + config.window_seconds - config.overlap_seconds)))
        return windows

    def images(self, rows, max_side):
        result = {row: {} for row in rows}
        embedded = [key for key in self.cameras if self.reader.features[key]["dtype"] == "image"]
        for row in rows:
            if embedded:
                images, errors = self.reader.read_image_frame(self.episode, row, embedded)
                if errors:
                    raise ValueError(f"第 {row} 帧图像无法解码: {errors}")
                for key, image in images.items():
                    result[row][key] = Image.fromarray(image.transpose(1, 2, 0))
        for key in self.cameras:
            if key not in embedded:
                self._video_images(key, rows, result)
        for images in result.values():
            for image in images.values():
                image.thumbnail((max_side, max_side))
        return result

    def _video_images(self, key, rows, result):
        import av

        pattern = self.reader.info.get("video_path")
        if not pattern:
            raise ValueError("外部视频缺少 meta/info.json 中的 video_path")
        path = self.reader.path / pattern.format(
            episode_chunk=self.episode // self.reader.info["chunks_size"],
            episode_index=self.episode, video_key=key,
        )
        # v2 videos are per episode. Row timestamps are episode-relative seconds.
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            origin = float((stream.start_time or 0) * stream.time_base)
            fps = float(stream.average_rate or self.reader.fps)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"视频 {path} 缺少有效帧率")
            tolerance = max(0.001, 1.5 / fps)
            for row in rows:
                target = float(self.times[row]) + origin
                container.seek(int(target / stream.time_base), stream=stream, backward=True)
                best, distance = None, float("inf")
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    timestamp = float(frame.pts * stream.time_base)
                    if abs(timestamp - target) < distance:
                        best, distance = frame, abs(timestamp - target)
                    if timestamp >= target:
                        break
                if best is None or distance > tolerance:
                    raise ValueError(f"视频 {path} 在时间 {self.times[row]}s 没有匹配帧")
                result[row][key] = best.to_image().convert("RGB")


class TransformersVisionModel:
    """Load the architecture declared by a local model, without remote code."""

    def __init__(self, model_path, device="cuda:0", dtype="auto", max_new_tokens=2048):
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"本地模型目录不存在: {path}")
        if device != "cpu" and not (device.startswith("cuda") and torch.cuda.is_available()):
            raise ValueError("CUDA 不可用，请检查驱动或明确使用 --device cpu（推理较慢）")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens 必须为正整数")
        if dtype == "auto":
            model_dtype = (torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported()
                           else torch.float16 if device.startswith("cuda") else torch.float32)
        else:
            model_dtype = getattr(torch, dtype)
        config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.template_kwargs = ({"enable_thinking": False}
                                if "enable_thinking" in str(self.processor.chat_template) else {})
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, config=config, local_files_only=True, trust_remote_code=False,
            dtype=model_dtype, attn_implementation="sdpa",
        ).to(device).eval()
        self.max_new_tokens = max_new_tokens
        self.metadata = dict(path=str(path), model_type=config.model_type, device=device,
                             dtype=str(model_dtype), transformers_version=transformers.__version__,
                             torch_version=torch.__version__, max_new_tokens=max_new_tokens,
                             do_sample=False, chat_template_kwargs=self.template_kwargs)

    def generate(self, messages):
        import torch

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", **self.template_kwargs,
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        if new_tokens.shape[1] >= self.max_new_tokens:
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            if new_tokens[0, -1].item() not in eos:
                raise ValueError("模型输出达到 max_new_tokens，可能被截断；请增加 --max-new-tokens")
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


def _json_object(text):
    try:
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        value = json.loads(stripped)
    except (ValueError, IndexError) as exc:
        raise ValueError("模型没有返回有效 JSON 对象") from exc
    if not isinstance(value, dict):
        raise ValueError("模型必须返回 JSON 对象")
    return value


def _labels(value):
    labels = value.get("subtasks")
    if (not isinstance(labels, list) or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)):
        raise ValueError("subtasks 必须为非空字符串列表")
    return list(dict.fromkeys([label.strip() for label in labels] + list(SPECIAL_LABELS)))


class SemanticSegmenter:
    def __init__(self, backend, config=None):
        self.backend = backend
        self.config = config or SegmentationConfig()

    def _request(self, source, rows, prompt, validate, trace, purpose):
        images = source.images(rows, self.config.image_max_side)
        content = [{"type": "text", "text": RULES + "\n完整任务：" + source.task + "\n" + prompt}]
        for sample, row in enumerate(rows):
            content.append({"type": "text", "text": (
                f"sample_id={sample}, row_index={row}, frame_index={int(source.frame_indices[row])}, "
                f"timestamp={float(source.times[row]):.9f}s")})
            for camera in source.cameras:
                content.extend([{"type": "text", "text": f"camera={camera}"},
                                {"type": "image", "image": images[row][camera]}])
        entry = dict(purpose=purpose, rows=rows, timestamps=source.times[rows].tolist(),
                     prompt=prompt, attempts=[])
        trace.append(entry)
        correction = ""
        for attempt in range(self.config.retries + 1):
            messages = [{"role": "user", "content": content + ([{"type": "text", "text": correction}]
                                                                  if correction else [])}]
            raw = self.backend.generate(messages)
            record = {"output": raw}
            entry["attempts"].append(record)
            try:
                return validate(_json_object(raw))
            except ValueError as exc:
                record["validation_error"] = str(exc)
                if attempt == self.config.retries:
                    raise ValueError(f"{purpose} 的模型输出校验失败: {exc}") from exc
                correction = f"上次输出不合法：{exc}。请重新判断，仅返回满足上述结构的 JSON。"

    def segment(self, source, labels=None):
        trace, config = [], self.config
        discover_labels = labels is None
        if labels is None:
            rows = source.sample(0, source.length, config.overview_frames)
            labels = self._request(source, rows,
                '列出画面中实际出现的子任务，按首次出现排序。返回 {"subtasks":["子任务名称",...]}。',
                _labels, trace, "overview")
        else:
            labels = _labels({"subtasks": labels})
        windows = source.windows(config)
        segments = []
        for index, (start, end) in enumerate(windows):
            rows = source.sample(start, end, config.max_frames, config.sample_fps)

            def validate(value):
                items = value.get("segments")
                if not isinstance(items, list) or not items:
                    raise ValueError("segments 必须为非空列表")
                previous = -1
                for item in items:
                    if not isinstance(item, dict):
                        raise ValueError("segment 必须为对象")
                    sample = item.get("start_sample")
                    if type(sample) is not int or not previous < sample < len(rows):
                        raise ValueError("start_sample 必须为输入范围内严格递增的整数")
                    label = item.get("subtask")
                    if not isinstance(label, str) or not label.strip() or not isinstance(item.get("evidence"), str):
                        raise ValueError("subtask 和 evidence 必须是字符串，subtask 不能为空")
                    if not discover_labels and label not in labels:
                        raise ValueError("subtask 必须来自指定词表")
                    previous = sample
                if items[0]["start_sample"] != 0:
                    raise ValueError("第一个 start_sample 必须为 0，覆盖整个窗口")
                return items

            prompt = ('词表：' + json.dumps(labels, ensure_ascii=False) + '\n'
                      + ('优先复用词表；发现词表未覆盖的实际子任务时可以使用新的明确名称。'
                         if discover_labels else '必须使用词表中的名称。') +
                      '将本窗口划分为连续阶段。相邻阶段不要使用相同名称。'
                      'start_sample 是阶段开始的 sample_id；阶段延续到下一阶段开始或窗口结束。'
                      '第一个 start_sample 必须是 0。返回 '
                      '{"segments":[{"start_sample":0,"subtask":"词表中的名称","evidence":"可见证据"},...]}。')
            items = self._request(source, rows, prompt, validate, trace, f"window_{index}")
            for item in items:
                if item["subtask"] not in labels:
                    labels.append(item["subtask"])
            # Give each overlapping pair one deterministic ownership seam.
            core_start = 0 if index == 0 else (start + windows[index - 1][1]) // 2
            core_end = source.length if index == len(windows) - 1 else (end + windows[index + 1][0]) // 2
            for number, item in enumerate(items):
                sample = item["start_sample"]
                left = max(core_start, rows[sample])
                right = min(core_end, rows[items[number + 1]["start_sample"]]
                            if number + 1 < len(items) else end)
                if left >= right:
                    continue
                boundary = dict(lower_row=rows[sample - 1] + 1 if sample else start,
                                upper_row=rows[sample], method="coarse_samples")
                if left != rows[sample] or (sample == 0 and index > 0):
                    boundary = dict(lower_row=start, upper_row=max(start, windows[index - 1][1] - 1)
                                    if index else start, method="window_seam")
                if segments and segments[-1]["subtask"] == item["subtask"]:
                    segments[-1]["end_frame_exclusive"] = right
                else:
                    segments.append(dict(start_frame=left, end_frame_exclusive=right,
                                         subtask=item["subtask"], evidence=item["evidence"],
                                         boundary_before=boundary if left else None))
        if config.refine_seconds:
            for index in range(1, len(segments)):
                left, right = segments[index - 1], segments[index]
                boundary = right["start_frame"]
                uncertainty = right["boundary_before"]
                lower = max(left["start_frame"] + 1, min(uncertainty["lower_row"], int(np.searchsorted(
                    source.times, source.times[boundary] - config.refine_seconds))))
                upper = min(right["end_frame_exclusive"] - 1, max(uncertainty["upper_row"], int(np.searchsorted(
                    source.times, source.times[boundary] + config.refine_seconds, side="right")) - 1))
                if lower > upper:
                    continue
                rows = source.sample(lower, upper + 1, config.refine_frames)

                def validate_boundary(value):
                    sample = value.get("boundary_sample")
                    if "boundary_sample" not in value or (sample is not None and (
                            type(sample) is not int or not 0 <= sample < len(rows))):
                        raise ValueError("boundary_sample 必须是输入中的 sample_id")
                    if not isinstance(value.get("evidence"), str):
                        raise ValueError("evidence 必须是字符串")
                    return value

                prompt = (f'细化阶段“{left["subtask"]}”到“{right["subtask"]}”的切换位置。'
                          '选择最早属于后一个阶段的 sample_id；若窗口已经属于后一个阶段则选择 0。'
                          '若画面不支持这一阶段切换，boundary_sample 返回 null，并说明原因。'
                          '返回 {"boundary_sample":整数,"evidence":"可见证据"}。')
                value = self._request(source, rows, prompt, validate_boundary, trace, f"boundary_{index}")
                sample = value["boundary_sample"]
                if sample is None or sample == 0:
                    # Without an observed preceding-stage sample, do not claim
                    # that the first image is an exact transition.
                    right["boundary_before"].update(
                        refinement="inconclusive" if sample is None else "left_censored",
                        evidence=value["evidence"],
                    )
                    continue
                left["end_frame_exclusive"] = right["start_frame"] = rows[sample]
                right["boundary_before"] = dict(
                    lower_row=rows[sample - 1] + 1 if sample else lower,
                    upper_row=rows[sample], method="refined_samples", evidence=value["evidence"],
                )
        cursor = 0
        for index, segment in enumerate(segments):
            start, end = segment["start_frame"], segment["end_frame_exclusive"]
            if start != cursor or not start < end <= source.length:
                raise ValueError("内部区间校验失败：分段存在遗漏、重叠或越界")
            segment.update(start_frame_index=int(source.frame_indices[start]),
                           last_frame_index=int(source.frame_indices[end - 1]),
                           start_time=float(source.times[start]), last_time=float(source.times[end - 1]))
            if index:
                boundary = segment["boundary_before"]
                boundary["lower_row"] = max(segments[index - 1]["start_frame"] + 1,
                                            min(start, boundary["lower_row"]))
                boundary["upper_row"] = min(end - 1, max(start, boundary["upper_row"]))
            cursor = end
        if cursor != source.length:
            raise ValueError("内部区间校验失败：未覆盖整个 Episode")
        return dict(version=1, prompt_version=PROMPT_VERSION, dataset_path=str(source.reader.path),
                    episode_index=source.episode, task=source.task, cameras=source.cameras,
                    length=source.length, timestamp_source=source.timestamp_source,
                    frame_interval="episode row indices [start_frame, end_frame_exclusive)",
                    time_interval="[start_time, last_time], timestamps of first and last included rows",
                    model=getattr(self.backend, "metadata", {"backend": type(self.backend).__name__}),
                    config=asdict(config), labels=labels, segments=segments, inference_calls=trace,
                    rules=RULES)


def _save_json(path, value):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def build_parser():
    parser = argparse.ArgumentParser(description="使用本地视觉语言模型对 LeRobot v2 长程任务做语义分段")
    parser.add_argument("--path", type=Path, required=True, help="LeRobot v2 数据集根目录")
    parser.add_argument("--model-path", type=Path, default=Path("playground/Pretrained_models/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--episodes", type=int, nargs="+", help="Episode 编号；省略则处理全集")
    parser.add_argument("--cameras", nargs="+", help="相机字段、短名称、别名或通配符；默认使用第一路")
    parser.add_argument("--task", help="覆盖完整任务描述；多任务 Episode 必须明确指定")
    parser.add_argument("--labels", nargs="+", help="可选子任务词表；省略时逐 Episode 生成")
    parser.add_argument("--output", type=Path, default=Path("data_process/output/semantic_segmentation"))
    parser.add_argument("--device", default="cuda:0", help="例如 cuda:0、cuda:1 或 cpu")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    for name, kind in (("sample_fps", float), ("window_seconds", float), ("overlap_seconds", float),
                       ("max_frames", int), ("overview_frames", int), ("image_max_side", int),
                       ("refine_seconds", float), ("refine_frames", int), ("retries", int)):
        parser.add_argument("--" + name.replace("_", "-"), type=kind,
                            default=getattr(SegmentationConfig(), name))
    parser.add_argument("--dry-run", action="store_true", help="校验数据、解码采样图并显示计划，不加载模型或保存报告")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = SegmentationConfig(**{key: getattr(args, key) for key in asdict(SegmentationConfig())})
        if args.max_new_tokens < 1:
            raise ValueError("max_new_tokens 必须大于 0")
        reader = LeRobotReader(args.path)
        episodes = list(dict.fromkeys(args.episodes if args.episodes is not None else reader.episode_indices))
        if not episodes or any(episode not in reader.episodes for episode in episodes):
            raise ValueError("Episode 列表为空或包含不存在的编号")
        output = args.output.expanduser().resolve()
        if output == reader.path or reader.path in output.parents:
            raise ValueError("输出目录必须位于源数据集之外")
        if args.dry_run:
            for episode in episodes:
                source = EpisodeSource(reader, episode, args.cameras, args.task)
                rows = source.sample(0, source.length, config.overview_frames)
                source.images(rows, config.image_max_side)
                print(json.dumps(dict(episode_index=episode, task=source.task, length=source.length,
                                      cameras=source.cameras, timestamp_source=source.timestamp_source,
                                      overview_rows=rows, windows=source.windows(config)), ensure_ascii=False))
            return 0
        # Validate the first episode before spending time loading model weights.
        source = EpisodeSource(reader, episodes[0], args.cameras, args.task)
        print(f"相机字段: {', '.join(source.cameras)}", flush=True)
        backend = TransformersVisionModel(args.model_path, args.device, args.dtype, args.max_new_tokens)
        segmenter = SemanticSegmenter(backend, config)
        run = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
        run.mkdir(parents=True, exist_ok=False)
        manifest = dict(version=1, dataset_path=str(reader.path), episodes=episodes, completed=[], errors=[],
                        status="running", model=backend.metadata, config=asdict(config))
        _save_json(run / "manifest.json", manifest)
        print(f"结果目录: {run}", flush=True)
        try:
            for episode in episodes:
                try:
                    source = EpisodeSource(reader, episode, args.cameras, args.task)
                    report = segmenter.segment(source, labels=args.labels)
                    _save_json(run / f"episode_{episode:06d}.json", report)
                    manifest["completed"].append(episode)
                    print(f"Episode {episode}: {len(report['segments'])} 个阶段", flush=True)
                except (ValueError, OSError, RuntimeError, KeyError, ImportError, NotImplementedError) as exc:
                    manifest["errors"].append(dict(episode_index=episode, error=str(exc)))
                    print(f"Episode {episode} 失败: {exc}", file=sys.stderr, flush=True)
                _save_json(run / "manifest.json", manifest)
        except KeyboardInterrupt:
            manifest["status"] = "interrupted"
            _save_json(run / "manifest.json", manifest)
            print("已停止，已完成 Episode 的报告已保留。", file=sys.stderr)
            return 130
        manifest["status"] = "partial" if manifest["errors"] else "complete"
        _save_json(run / "manifest.json", manifest)
        return 1 if manifest["errors"] else 0
    except (ValueError, OSError, KeyError, RuntimeError, ImportError, NotImplementedError) as exc:
        parser.exit(1, f"语义分段失败: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
