"""Robotic Data 质量检查。只生成报告和保留候选，不修改信号或源文件。"""

from collections import Counter, OrderedDict
from concurrent.futures import CancelledError
from dataclasses import asdict
from hashlib import blake2b
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import QUALITY_POLICY, QualityPolicy
from .dataset import BrowserModel, json_value


def _threshold(value, width, dimensions):
    if value is None:
        return None
    try:
        return np.broadcast_to(np.asarray(value, dtype=float), (width,))[dimensions]
    except ValueError as exc:
        raise ValueError(f"逐维阈值需要 1 个或 {width} 个数值") from exc


def _derivatives(values, times, kind):
    """在真实时间间隔上差分；异常索引指向差分窗口的最后一帧。"""
    order = {"position": 0, "command": 0, "velocity": 1, "acceleration": 2}[kind]
    indices = np.arange(len(times))
    result = {}
    while order <= 3 and len(values):
        if order:
            result[order] = values, indices
        if len(values) < 2:
            break
        values = np.diff(values, axis=0) / np.diff(times)[:, None]
        times = (times[:-1] + times[1:]) / 2
        indices = indices[1:]
        order += 1
    return result


class QualityInspector:
    """与浏览器使用独立数据实例，在一个后台线程中执行；缓存小型图像指标。"""

    def __init__(self, path, *, state_key=None, action_key=None):
        self.model = BrowserModel(path, state_key=state_key, action_key=action_key)
        self._image_cache = OrderedDict()
        self._episode_tasks = {}
        for task in self.model.task_stats:
            for eid in task.episode_ids:
                self._episode_tasks.setdefault(eid, []).append(task.name)

    def validate_policy(self, policy: QualityPolicy) -> None:
        """检查策略与当前数据集的维度、相机及时间戳映射是否兼容。"""
        clocks = list(policy.camera_timestamp_keys.values())
        if policy.state_timestamp_key in clocks or len(clocks) != len(set(clocks)):
            raise ValueError("同步检查需要各传感器独立的采集时间戳字段，不能把公共 timestamp 或同一列重复映射为多个传感器")
        for role in ("state", "action"):
            field = getattr(self.model.schema, role)
            rule = getattr(policy, role)
            dimensions = list(rule.dimensions) if rule.dimensions is not None else list(range(field.width))
            if field.key and any(i >= field.width for i in dimensions):
                raise ValueError(f"{role} 仅有 {field.width} 维，维度选择越界")
            for value in (rule.static_epsilon, rule.jerk_limit):
                if field.key:
                    _threshold(value, field.width, dimensions)
        for camera, key in policy.camera_timestamp_keys.items():
            if camera not in self.model.loader.camera_keys:
                raise ValueError(f"未知相机: {camera}")
            feature = self.model.loader.features.get(key)
            if feature is None or feature["dtype"] in {"image", "video"}:
                raise ValueError(f"相机 {camera} 的时间戳字段 {key!r} 不存在或不是数值字段")

    @staticmethod
    def _cancel(cancel):
        if cancel is not None and cancel.is_set():
            raise CancelledError()

    def inspect(self, episode_ids, policy: QualityPolicy = QUALITY_POLICY, *, progress=None, cancel=None):
        self.validate_policy(policy)
        ids = list(dict.fromkeys(int(i) for i in episode_ids))
        if not ids:
            raise ValueError("检查范围没有 Episode")
        if any(i not in self.model.loader.episodes for i in ids):
            raise ValueError("检查范围包含不存在的 Episode")
        rows = []
        cancelled = False
        for index, eid in enumerate(ids):
            try:
                self._cancel(cancel)
                row = self._episode(eid, policy, cancel)
            except CancelledError:
                cancelled = True
                break
            rows.append(row)
            if progress:
                progress(index + 1, len(ids), eid)
        counts = Counter(row["decision"] for row in rows)
        stages = [
            ("数据完整性 / 时间戳", lambda name: name in {"data_integrity", "timestamps", "frame_indices", "frame_gaps", "state.finite", "action.finite"} or name.startswith("camera_clock:")),
            ("静止比例", lambda name: name.endswith(".static")),
            ("速度 / 加速度 Z-score", lambda name: name.endswith("_zscore")),
            ("Jerk", lambda name: name.endswith(".jerk")),
            ("图像质量", lambda name: name.startswith(("image_integrity", "blur:", "occlusion_proxy:", "repeated_image_proxy:"))),
            ("多视角时间同步", lambda name: name.startswith("multiview_")),
            ("视频—状态时间同步", lambda name: name.startswith("video_state_sync:")),
        ]
        remaining = rows[:]
        funnel = []
        for name, match in stages:
            before = len(remaining)
            remaining = [row for row in remaining if not any(match(reason) for reason in row["rejected_reasons"])]
            funnel.append({"stage": name, "input_episodes": before,
                           "removed_episodes": before - len(remaining), "remaining_episodes": len(remaining),
                           "remaining_frames": sum(row["length"] for row in remaining)})
        return {
            "version": 1, "source": str(self.model.loader.path),
            "policy": asdict(policy), "scope_episode_ids": ids,
            "complete": not cancelled and len(rows) == len(ids),
            "selection_unit": "episode", "source_modified": False,
            "summary": {
                "checked_episodes": len(rows), "total_requested": len(ids),
                "passed": counts["passed"], "review": counts["review"], "rejected": counts["rejected"],
                "checked_frames": sum(row["length"] for row in rows),
                "candidate_hours": sum(row["duration_seconds"] for row in rows if row["decision"] != "rejected") / 3600,
                "funnel": funnel,
                "candidate_frames": sum(row["length"] for row in rows if row["decision"] != "rejected"),
                "rejected_frames": sum(row["length"] for row in rows if row["decision"] == "rejected"),
            },
            "episodes": rows,
        }

    def _episode(self, eid, policy, cancel):
        metadata = self.model.loader.episodes[eid]
        checks = []

        def add(name, status, metrics=None, frames=(), message=""):
            checks.append({"name": name, "status": status, "metrics": metrics or {},
                           "frames": [int(i) for i in frames], "message": message})

        def finish():
            rejected = [c["name"] for c in checks if c["status"] == "rejected"]
            review = [c["name"] for c in checks if c["status"] in {"review", "unverified"}]
            return {
                "episode_index": eid, "tasks": self._episode_tasks.get(eid, []), "length": metadata["length"],
                "duration_seconds": metadata["length"] / self.model.loader.fps,
                "decision": "rejected" if rejected else "review" if review else "passed",
                "rejected_reasons": rejected, "review_reasons": review,
                "flagged_frames": sorted({i for check in checks for i in check["frames"]}),
                "checks": checks,
            }

        try:
            episode = self.model.read_episode(eid)
        except Exception as exc:
            add("data_integrity", "rejected", message=str(exc))
            return finish()
        times = episode.times
        if "timestamp" not in episode.values:
            add("timestamp_source", "unverified", message="时间戳由 FPS 推算，不能验证真实采样间隔")
        dt = np.diff(times)
        invalid = np.flatnonzero(dt <= 0) + 1
        if len(invalid):
            add("timestamps", "rejected", frames=invalid, message="时间戳未严格递增，停止差分检查")
            return finish()
        gaps = np.flatnonzero(dt > policy.frame_gap_factor / self.model.loader.fps) + 1
        add("frame_gaps", "rejected" if len(gaps) else "passed",
            {"max_gap_seconds": float(dt.max()) if len(dt) else None,
             "gap_limit_seconds": policy.frame_gap_factor / self.model.loader.fps}, gaps,
            "长间隔表明采样缺口，不能据此确认缺的是某个相机的帧")
        if "frame_index" in episode.values:
            indices = np.asarray(episode.values["frame_index"]).reshape(-1)
            bad = np.flatnonzero(indices != np.arange(episode.length))
            add("frame_indices", "rejected" if len(bad) else "passed", frames=bad)
        for role in ("state", "action"):
            self._signal(episode, role, getattr(policy, role), add)
        self._cancel(cancel)
        if policy.inspect_images:
            self._images(episode, policy, add, cancel)
        else:
            add("images", "unverified", message="本轮未检查图像")
        self._synchronization(episode, policy, add)
        add("semantic_occlusion", "unverified", message="低对比/遮黑仅作代理指标，未运行遮挡语义模型")
        return finish()

    def _signal(self, episode, role, rule, add):
        field = getattr(episode.schema, role)
        array = getattr(episode, role)
        if array is None:
            add(f"{role}.mapping", "unverified", message=f"请指定 {role} 字段")
            return
        dimensions = list(rule.dimensions) if rule.dimensions is not None else list(range(field.width))
        values = array[:, dimensions].astype(np.float64)
        invalid = np.flatnonzero(~np.isfinite(values).all(axis=1))
        if len(invalid):
            add(f"{role}.finite", "rejected", frames=invalid, message="存在 NaN / Inf")
            return
        epsilon = _threshold(rule.static_epsilon, field.width, dimensions)
        if rule.kind in {"position", "command"}:
            static = np.all(np.abs(np.diff(values, axis=0)) <= epsilon, axis=1)
        elif rule.kind == "velocity":
            static = np.all(np.abs(values) <= epsilon, axis=1)
        else:
            static = np.array([], dtype=bool)
        if len(static):
            ratio = float(static.mean())
            add(f"{role}.static", "rejected" if ratio > rule.max_static_ratio else "passed",
                {"ratio": ratio, "limit": rule.max_static_ratio, "dimensions": dimensions,
                 "epsilon": epsilon.tolist(), "kind": rule.kind},
                message="position/command 使用整组维度均不变化的相邻帧比例；velocity 使用近零值比例")
        else:
            add(f"{role}.static", "unverified", message="帧数不足或仅有加速度，无法推断静止")
        derivatives = _derivatives(values, episode.times, rule.kind)
        for order, name, threshold in (
            (1, "velocity_zscore", rule.velocity_zscore),
            (2, "acceleration_zscore", rule.acceleration_zscore),
            (3, "jerk", rule.jerk_limit),
        ):
            samples, frame_ids = derivatives.get(order, (np.empty((0, len(dimensions))), np.array([], dtype=int)))
            if not len(samples):
                add(f"{role}.{name}", "unverified", message="信号类型不提供此量或帧数不足")
                continue
            metrics = {"peak_abs_per_dim": np.max(np.abs(samples), axis=0).tolist(),
                       "dimensions": dimensions, "kind": rule.kind, "sample_count": len(samples)}
            if not np.isfinite(samples).all():
                add(f"{role}.{name}", "rejected", frames=frame_ids[~np.isfinite(samples).all(axis=1)],
                    message="差分结果出现非有限数值，请检查时间戳精度、信号范围及单位")
                continue
            if threshold is None:
                add(f"{role}.{name}", "unverified" if order == 3 else "disabled", metrics,
                    message="未配置阈值；command 的差分量不自动解释为物理速度/加速度/Jerk")
                continue
            if order == 3:
                limit = _threshold(threshold, field.width, dimensions)
                bad = np.any(np.abs(samples) > limit, axis=1)
                metrics["threshold_per_dim"] = limit.tolist()
            else:
                if len(samples) < 4:
                    add(f"{role}.{name}", "unverified", metrics, message="少于 4 个样本，Z-score 不可靠")
                    continue
                std = samples.std(axis=0)
                scores = np.divide(np.abs(samples - samples.mean(axis=0)), std,
                                   out=np.zeros_like(samples), where=std > 1e-12)
                bad = np.any(scores > threshold, axis=1)
                metrics.update(threshold=threshold, max_abs_z_per_dim=scores.max(axis=0).tolist(), reference="episode")
            ratio = float(bad.mean())
            metrics.update(bad_ratio=ratio, allowed_bad_ratio=rule.max_bad_ratio)
            add(f"{role}.{name}", "rejected" if ratio > rule.max_bad_ratio else "passed",
                metrics, frame_ids[bad])

    def _image_measurements(self, episode, policy, cancel):
        cache_key = (episode.index, policy.image_stride, policy.image_width)
        if cache_key in self._image_cache:
            self._image_cache.move_to_end(cache_key)
            return self._image_cache[cache_key]
        cameras = self.model.camera_keys
        samples = sorted(set(range(0, episode.length, policy.image_stride)) | {episode.length - 1})
        result = {camera: [] for camera in cameras}
        for index in samples:
            self._cancel(cancel)
            images, errors = self.model.loader.read_image_frame(episode.index, index, cameras)
            for camera in cameras:
                entry = {"frame": index}
                try:
                    if camera in errors:
                        raise ValueError(errors[camera])
                    array = images[camera]
                    image = Image.fromarray(array.transpose(1, 2, 0)).convert("L")
                    image = image.resize((policy.image_width, max(8, round(image.height * policy.image_width / image.width))),
                                         Image.Resampling.BILINEAR)
                    gray = np.asarray(image, dtype=float)
                    laplace = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
                    entry.update(blur_variance=float(laplace.var()), std=float(gray.std()),
                                 saturation=max(float((gray < 5).mean()), float((gray > 250).mean())),
                                 fingerprint=blake2b(image.tobytes(), digest_size=8).hexdigest())
                except Exception as exc:
                    entry["error"] = str(exc)
                result[camera].append(entry)
        self._image_cache[cache_key] = result
        if len(self._image_cache) > 16:
            self._image_cache.popitem(last=False)
        return result

    def _images(self, episode, policy, add, cancel):
        if self.model.video_keys:
            add("external_video", "unverified", message="当前读取器尚未支持外部视频解码")
        if not self.model.camera_keys:
            add("images", "unverified", message="没有可检查的内嵌图像")
            return
        try:
            measures = self._image_measurements(episode, policy, cancel)
        except CancelledError:
            raise
        except Exception as exc:
            add("image_integrity", "rejected", message=str(exc))
            return
        if policy.image_stride > 1:
            add("image_coverage", "unverified", {"stride": policy.image_stride}, message="抽样检查不能覆盖未采样的帧")
        state = episode.state
        movement = np.zeros(episode.length, dtype=bool)
        if state is not None and np.isfinite(state).all():
            dims = list(policy.state.dimensions) if policy.state.dimensions is not None else list(range(state.shape[1]))
            epsilon = _threshold(policy.state.static_epsilon, state.shape[1], dims)
            if policy.state.kind == "velocity":
                movement = np.any(np.abs(state[:, dims]) > epsilon, axis=1)
            elif policy.state.kind in {"position", "command"}:
                movement[1:] = np.any(np.abs(np.diff(state[:, dims], axis=0)) > epsilon, axis=1)
        for camera, samples in measures.items():
            errors = [s["frame"] for s in samples if "error" in s]
            valid = [s for s in samples if "error" not in s]
            add(f"image_integrity:{camera}", "rejected" if errors else "passed",
                {"sampled": len(samples), "total": episode.length,
                 "errors": [s for s in samples if "error" in s]}, errors)
            if not valid:
                continue
            scores = np.array([s["blur_variance"] for s in valid])
            metrics = {"min": float(scores.min()), "median": float(np.median(scores)),
                       "resize_width": policy.image_width, "sampled": len(valid)}
            if policy.blur_variance_min is None:
                add(f"blur:{camera}", "unverified", metrics, message="请先根据场景与固定缩放尺寸校准清晰度阈值")
            else:
                bad = [s["frame"] for s in valid if s["blur_variance"] < policy.blur_variance_min]
                ratio = len(bad) / len(valid)
                metrics.update(bad_ratio=ratio, threshold=policy.blur_variance_min)
                add(f"blur:{camera}", "rejected" if ratio > policy.max_bad_image_ratio else "passed", metrics, bad)
            uniform = [s["frame"] for s in valid if s["std"] < policy.uniform_std_max and s["saturation"] >= policy.saturation_fraction]
            add(f"occlusion_proxy:{camera}", "review" if uniform else "passed",
                {"count": len(uniform)}, uniform, "近乎纯黑/纯白低纹理帧，只作为遮挡或曝光异常线索")
            repeated = []
            run = []
            for sample in samples + [{"frame": episode.length}]:
                if run and ("fingerprint" not in sample or sample["fingerprint"] != run[-1]["fingerprint"]):
                    if len(run) >= policy.repeated_run and movement[run[0]["frame"]:run[-1]["frame"]+1].any():
                        repeated.extend(s["frame"] for s in run)
                    run = []
                if "fingerprint" in sample:
                    run.append(sample)
            add(f"repeated_image_proxy:{camera}", "review" if repeated else "passed",
                {"minimum_run": policy.repeated_run}, repeated,
                "机器人有变化时画面重复，仅提示疑似冻结；不能等同于掉帧")

    def _synchronization(self, episode, policy, add):
        clocks = {}
        state_time = episode.values.get(policy.state_timestamp_key)
        if state_time is not None:
            state_time = np.asarray(state_time).reshape(-1)
        for camera in self.model.loader.camera_keys:
            key = policy.camera_timestamp_keys.get(camera)
            if not key:
                add(f"video_state_sync:{camera}", "unverified", message="没有独立的相机采集时间戳，同行存储不构成同步证据")
                continue
            times = np.asarray(episode.values[key], dtype=float).reshape(-1)
            if len(times) != episode.length or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                add(f"camera_clock:{camera}", "rejected", message="相机时间戳数量、有限性或递增性无效")
                continue
            clocks[camera] = times
            if state_time is None or len(state_time) != episode.length or not np.isfinite(state_time).all():
                add(f"video_state_sync:{camera}", "unverified", message="缺少有效的状态采集时间戳")
            else:
                delta = np.abs(times - state_time)
                bad = np.flatnonzero(delta > policy.sync_tolerance_seconds)
                add(f"video_state_sync:{camera}", "rejected" if len(bad) else "passed",
                    {"max_skew_seconds": float(delta.max()), "limit_seconds": policy.sync_tolerance_seconds}, bad,
                    "要求配置的时间戳处于同一时钟域")
        if len(clocks) >= 2:
            stack = np.array(list(clocks.values()))
            skew = stack.max(axis=0) - stack.min(axis=0)
            bad = np.flatnonzero(skew > policy.sync_tolerance_seconds)
            add("multiview_time_sync", "rejected" if len(bad) else "passed",
                {"max_skew_seconds": float(skew.max()), "cameras": list(clocks)}, bad)
        if len(self.model.loader.camera_keys) > 1 and len(clocks) != len(self.model.loader.camera_keys):
            add("multiview_time_coverage", "unverified", message="没有覆盖全部相机的独立采集时间戳")
        if len(self.model.loader.camera_keys) > 1:
            add("multiview_geometry", "unverified", message="未提供外参/共同视觉事件，未验证空间一致性")


def save_quality_report(report, path):
    """保存完整阈值快照、逐帧异常索引和检查覆盖；不写入原始数据集。"""
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(Path(report["source"]).resolve()):
        raise ValueError("请将质量报告保存到源数据集目录之外")
    path.write_text(json.dumps(json_value(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
