"""浏览器应用：数值与图像视图、质量任务和 JSON 数据转换；不依赖 HTTP。"""

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from io import BytesIO
import math
from pathlib import Path
import secrets
from threading import Event, Lock, RLock

from PIL import Image

from .config import QUALITY_POLICY, quality_policy_from_dict
from .dataset import BrowserModel, json_value
from .processing import QualityInspector


def clean_json(value):
    """JSON 不允许 NaN/Inf；保留字段并以 null 表示，以便检查异常数据。"""
    value = json_value(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


class BrowserApplication:
    def __init__(self, model, episode=None):
        self.model = model
        self.initial_episode = episode
        self.token = secrets.token_urlsafe(32)
        self.dataset_id = secrets.token_urlsafe(12)
        self.data_lock = RLock()
        self.job_lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="web-quality")
        self.cancel = Event()
        self.job = None
        self.inspector = QualityInspector(model.loader.path, state_key=model.schema.state.key,
                                          action_key=model.schema.action.key)
        self.episode = None

    def info(self):
        with self.data_lock:
            model = self.model
            return dict(path=str(model.loader.path), dataset_id=self.dataset_id, fps=model.loader.fps,
                        total_frames=len(model.loader), total_episodes=model.loader.num_episodes,
                        tasks=[asdict(row) for row in model.task_stats], episodes=model.loader.episodes,
                        schema=asdict(model.schema), cameras=model.camera_keys, videos=model.video_keys,
                        initial_episode=self.initial_episode, policy=asdict(QUALITY_POLICY))

    def open_dataset(self, path):
        """先验证新数据集，再整体替换；失败时保留当前数据和质量结果。"""
        if not isinstance(path, str) or not path.strip():
            raise ValueError("请输入数据集根目录")
        with self.data_lock:
            with self.job_lock:
                if self.job and self.job["status"] == "running":
                    raise ValueError("质量检查正在运行，请先在质量筛选面板停止检查，再切换数据集")
            try:
                model = BrowserModel(path.strip())
                # 验证首条轨迹可读，避免无效路径或文件损坏清空当前页面。
                episode = model.read_episode(model.loader.episode_indices[0]) if model.loader.episode_indices else None
                inspector = QualityInspector(model.loader.path, state_key=model.schema.state.key,
                                             action_key=model.schema.action.key)
            except (OSError, ValueError, KeyError, IndexError) as exc:
                raise ValueError(f"无法打开数据集：{exc}") from exc
            with self.job_lock:
                self.model = model
                self.inspector = inspector
                self.episode = episode
                self.initial_episode = None
                self.job = None
                self.cancel.clear()
                self.dataset_id = secrets.token_urlsafe(12)
            return self.info()

    def directories(self, path=None):
        """浏览服务所在机器的目录，不读取或上传浏览器客户端文件。"""
        if path is not None and not isinstance(path, str):
            raise ValueError("目录路径必须为字符串")
        directory = Path(path.strip()).expanduser().resolve() if path and path.strip() else self.model.loader.path
        if not directory.is_dir():
            raise ValueError(f"目录不存在：{directory}")
        children = []
        for child in directory.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                children.append(dict(name=child.name, path=str(child)))
        return dict(path=str(directory), parent=str(directory.parent),
                    directories=sorted(children, key=lambda row: row["name"].casefold()),
                    is_dataset=(directory / "meta/info.json").is_file())

    def _get_episode(self, index):
        # 调用方持有 data_lock，避免读取器的两组列缓存被并发请求打断。
        if self.episode is None or self.episode.index != index:
            self.episode = self.model.read_episode(index)
        return self.episode

    def episode_data(self, index):
        with self.data_lock:
            episode = self._get_episode(index)
            return clean_json(dict(index=index, length=episode.length, metadata=episode.metadata,
                                   times=episode.times, state=episode.state, action=episode.action,
                                   values=episode.values))

    def frame_data(self, index, frame):
        with self.data_lock:
            episode = self._get_episode(index)
            values = episode.frame(frame)
            images = {}
            arrays, errors = self.model.loader.read_image_frame(index, frame, self.model.camera_keys)
            for camera, array in arrays.items():
                try:
                    image = Image.fromarray(array.transpose(1, 2, 0))
                    image.thumbnail((640, 480))
                    stream = BytesIO()
                    image.save(stream, format="JPEG", quality=85)
                    images[camera] = "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
                except Exception as exc:
                    errors[camera] = str(exc)
            return clean_json(dict(episode=index, frame=frame, values=values, images=images, errors=errors))

    def image_data(self, index, frame, camera):
        if camera not in self.model.camera_keys:
            raise ValueError("未知相机")
        with self.data_lock:
            array = self.model.loader.read_frame(index, frame, [camera])[camera]
            stream = BytesIO()
            Image.fromarray(array.transpose(1, 2, 0)).save(stream, format="PNG")
            return stream.getvalue()

    def start_quality(self, body):
        with self.data_lock:
            policy = quality_policy_from_dict(body.get("policy", {}))
            self.inspector.validate_policy(policy)
            ids = body.get("episode_ids")
            if not isinstance(ids, list) or not ids or any(type(i) is not int or i not in self.model.loader.episodes for i in ids):
                raise ValueError("检查范围必须包含有效 Episode 编号")
            ids = list(dict.fromkeys(ids))
            with self.job_lock:
                if self.job and self.job["status"] == "running":
                    raise ValueError("已有质量检查正在运行，请等待或停止")
                job_id = secrets.token_urlsafe(12)
                self.cancel.clear()
                self.job = dict(id=job_id, status="running", completed=0, total=len(ids), report=None, error=None)
            self.executor.submit(self._inspect, ids, policy)
            return dict(id=job_id)

    def _inspect(self, ids, policy):
        def progress(count, total, eid):
            with self.job_lock:
                self.job.update(completed=count, total=total, episode=eid)
        try:
            report = self.inspector.inspect(ids, policy, progress=progress, cancel=self.cancel)
            with self.job_lock:
                self.job.update(status="done", report=report)
        except Exception as exc:
            with self.job_lock:
                self.job.update(status="error", error=str(exc))

    def quality_status(self, job_id):
        with self.job_lock:
            if self.job is None or self.job["id"] != job_id:
                raise KeyError("检查任务不存在或已被新一轮检查替换")
            return clean_json(self.job)

    def stop_quality(self, job_id):
        with self.job_lock:
            if self.job is None or self.job["id"] != job_id:
                raise KeyError("检查任务不存在或已被新一轮检查替换")
            self.cancel.set()
        return dict(stopped=True)

    def close(self):
        self.cancel.set()
        self.executor.shutdown(wait=True, cancel_futures=True)

