"""
LeRobot v2 数据访问：
    Parquet 读取
    字段适配
    Task 统计
    Episode 数值视图

公共入口
    LeRobotReader(原始数据)与 BrowserModel(浏览和分析)
    
    !该模块不依赖桌面界面，不修改原始数据集。
"""

from bisect import bisect_right
from dataclasses import dataclass
from io import BytesIO
import json
from math import prod
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from .config import DEFAULT_DATA_PATH, FIELD_ALIASES


class LeRobotReader:
    """按需读取轨迹，缓存当前轨迹最近使用的两组 Parquet 列。

    初始化只读取 meta 下的元数据。read_episode 返回按字段组织的数据，
    第 0 维为时间；read_frame 和 __getitem__ 返回单帧字段字典。
    数值与相机分别缓存，切换轨迹时释放旧表。
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_DATA_PATH,
        decode_images: bool = True,
    ):
        self.path = Path(path).expanduser().resolve()
        self.decode_images = decode_images
        with (self.path / "meta/info.json").open(encoding="utf-8") as stream:
            self.info = json.load(stream)
        if self.info.get("codebase_version") not in {"v2.0", "v2.1"}:
            raise ValueError("仅支持每条轨迹一个 Parquet 文件的 LeRobot v2.x 数据集")
        self.features = self.info["features"]
        self.fps = self.info["fps"]
        self.episodes = {
            item["episode_index"]: item
            for item in self._read_jsonl("episodes.jsonl")
        }
        self.tasks = {
            item["task_index"]: item["task"]
            for item in self._read_jsonl("tasks.jsonl")
        }
        self.episode_indices = sorted(self.episodes)
        self.num_episodes = len(self.episode_indices)
        self._offsets = [0]
        for index in self.episode_indices:
            self._offsets.append(self._offsets[-1] + self.episodes[index]["length"])
        self.num_frames = self._offsets[-1]
        self.camera_keys = [
            key for key, feature in self.features.items()
            if feature["dtype"] in {"image", "video"}
        ]
        self._cached_episode = None
        self._tables = {}

    def _read_jsonl(self, name: str) -> list[dict]:
        with (self.path / "meta" / name).open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def __len__(self) -> int:
        return self.num_frames

    def _read_table(self, episode_index: int, columns: Sequence[str] | None = None) -> pa.Table:
        if episode_index not in self.episodes:
            raise IndexError(f"轨迹不存在: {episode_index}")
        selected = tuple(self.features) if columns is None else tuple(columns)
        if not selected:
            raise ValueError("columns 不能为空")
        unknown = set(selected) - self.features.keys()
        if unknown:
            raise KeyError(f"未知字段: {sorted(unknown)}")
        video_keys = [key for key in selected if self.features[key]["dtype"] == "video"]
        if video_keys:
            raise NotImplementedError(f"暂不支持外部视频读取: {video_keys}")
        if episode_index != self._cached_episode:
            self._tables.clear()
            self._cached_episode = episode_index
        table = self._tables.pop(selected, None)
        if table is None:
            relative_path = self.info["data_path"].format(
                episode_chunk=episode_index // self.info["chunks_size"],
                episode_index=episode_index,
            )
            # ParquetFile 不会将单个文件的读取扩展为整个数据集扫描。
            table = pq.ParquetFile(self.path / relative_path).read(columns=list(selected))
            expected = self.episodes[episode_index]["length"]
            if table.num_rows != expected:
                raise ValueError(
                    f"轨迹 {episode_index} 帧数与元数据不符: {table.num_rows} != {expected}"
                )
        self._tables[selected] = table
        if len(self._tables) > 2:
            del self._tables[next(iter(self._tables))]
        return table

    def _decode_image(self, value: dict) -> np.ndarray:
        if value.get("bytes") is not None:
            source = BytesIO(value["bytes"])
        elif value.get("path"):
            source = self.path / value["path"]
        else:
            raise ValueError("图像记录缺少 bytes 和 path")
        with Image.open(source) as image:
            return np.asarray(image.convert("RGB")).transpose(2, 0, 1).copy()

    def _convert_frame(self, row: dict) -> dict:
        result = {}
        for key, value in row.items():
            dtype = self.features[key]["dtype"]
            if dtype == "image":
                result[key] = self._decode_image(value) if self.decode_images else value
            else:
                array = np.asarray(value, dtype=dtype)
                result[key] = array[()] if array.ndim == 0 else array
        if "task_index" in result:
            result["task"] = self.tasks[int(result["task_index"])]
        return result

    def read_frame(
        self,
        episode_index: int,
        frame_index: int = 0,
        columns: Sequence[str] | None = None,
    ) -> dict:
        """读取一条轨迹内的某一帧，frame_index 从 0 开始，支持负索引。"""
        if episode_index not in self.episodes:
            raise IndexError(f"轨迹不存在: {episode_index}")
        length = self.episodes[episode_index]["length"]
        if frame_index < 0:
            frame_index += length
        if not 0 <= frame_index < length:
            raise IndexError(f"帧索引超出轨迹 {episode_index} 范围: {frame_index}")
        table = self._read_table(episode_index, columns)
        return self._convert_frame(table.slice(frame_index, 1).to_pylist()[0])

    def __getitem__(self, index: int) -> dict:
        """按全局帧索引读取，轨迹按 episode_index 升序排列，支持负索引。"""
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"全局帧索引越界: {index}")
        position = bisect_right(self._offsets, index) - 1
        return self.read_frame(self.episode_indices[position], index - self._offsets[position])

    def iter_frames(self, episode_index: int, columns: Sequence[str] | None = None) -> Iterator[dict]:
        """逐帧读取一条轨迹，图像在迭代到对应帧时才解码。"""
        table = self._read_table(episode_index, columns)
        for index in range(table.num_rows):
            yield self._convert_frame(table.slice(index, 1).to_pylist()[0])

    def read_episode(self, episode_index: int, columns: Sequence[str] | None = None) -> dict:
        """读取完整轨迹；建议用 columns 只读取需要的字段以节省内存。

        数值字段沿时间维堆叠为数组，图像和任务文本返回列表。
        """
        result = {}
        for frame in self.iter_frames(episode_index, columns):
            for key, value in frame.items():
                result.setdefault(key, []).append(value)
        for key in result:
            if key != "task" and key not in self.camera_keys:
                result[key] = np.asarray(result[key], dtype=self.features[key]["dtype"])
        return result


@dataclass(frozen=True)
class VectorField:
    key: str | None
    names: tuple[str, ...]

    @property
    def width(self) -> int:
        return len(self.names)

    def series(self, values: dict, length: int) -> np.ndarray | None:
        """为绘图提供 (帧数, 维度) 视图，不修改原始数组。"""
        if self.key is None:
            return None
        array = np.asarray(values[self.key])
        if array.ndim == 0 or array.shape[0] != length or prod(array.shape[1:]) != self.width:
            raise ValueError(f"字段 {self.key} 的实际形状 {array.shape} 与 {self.width} 维元数据不符")
        return array.reshape(length, self.width)


def _vector_field(features: dict, role: str, explicit: str | None) -> VectorField:
    if explicit is not None:
        if explicit not in features:
            raise ValueError(f"字段 {explicit!r} 不存在，可用字段: {', '.join(features)}")
        key = explicit
    else:
        aliases = FIELD_ALIASES[role]
        candidates = [key for key in features if key.casefold() in aliases]
        if len(candidates) > 1:
            raise ValueError(f"{role} 字段有多个候选 {candidates}，请使用 --{role}-key 明确指定")
        key = candidates[0] if candidates else None
    if key is None:
        return VectorField(None, ())
    feature = features[key]
    try:
        numeric = np.dtype(feature["dtype"]).kind in "biuf"
    except TypeError:
        numeric = False
    if not numeric:
        raise ValueError(f"{role} 字段 {key!r} 必须是实数数值字段")
    width = prod(feature.get("shape", [1]))
    if width < 1:
        raise ValueError(f"字段 {key!r} 的 shape 不能为空维度")
    names = feature.get("names")
    while isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    if not isinstance(names, list) or len(names) != width or not all(isinstance(n, str) for n in names):
        names = [f"dim_{i}" for i in range(width)]
    return VectorField(key, tuple(names))


@dataclass(frozen=True)
class DatasetSchema:
    state: VectorField
    action: VectorField

    @classmethod
    def from_features(
        cls, features: dict, *, state_key: str | None = None, action_key: str | None = None
    ) -> "DatasetSchema":
        state = _vector_field(features, "state", state_key)
        action = _vector_field(features, "action", action_key)
        if state.key is not None and state.key == action.key:
            raise ValueError("State 与 Action 不能映射到同一个字段")
        return cls(state, action)

    def summary(self) -> str:
        return " / ".join(
            f"{role}: {field.key} ({field.width} 维)" if field.key else f"{role}: 未映射 (--{role.lower()}-key)"
            for role, field in (("State", self.state), ("Action", self.action))
        )


@dataclass(frozen=True)
class TaskStats:
    name: str
    episode_ids: tuple[int, ...]
    total_frames: int
    fps: float

    @property
    def mean_frames(self) -> float:
        return self.total_frames / len(self.episode_ids) if self.episode_ids else 0.0

    @property
    def mean_seconds(self) -> float:
        return self.mean_frames / self.fps


@dataclass
class EpisodeData:
    index: int
    metadata: dict
    values: dict
    times: np.ndarray
    schema: DatasetSchema

    @property
    def length(self) -> int:
        return len(self.times)

    def frame(self, index: int) -> dict:
        if not 0 <= index < self.length:
            raise IndexError(index)
        return {key: value[index] for key, value in self.values.items()}

    @property
    def state(self) -> np.ndarray | None:
        return self.schema.state.series(self.values, self.length)

    @property
    def action(self) -> np.ndarray | None:
        return self.schema.action.series(self.values, self.length)


class BrowserModel:
    """通过同一读取器提供统计、数值与图像；GUI 在单一工作线程中调用。"""

    def __init__(
        self, path: str | Path = DEFAULT_DATA_PATH, *,
        state_key: str | None = None, action_key: str | None = None,
    ):
        self.loader = LeRobotReader(path)
        self.schema = DatasetSchema.from_features(
            self.loader.features, state_key=state_key, action_key=action_key
        )
        if self.loader.fps <= 0:
            raise ValueError("数据集 FPS 必须大于 0")
        self.numeric_keys = tuple(
            key for key, feature in self.loader.features.items()
            if feature["dtype"] not in {"image", "video"}
        )
        self.camera_keys = tuple(
            key for key in self.loader.camera_keys
            if self.loader.features[key]["dtype"] == "image"
        )
        self.video_keys = tuple(
            key for key in self.loader.camera_keys
            if self.loader.features[key]["dtype"] == "video"
        )
        self.task_stats = self._build_task_stats()

    def _build_task_stats(self) -> list[TaskStats]:
        by_name = {name: [] for name in self.loader.tasks.values()}
        for eid in self.loader.episode_indices:
            metadata = self.loader.episodes[eid]
            names = metadata.get("tasks", metadata.get("task"))
            if isinstance(names, str):
                names = [names]
            if not names:
                indices = metadata.get("task_indices", metadata.get("task_index", []))
                if not isinstance(indices, (list, tuple)):
                    indices = [indices]
                names = [self.loader.tasks.get(i, f"task_index={i}") for i in indices]
            for name in dict.fromkeys(names or ["（未标记 Task）"]):
                by_name.setdefault(name, []).append(eid)
        return [
            TaskStats(
                name, tuple(ids),
                sum(self.loader.episodes[eid]["length"] for eid in ids),
                float(self.loader.fps),
            )
            for name, ids in sorted(by_name.items())
        ]

    def read_episode(self, index: int) -> EpisodeData:
        values = self.loader.read_episode(index, columns=self.numeric_keys)
        length = self.loader.episodes[index]["length"]
        if length <= 0:
            raise ValueError(f"Episode {index} 没有帧")
        times = np.asarray(
            values.get("timestamp", np.arange(length) / self.loader.fps),
            dtype=float,
        ).reshape(-1)
        if len(times) != length or not np.all(np.isfinite(times)):
            raise ValueError("时间戳数量不符或包含非有限值")
        if np.any(np.diff(times) < 0):
            raise ValueError("时间戳必须按时间递增")
        # 在后台读取阶段验证维度，让错误与其他读取错误使用同一条界面提示通道。
        for field in (self.schema.state, self.schema.action):
            field.series(values, length)
        return EpisodeData(index, self.loader.episodes[index], values, times, self.schema)

    def read_images(self, episode: int, frame: int) -> dict:
        if not self.camera_keys:
            return {}
        return self.loader.read_frame(episode, frame, columns=self.camera_keys)


def json_value(value):
    """将数值字段转为可检查的 JSON；图像像素在相机区检查。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value
