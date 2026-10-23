"""数据集默认路径、字段映射及质量检查策略。"""

from pathlib import Path
from dataclasses import asdict, dataclass, field
import math

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "playground/Dataset/demo_clean_repo"

# 语义角色 -> 数据集中可能使用的字段名。适配新字段时，在此追加别名。
# 多个候选同时存在时，使用 --state-key / --action-key 指定实际字段。
FIELD_ALIASES = {
    "state": ("observation.state", "observations.state", "state", "robot_state"),
    "action": ("action", "actions", "robot_action"),
}


@dataclass(frozen=True)
class SignalPolicy:
    """阈值使用原始字段单位；元组阈值按原始维度逐一指定。"""

    kind: str = "position"  # position / velocity / acceleration / command
    dimensions: tuple[int, ...] | None = None
    static_epsilon: float | tuple[float, ...] = 1e-6
    max_static_ratio: float = 0.95
    velocity_zscore: float | None = 6.0
    acceleration_zscore: float | None = 6.0
    jerk_limit: float | tuple[float, ...] | None = None
    max_bad_ratio: float = 0.01

    def __post_init__(self):
        if self.kind not in {"position", "velocity", "acceleration", "command"}:
            raise ValueError("信号类型必须为 position / velocity / acceleration / command")
        for name in ("max_static_ratio", "max_bad_ratio"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} 必须在 [0, 1] 内")
        for name in ("static_epsilon", "jerk_limit", "velocity_zscore", "acceleration_zscore"):
            value = getattr(self, name)
            if value is not None:
                values = value if isinstance(value, tuple) else (value,)
                if not values or any(not math.isfinite(v) or v < 0 for v in values):
                    raise ValueError(f"{name} 必须为非负有限数值")
        if self.dimensions is not None and (
            not self.dimensions or len(set(self.dimensions)) != len(self.dimensions)
            or any(not isinstance(i, int) or i < 0 for i in self.dimensions)
        ):
            raise ValueError("维度必须是互不重复的非负整数")


@dataclass(frozen=True)
class QualityPolicy:
    state: SignalPolicy = field(default_factory=SignalPolicy)
    action: SignalPolicy = field(default_factory=lambda: SignalPolicy(kind="command"))
    inspect_images: bool = True
    image_stride: int = 1
    image_width: int = 256
    blur_variance_min: float | None = None  # 需对固定图像尺寸校准后填写
    max_bad_image_ratio: float = 0.01
    uniform_std_max: float = 2.0
    saturation_fraction: float = 0.99
    repeated_run: int = 5
    frame_gap_factor: float = 1.5
    sync_tolerance_seconds: float = 0.05
    # 必须由用户指定同一时钟域的真实采集时间戳；缺失时同步检查为未验证。
    state_timestamp_key: str = "timestamp"
    camera_timestamp_keys: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.image_stride < 1 or self.image_width < 8 or self.repeated_run < 2:
            raise ValueError("图像步长至少为 1、缩略图宽度至少为 8、重复帧长度至少为 2")
        if not 0 <= self.max_bad_image_ratio <= 1 or not 0 < self.saturation_fraction <= 1:
            raise ValueError("图像异常比例必须在 [0, 1] 内，饱和比例必须在 (0, 1] 内")
        for name in ("uniform_std_max", "frame_gap_factor", "sync_tolerance_seconds", "blur_variance_min"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} 必须为非负有限数值")
        if self.frame_gap_factor <= 1:
            raise ValueError("掉帧间隔倍数必须大于 1")


QUALITY_POLICY = QualityPolicy()


def quality_policy_from_dict(values: dict) -> QualityPolicy:
    """将浏览器 / JSON 参数还原为质量策略，并检查字段及类型。"""
    if not isinstance(values, dict):
        raise ValueError("质量参数必须为 JSON 对象")
    defaults = asdict(QUALITY_POLICY)
    if values.keys() - defaults.keys():
        raise ValueError("质量参数包含未知字段")
    for role in ("state", "action"):
        rule = values.get(role, defaults[role])
        if not isinstance(rule, dict):
            raise ValueError(f"{role} 参数必须为对象")
        rule = {**defaults[role], **rule}
        for key in ("dimensions", "static_epsilon", "jerk_limit"):
            if isinstance(rule.get(key), list):
                rule[key] = tuple(rule[key])
        defaults[role] = SignalPolicy(**rule)
    defaults.update({key: value for key, value in values.items() if key not in {"state", "action"}})
    for key in ("image_stride", "image_width", "repeated_run"):
        if type(defaults[key]) is not int:
            raise ValueError(f"{key} 必须为整数")
    if type(defaults["inspect_images"]) is not bool:
        raise ValueError("inspect_images 必须为布尔值")
    clocks = defaults["camera_timestamp_keys"]
    if not isinstance(clocks, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in clocks.items()):
        raise ValueError("相机时间戳映射必须为字段名到字段名的对象")
    if not isinstance(defaults["state_timestamp_key"], str):
        raise ValueError("状态时间戳字段必须为字符串")
    return QualityPolicy(**defaults)
