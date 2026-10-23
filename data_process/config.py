"""数据集默认路径及字段别名映射。"""

from pathlib import Path

DEFAULT_DATA_PATH = Path("/home/ma/vla/playground/Dataset/demo_clean_repo")

# 语义角色 -> 数据集中可能使用的字段名。适配新字段时，在此追加别名。
# 多个候选同时存在时，使用 --state-key / --action-key 指定实际字段。
FIELD_ALIASES = {
    "state": ("observation.state", "observations.state", "state", "robot_state"),
    "action": ("action", "actions", "robot_action"),
}
