# LeRobot / VLA 数据工具

支持 LeRobot v2.0 / v2.1 的本地读取，以及 Task → Episode → Frame 交互式浏览。
原始数据集保持只读；字段别名、向量维度和相机数量可随数据集变化。

## 框架

```text
data_process/
├── config.py        # 默认数据路径、State / Action 字段别名
├── dataset.py       # 数据读取、字段适配、Task 统计、Episode 数据模型
├── charts.py        # 轨迹曲线、Task 统计窗口、公共表格
├── viewer.py        # 桌面布局、导航、后台读取、播放与帧同步
├── __main__.py      # 命令行参数及 read / view 调度
├── __init__.py      # 公共数据 API
├── requirements.txt
└── README.md
```

依赖方向：`config → dataset → charts / viewer`。
数据模块不导入 GUI，终端读取和其他数据处理程序可以单独使用。
字段适配与统计使用同一份元数据；数值和相机共用一个读取器，缓存当前轨迹的两组列。

## 启动

在项目根目录运行：

```bash
# 独立桌面窗口；省略 view 也可以
.venv/bin/python -m data_process view

# 浏览 Piper，定位第 0 条轨迹
.venv/bin/python -m data_process view --path playground/Dataset/piper_dataset --episode 0

# 终端预览一帧
.venv/bin/python -m data_process read --path playground/Dataset/piper_dataset --episode 0 --frame 10
```

默认数据路径在 `config.py` 的 `DEFAULT_DATA_PATH` 中设置，`--path` 可以覆盖。
桌面窗口需要可用的显示服务。依赖安装：

```bash
uv pip install --python .venv/bin/python -r data_process/requirements.txt
```

Python 还需要 Tk 支持（`import tkinter`）。

## 字段适配

修改 `config.py` 中的 `FIELD_ALIASES` 可以追加新字段名。常见字段自动识别：

| 角色 | 原数据集 | Piper |
| --- | --- | --- |
| State | `observation.state`，14 维 | `state`，7 维 |
| Action | `action`，14 维 | `actions`，7 维 |
| 相机 | 3 路 | `image` / `wrist_image`，2 路 |

自动识别有多个候选或字段名不在配置中时，显式指定：

```bash
.venv/bin/python -m data_process view --path /path/to/dataset --state-key qpos --action-key command
```

关节名来自元数据，缺少逐维名称时使用 `dim_0` 等。多轴数值按原始顺序展平用于绘图；
原始字段名和形状保持不变。图像统一解码为 CHW RGB `uint8`。
State / Action 按维度索引对比，不推断单位、坐标系或动作含义。
当前不支持外部视频、LeRobot v3 或 HDF5；这些存储结构需要独立的读取适配。

## 界面操作

- Task：搜索任务名，筛选 Episode；统计窗口支持排序、分布图、CSV 导出和点击跳转。
- Episode：列表选择、按编号定位、前后切换；查看帧数、FPS、时长和完整元数据。
- Frame：滑动条、帧号、左右键逐帧查看；空格播放 / 暂停，支持倍速与循环。
- 图像：同步显示全部内嵌相机，双击查看当前帧的原始分辨率快照。
- 曲线：显示全部 State / Action 维度，支持多选、缩放、平移、悬停查值和单击跳帧。
- 数值：右侧展示每一维的 State、Action、差值及当前帧全部原始数值字段。

界面中的图像、曲线游标和数值同步更新，过期读取结果不会覆盖新选择。
播放以 FPS 为目标速度，实际速度受解码和刷新耗时影响。
统计时长采用 `帧数 / FPS`；首尾时间戳差在完整元数据中单独显示。
多任务轨迹会分别计入各 Task，因此跨 Task 汇总可能重复。

## 数据 API

```python
from data_process import LeRobotReader, BrowserModel

reader = LeRobotReader("playground/Dataset/piper_dataset")
frame = reader.read_frame(episode_index=0, frame_index=10)
raw_episode = reader.read_episode(0, columns=["state", "actions"])

model = BrowserModel("playground/Dataset/piper_dataset")
episode = model.read_episode(0)
state = episode.state           # (帧数, 实际维度数)
action = episode.action
values = episode.frame(10)      # 原始字段名和形状
images = model.read_images(0, 10)
stats = model.task_stats
```
