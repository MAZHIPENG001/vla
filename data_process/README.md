# LeRobot / VLA 数据工具

在本地浏览器中按 **Task → Episode → Frame** 浏览 LeRobot v2.0 / v2.1 数据，检查质量并预览候选子集。
源数据集只读。程序依赖 NumPy、Pillow 和 PyArrow，HTTP 服务使用 Python 标准库，前端使用原生 ES 模块和 Canvas。

## 启动

在项目根目录运行：

```bash
uv pip install --python .venv/bin/python -r data_process/requirements.txt

# 启动服务并打开浏览器；省略 view 也可以
.venv/bin/python -m data_process view

# 指定数据集和初始轨迹
.venv/bin/python -m data_process view --path playground/Dataset/piper_dataset --episode 0

# 自定义端口，不自动打开浏览器
.venv/bin/python -m data_process view --port 9000 --no-browser

# 在终端读取一帧
.venv/bin/python -m data_process read --path playground/Dataset/piper_dataset --episode 0 --frame 10
```

默认地址为 **http://127.0.0.1:8765**，按 `Ctrl+C` 停止。默认数据路径为项目下的
`playground/Dataset/demo_clean_repo`，可用 `--path` 覆盖。无需桌面显示服务、前端构建或 CDN。

远程访问可使用 `ssh -L 8765:127.0.0.1:8765 user@server`，然后在远端运行 `view --no-browser`。
`--host 0.0.0.0` 可供受信任局域网访问；服务没有登录系统，默认仅监听本机。

## 结构与职责

```text
data_process/
├── __main__.py      # 命令行：read / view
├── __init__.py      # 公共数据 API
├── config.py        # 默认路径、字段别名、质量策略及参数解析
├── dataset.py       # Parquet 读取、图像解码、字段适配与 Task / Episode 模型
├── processing.py    # 质量检查、指标缓存、报告与分阶段统计
├── application.py   # 浏览数据、图像编码、后台质量任务和 JSON 转换
├── web_server.py    # HTTP 路由、静态资源、服务生命周期
├── web/
│   ├── index.html   # 页面结构
│   ├── style.css    # 页面样式和窄屏布局
│   ├── common.js    # 共享浏览状态、数据范围与通用操作
│   ├── app.js       # 页面入口、导航、帧同步和播放
│   ├── chart.js     # 曲线、维度选择与缩放手势
│   ├── statistics.js # Task 统计、分布图和 CSV 下载
│   └── quality.js   # 质量表单、任务轮询、报告与筛选预览
├── tests/test_web.py # 数据适配、质量任务与只读回归测试
├── requirements.txt
└── README.md
```

依赖方向为 `config → dataset → processing → application → web_server`。
命令行负责启动；数据和质量模块可单独使用，不导入 HTTP 或界面。`web_server.py` 只处理传输，
`application.py` 负责串行访问浏览读取器，并在独立线程中运行质量检查。
前端功能模块通过入口传入跳转、筛选回调，共享浏览状态，不互相循环导入。

浏览 API：`GET /api/info`、`/api/episode?episode=0`、`/api/frame?episode=0&frame=10`、
`/api/image?episode=0&frame=10&camera=image`。
质量任务：`POST /api/quality/start`、`POST /api/quality/stop`、`GET /api/quality/status?id=…`。
POST 使用页面令牌和同源校验；源数据没有写入接口。

## 页面操作

- 任务：搜索、筛选轨迹；统计面板支持排序、长度分布、CSV 下载和点击跳转。
- 轨迹：列表选择、编号定位、前后切换及元数据查看。
- 帧：拖动滑条、输入帧号、左右键逐帧；空格播放 / 暂停，支持倍速与循环。
- 图像：同步显示内嵌相机的 JPEG 缩略图，点击读取原始分辨率 PNG；单路解码失败单独提示。
- 曲线：State / Action 多维选择、滚轮缩放、Shift 拖动平移、悬停查值、单击跳帧。
- 数值：逐维 State、Action、Action − State，以及当前帧原始数值字段。

图像、曲线和数值同步提交，过期响应不会覆盖新选择。播放速度受读取和解码耗时影响。
统计时长为 `帧数 / FPS`；多任务轨迹分别计入各 Task，跨 Task 汇总可能重复。

## 字段与数据 API

State / Action 别名位于 `config.py` 的 `FIELD_ALIASES`。字段不匹配或候选不唯一时指定：

```bash
.venv/bin/python -m data_process view --path /path/to/dataset --state-key qpos --action-key command
```

维度名来自元数据，缺失时使用 `dim_0` 等；多轴数值按原始顺序展平绘图，原始字段和形状保留。
State / Action 仅按维度索引对比，不推断单位或动作含义。目前不支持外部视频、LeRobot v3 和 HDF5。

```python
from data_process import BrowserModel, LeRobotReader

reader = LeRobotReader("playground/Dataset/piper_dataset")
frame = reader.read_frame(episode_index=0, frame_index=10)
raw = reader.read_episode(0, columns=["state", "actions"])

model = BrowserModel("playground/Dataset/piper_dataset")
episode = model.read_episode(0)
state, action = episode.state, episode.action  # (帧数, 实际维度数)
values = episode.frame(10)                    # 保留原始字段名和形状
images = model.read_images(0, 10)             # CHW RGB uint8
```

## 质量检查

在 **质量筛选** 中调整策略，选择当前轨迹、当前 Task 的原始范围或全部数据集，运行检查。
点击结果行查看指标和异常索引，点击查看按钮跳转到原始轨迹的首个异常帧。
完整检查且参数未变时可预览保留候选；可包含待复核轨迹。预览只包含本轮范围，空结果会清空浏览区，
使用 **恢复全部数据** 返回全集。

关闭面板后检查继续，可停止并下载部分报告；部分报告不能应用为完整预览。
同一服务同时执行一轮检查，新任务替换服务端上一轮报告；各页面独立保存浏览选择和预览范围。

| 检查 | 定义与限制 |
| --- | --- |
| 完整性与采样 | 检查 NaN/Inf、帧索引、严格递增时间戳、采样缺口和图像解码 |
| 静止比例 | position / command：所有选中维度在相邻帧均不变的比例；velocity：近零值帧比例；acceleration 无法推出静止 |
| 速度 / 加速度 | 按真实时间间隔差分，逐 Episode、逐维计算绝对 Z-score，按异常样本比例筛选 |
| Jerk | position 三阶、velocity 二阶、acceleration 一阶时间差分；阈值使用原始单位及秒，缺失时未验证 |
| 图像清晰度 | 固定宽度缩略图的灰度离散 Laplacian 方差；需校准下限 |
| 遮挡 / 冻结 | 近黑白低纹理、机器人变化时图像连续重复，仅为疑似指标，未运行语义遮挡模型 |
| 时间同步 | 比较同一时钟域的独立采集时间戳；缺失时未验证，多视角空间对齐仍需外参或共同视觉事件 |

静止、Z-score 和允许异常比例等均为可调整的初始值。差分时间位于前阶时间的中点，异常索引指向差分窗口最后一帧。
Z-score 以单条轨迹为参考，不能检测所有持续性尺度异常；command 的差分不自动代表物理速度或 Jerk。
容差和 Jerk 可逐原始维度指定，维度索引从 0 开始。图像步长大于 1 时仅为抽样覆盖，不能认证未检查帧。
公共 `timestamp` 不能重复映射为不同传感器的采集时钟，例如应提供真实存在的
`{"image": "image_timestamp", "wrist_image": "wrist_timestamp"}`。

筛选以整条 Episode 为单位，保留原始信号和图像，不平滑、插值、删帧、重对齐或导出清洗数据集。
下载的 JSON 报告包含源路径、策略快照、范围、完整性、逐条指标和分阶段统计，建议保存到源数据集之外。

```python
from dataclasses import replace
from data_process.config import QUALITY_POLICY
from data_process.processing import QualityInspector, save_quality_report

policy = replace(QUALITY_POLICY, image_stride=5)
report = QualityInspector("playground/Dataset/piper_dataset").inspect([0, 1], policy)
save_quality_report(report, "/tmp/piper_quality.json")
```

## 验证

```bash
.venv/bin/python -B -m unittest discover -s data_process/tests -v
```
