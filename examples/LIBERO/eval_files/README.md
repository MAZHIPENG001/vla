# LIBERO 评测时序

`eval_libero.py` 默认启用实时评测（`realtime=True`）：

- 主线程按 `control_freq`（默认 20 Hz）调用 `env.step()`，所有渲染也留在主线程。
- 需要新动作块时，后台线程使用请求发起时的两路相机观测进行推理；同一时刻最多一个请求。
- 等待期间，每个控制周期发送零位移、零旋转，保持机械臂当前位置；夹爪沿用上一次开合命令。第一次推理返回前使用 `LIBERO_DUMMY_ACTION`，即零位移、零旋转、张开夹爪。物理仿真持续推进。
- 新动作块返回后，从第一个预测动作开始依次执行。等待步不推进模型动作索引，也不丢弃预测动作；已有缓存动作无需额外等待周期。动作块执行完后再请求下一块。
- 等待步与模型动作步共同占用任务原有的 `max_steps` 预算，并且均记录视频；视频 FPS 与控制频率一致，包含成功时的最后一帧。初始 `num_steps_wait` 稳定阶段仍不计入视频和任务预算。
- 视频左上角显示状态：等待推理时为红色“Waiting for inference”，执行新预测动作（包括动作缓存）时为绿色“Executing”。文字在 rollout 结束后叠加，不影响控制计时或送给模型的相机图像。
- episode 结束时等待未完成请求退出，再重置下一个 episode，避免旧请求修改新 episode 的缓存；这段清理等待不属于 rollout 视频。

例如，20 Hz 下推理耗时约 0.2 秒，会对应约 4 个保持机械臂位置的仿真步，实际步数会受请求完成时间与控制周期边界影响。300 步预算对应 15 秒仿真时间，其中包括推理等待。位移和旋转是增量命令，等待期间不重复发送上一条运动增量。

`eval_libero.sh` 显式传入实时开关和控制频率，支持环境变量 `REALTIME=true|false`（也接受 `1|0`）与 `CONTROL_FREQ`（正整数，单位 Hz）。沿用已配置的 `LIBERO_HOME`、`LIBERO_PYTHON` 等变量即可：

```bash
# 默认实时评测，20 Hz。
bash examples/LIBERO/eval_files/eval_libero.sh

# 显式设置实时评测和控制频率。
REALTIME=true CONTROL_FREQ=20 bash examples/LIBERO/eval_files/eval_libero.sh

# 同步评测，推理期间暂停仿真。
REALTIME=false bash examples/LIBERO/eval_files/eval_libero.sh
```

其他 shell 入口沿用 Python 默认行为。直接运行 Python 时可用：

```bash
# 修改控制频率，同时修改环境控制频率和视频 FPS。
python examples/LIBERO/eval_files/eval_libero.py --args.control-freq 20

# 同步对照：推理时暂停仿真；视频仍按实际仿真控制频率保存。
python examples/LIBERO/eval_files/eval_libero.py --args.no-realtime
```

日志输出 rollout 的仿真时长、实际耗时、模型动作数、推理等待步数。若仿真和渲染本身无法达到目标控制频率，会输出警告；这种情况下无法保证仿真时间与真实时间一致，视频中体现的推理延迟可能偏小。

状态文字使用 Pillow 自带字体，无需安装或配置系统字体。

无需仿真依赖的回归检查：

```bash
.venv/bin/python -m unittest discover -s tests -v
```
