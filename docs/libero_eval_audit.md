# LIBERO 评测核对记录

核对日期：2026-10-30；2026-10-31 补查 state 默认值与时间索引。以环境源码和训练输入为依据，StarVLA 评测代码仅作对照。本轮按用户要求停止本地权重加载，不包含真实策略 rollout 或成功率复测。

核对对象：

- 本地 LIBERO：`/home/ma/GithubDoc/LIBERO`，commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`。
- 本地上游 StarVLA：`/home/ma/GithubDoc/starVLA`，commit `3422b9f2387b6f682cf02802904a77b23ab13afd`。
- 评测结构参考：`/home/ma/GithubDoc/openpi/examples/libero/main.py`，采用 episode 内部的动作队列，队列为空时重新推理。
- 发布模型：`playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt`，及其运行目录内的 `config.yaml`、`dataset_statistics.json`。
- 本地训练数据：`playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot`。它的夹爪编码与发布 checkpoint 不同。

| 核对项 | 证据与结论 | 处理 |
| --- | --- | --- |
| 环境夹爪方向 | LIBERO 默认 Panda/default gripper，包装器不改动作符号；robosuite 1.4.0 的 PandaGripper 定义 -1 张开、+1 闭合 | 显式区分训练编码，不能对所有模型统一翻转 |
| 发布 checkpoint 夹爪 | 保存的 min/max 为 0/1，mask 最后一维为 false；OpenVLA LIBERO 标准化将原始闭合命令转换成 1=open、0=close。仅统计范围不能证明开合方向 | 发布模型保留 binary_open：0→+1 闭合，1→-1 张开 |
| 本地数据夹爪 | parquet 取值为 -1/+1。抽查前 8 条演示，在指间距 0.01～0.07 的活动区间内，-1 后平均指间距增加约 0.00579（45 个样本），+1 后减少约 0.00239（192 个样本） | 这类原始数据训练的模型选择 signed_close；旧映射会反转两端 |
| 两路相机 | DataConfig.video_keys 和 modality.json 都是 primary→wrist | 保留 agentview_image→robot0_eye_in_hand_image |
| 180° 旋转 | OpenVLA 再生成脚本说明 HDF5→RLDS 阶段旋转 180°；评测应匹配数据转换。旋转不是 LIBERO 环境本身的统一要求 | 保留两轴反转，不额外转换 RGB/BGR 或 crop |
| 256→224 | 训练 _pack_sample 使用 RGB PIL resize((224,224))，默认 BICUBIC；原客户端为 BILINEAR，上游也有同样差异 | 改为 BICUBIC，测试与训练调用逐像素一致 |
| state 默认值 | 用户明确 state 默认输入模型；此前训练、服务端、客户端把缺失 include_state 当作 false，会静默删掉状态 | 默认改为 true；训练和服务端共用解析函数；只有显式 false 才关闭；旧 server 未提供元数据时客户端也默认发送 |
| state 时间索引 | LIBERO DataConfig 取 range(-16,0)，OFT 文本化只读取 state[0]，因此帧 t 使用 max(t-16,0) 的状态；评测却输入 t 的状态 | 改为 [0]，训练和评测均使用当前帧、8 维、float16 状态；不能借此修复旧权重已经学到的时间关系 |
| BF16 输出 | 启动脚本默认 --use_bf16，OFT 动作头也是 BF16，原 .cpu().numpy() 会抛出 Unsupported ScalarType BFloat16 | 转为 float32 再返回 NumPy；这是推理异常修复，不是成功率提升的证据 |
| Action Chunk | future window=7，包含当前动作共 8 步。参考 openpi，每个 episode 新建 deque，队列为空才推理 | 取新预测的前 replan_interval 个动作，逐个 popleft；移除全局 step 和预测索引，episode 结束时直接丢弃队列；短预测执行完后重推理 |
| 反归一化 | 当前与上游 LIBERO DataConfig 都只对前 6 维使用 min/max，夹爪直通，不是 q01/q99 | 保留训练变换逆路径，增加统计维数校验 |
| 成功判定 | 本地 bddl_base_domain.py 的 step 将 done 覆盖为 _check_success() | 现有 done 计数正确 |
| 控制器单位 | 环境使用 OSC_POSE，robosuite 自行把 [-1,1] 输入缩放到位置/旋转增量 | 反归一化后直接送入环境，不再额外乘位置/旋转系数 |
| 等待协议 | StarVLA/OpenVLA 等待 10 步并张开夹爪；LIBERO 自带 lifelong 评测等待 5 步、零夹爪动作 | 发布模型保留前者，不能混用不同策略的评测协议 |
| 无用代码清理 | 删除未使用的 ensemble、sticky gripper、图像历史、可视化、DDIM 参数、raw_action 拆分再拼接、计时与调试代码 | 删除 pretrained_path、post_process_action、job_name 和客户端 include_state 覆盖；实际模型和 state 开关由 server 元数据决定，日志显示真实 checkpoint |

发布配置未保存完整训练代码版本、显式相机和 normalization mode。当前 DataConfig、模型说明及数值验证支持上述设置，但紧凑配置不能证明历史训练管线的全部细节。不能用 `results/debug/dataset_statistics.json` 替换发布模型统计：其数据来源及夹爪范围不同。

验证结果：

- `MPLCONFIGDIR=/tmp/vla-mpl NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python -m unittest discover -s tests -v`：14 项通过。测试隔离 websocket 传输层与仿真依赖，覆盖 rollout 的 1/2/4/8 重规划、每次使用最新观测、episode/任务切换、短预测、非法响应、相机像素、默认 state 在训练打包与服务端保留、旧 server 兼容、显式关闭 state、BF16 动作头和资源关闭。仿真环境和 VLM 使用轻量替身，不能视为真实任务成功率。
- 用实际发布 checkpoint 统计验证随机 (8,7) 预测：与六维 min/max、夹爪直通的独立计算最大误差 0.0；再经训练正向变换恢复预测，最大误差 1.1920929e-07。
- 三个 shell 入口（server、单次 eval、批量 eval）通过 bash -n。state CLI 与三个 LIBERO YAML 配置检查通过，CLI 检查隔离了 websocket 传输依赖。尚未运行仿真探针或成功率复测：当前 .venv 未安装 LIBERO/robosuite，源码存在不等于运行依赖就绪。

在已有 LIBERO 运行环境执行独立探针，不需要 policy server：

```bash
export LIBERO_HOME="$HOME/GithubDoc/LIBERO"
export PYTHONPATH="$LIBERO_HOME:$PWD:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# 沿用已配置的 LIBERO_CONFIG_PATH；LIBERO_PYTHON 指向已安装仿真依赖的 Python。
"$LIBERO_PYTHON" -m examples.LIBERO.eval_files.check_libero_contract
```

探针从同一初态分别执行 -1/+1，输出实际 gripper 类型、robosuite 版本、执行前后 qpos、指间距，并检查 -1 的最终指间距大于 +1。若碰撞或安装差异导致不符合预期，应查看测量值。

服务端修复后需重启 policy server。发布模型先保留 8 步基线，再固定 suite、seed、initial states 对比：

```bash
for interval in 8 4 2 1; do
  REPLAN_INTERVAL="$interval" GRIPPER_ENCODING=binary_open \
    bash examples/LIBERO/eval_files/eval_libero.sh
done
```

两个 shell 入口均支持 REPLAN_INTERVAL 和 GRIPPER_ENCODING，视频/批量日志按两者分目录或命名，避免覆盖。直接调用 Python 时使用 `--args.replan-interval 2 --args.gripper-encoding binary_open`；不传 interval 时执行完整预测块。

| GRIPPER_ENCODING | 模型输出语义 | 环境命令 |
| --- | --- | --- |
| binary_open（发布模型默认） | 0 闭合，1 张开 | 0→+1，1→-1，阈值 0.5 |
| binary_close | 0 张开，1 闭合 | 0→-1，1→+1，阈值 0.5 |
| signed_close（本地原始数据） | -1 张开，+1 闭合 | 保持符号，零保持夹爪 |

state 输入为 EEF xyz、axis-angle 旋转向量和两个手指 qpos，共 8 维；modality.json 的 pad/gripper 名称实际对应两个手指位置，不应替换成动作的二值开合值。LIBERO 的三个训练配置均显式 include_state: true，state_dim 修正为 8。客户端在文本化前使用训练打包相同的 float16 精度。

历史紧凑配置未保存 include_state，单凭这一点以及权重名称/形状不能确定它的训练提示词。此前将这类配置判定为“不用 state”的结论已撤回。当前默认遵循用户明确的 state 输入约定；这不证明旧 checkpoint 已与新默认设置对齐。修改 DataConfig 的时间索引只影响后续数据加载，不能改变已训练模型。

本仓库 LIBERO transform 未归一化 state；评测也保留同样的原始量，未擅自增加 min/max 归一化。OFT 的分桶函数只覆盖 [-1,1]，实际约 3 rad 的旋转向量可能落入最高桶，属于需要后续统一训练/推理表示的风险；不能只在评测时归一化后就声称适配了旧模型。

重启 server 后，日志中的 include_state 应为 True。普通 server 启动脚本和批量入口均接受 INCLUDE_STATE=auto|true|false；auto 使用 checkpoint 的显式设置，缺省为 true。直接运行 server_policy.py 时使用 --include-state 或 --no-include-state。显式开关只用于已知训练约定或固定条件的对照，不再根据“配置缺失”推断关闭。

整理后的职责：eval_libero.py 管理 episode、动作队列、统计和视频；ModelClient.infer(example) 只返回未经夹爪映射的 (T,7) 反归一化动作块；libero_contract.py 管理相机、夹爪和可选 state。旧 ModelClient.step/raw_action 接口已移除。发生评测异常时向上抛出，并在 finally 中关闭环境和连接；批量入口也会清理自己启动的 server。

外部依据：

- [robosuite 1.4.0 PandaGripper](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/models/grippers/panda_gripper.py)：环境开合方向。
- [OSC_POSE 配置](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/config/osc_pose.json)：输入范围与缩放。
- [OpenVLA 数据变换](https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/oxe/transforms.py)：libero_dataset_transform 的夹爪翻转。
- [OpenVLA 数据再生成说明](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py)：180° 旋转发生在 HDF5→RLDS。
- [StarVLA 客户端对照](https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/simBenchmarks/LIBERO/eval_files/model2libero_interface.py)：bilinear 和整块缓存不能替代训练/环境契约核对。

