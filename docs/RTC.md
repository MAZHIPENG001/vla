# LIBERO RTC：按论文实现

实现依据为本仓库 [RTC.pdf](RTC.pdf) 第 3 节的式 (2)–(5)、Algorithm 1 和附录 A.5。RTC 的连续性由 Flow Matching 采样中的前缀条件生成实现。新结果返回后按时间索引替换当前 chunk，不再对两个模型输出做额外线性插值。

此前版本将输出融合权重命名为 `RTC_BETA`，另加 `RTC_GUIDANCE_BETA`，并根据预测延迟设置请求门槛；这些是论文之外的设计，现已移除。现在只有一个 β：论文式 (2) 的 guidance weight clipping，默认值为附录表 4 中的 5。

## 参数与论文符号

| 论文符号 | 本项目来源 | 含义 |
| --- | --- | --- |
| H | 服务端 metadata 的 `action_chunk_size` | 预测动作长度，来自模型配置，不在客户端硬编码 |
| s_min | `rtc_stride` / `RTC_STRIDE` | Algorithm 1 第 13 行的最小执行长度；`auto` 采用 1，也可显式设置正整数 |
| s | 提交请求时实际经过的控制步 | Algorithm 1 第 14 行，等于请求控制步减当前 chunk 起点；不是额外的调参项 |
| d_init | `rtc_delay` / `RTC_DELAY` | `auto` 从首次普通推理测量；数字只作为初始延迟种子 |
| d | 最近 b 次实际延迟的最大值 | Algorithm 1 第 17 行，每次提交新请求时计算，用于前缀 mask |
| β | `rtc_beta` / `RTC_BETA`，默认 5 | 式 (2) 的引导上限，不是输出插值比例，也不限制在 [0,1] |
| b | 控制器内的 10 项延迟窗口 | 按附录表 4 的真实机器人设置 |
| n | 模型配置 `num_inference_timesteps` | Flow Matching Euler 采样步数，复用当前权重对应的动作头配置 |

表 4 的 π0.5 实验使用 n=5、H=50、s_min=25。它们是该实验的设置，不是 Algorithm 1 对所有模型的固定常数。本项目复用 QwenPI 的 H 和 n；`RTC_STRIDE=auto` 取 s_min=1，以保留无需人工指定请求间隔的入口。s_min 是响应速度/计算负载的选择，不能从推理耗时唯一反推出任务最优值；实际 s 则按论文随串行推理的真实延迟自然变化。

除上述算法设置外，`RTC_ENABLED` 是部署开关，`INFERENCE_TIMEOUT` 是通用请求超时保护，不是 RTC 论文的超参数。Python 参数默认关闭 RTC；现有 `eval_libero.sh` 默认启用 RTC。它们均不改变训练、权重、数据、图像处理、state 格式、夹爪编码和 LIBERO 环境配置。

## 启动与旧配置迁移

按原流程启动更新后的模型服务，再使用原脚本。stride、delay 和 β 均可省略：

```bash
LIBERO_HOME=/path/to/LIBERO \
LIBERO_PYTHON=/path/to/python \
CKPT=/path/to/checkpoints/steps_100000_pytorch_model.pt \
HOST=127.0.0.1 PORT=6694 \
RTC_ENABLED=true \
bash examples/LIBERO/eval_files/eval_libero.sh
```

完整的算法配置入口只有：

```bash
RTC_ENABLED=true RTC_STRIDE=auto RTC_DELAY=auto RTC_BETA=5 \
bash examples/LIBERO/eval_files/eval_libero.sh
```

手动设置 `RTC_STRIDE=4` 表示 s_min=4；`RTC_DELAY=2` 表示 d_init=2。两项可各自独立采用 `auto` 或数字。始终只有一个在途推理，实际动作对齐不使用 d_init 或预测 d 替代真实控制步。

旧配置需要显式迁移，脚本发现旧变量会报错，不会静默忽略：

* 删除 `RTC_GUIDANCE_BETA`，用 `RTC_BETA` 设置论文中的 β。
* 删除 `RTC_INPAINTING`。开启 RTC 就执行前缀引导；不支持该能力的模型使用 `RTC_ENABLED=false`。
* 将 `RTC_TIMEOUT` 改为 `INFERENCE_TIMEOUT`，对应 Python 的 `inference_timeout`。默认 10 秒，仅用于请求与清理的时间上限。
* 旧 `RTC_BETA=0.5` 的“输出融合比例”含义已取消。若继续传 0.5，它表示论文的 guidance clipping β=0.5；默认设置应省略该变量或使用 5。

RTC 要求 `REALTIME=true`。手动 s_min 满足 `0 < s_min < H`，手动 d_init 满足 `0 ≤ d_init < H`；β 与请求超时必须为有限正数。

## Algorithm 1 的控制时序

原有等待帧来自 `_run_episode()` → `ModelClient.needs_inference(policy_step)` → `ModelClient.step()`：旧缓存耗尽后才请求下一批动作，Future 未完成时执行零增量保持。RTC 关闭后仍使用这条原有路径。

RTC 使用 `_run_rtc_episode()` → `RTCController.next_action()` → 后台 `ModelClient.predict_chunk()`。主线程负责 `env.step()`、观测快照和渲染，工作线程只执行模型请求。控制器是 Algorithm 1 的单在途请求实现：

1. 首批无前缀推理建立论文所需的初始 chunk；等待时安全保持。初始 chunk 在返回时建立起点，并由这次请求测量自动 d_init。
2. 当前控制步相对当前 chunk 起点的偏移达到 s_min 且没有在途请求时，提交下一次推理。将这个实际偏移记为 s，捕获 `A_prev = A_cur[s:H]` 以及该时刻观测。
3. 用最近 b 次实际控制周期延迟的最大值预测 d。新 chunk 在后台按式 (2)–(5) 生成，前台继续执行当前 chunk。
4. 请求在全局控制步 r 提交，主线程在控制步 a 发现结果可用，则实际延迟为 `a-r`。替换为新 chunk 后，当前索引为 `a-r`，对应 Algorithm 1 第 20–22 行；新 chunk 的起点为 r。
5. 替换后如相对新 chunk 的偏移已达到 s_min，可立即提交下一次请求。因此正常连续运行时，实际 s 自然受 `max(s_min, 前次实际推理延迟)` 约束。

例如 H=16、s_min=1，初始 chunk 在控制步 3 开始执行。在控制步 4 提交新推理，此时 s=1，剩余前缀长度 K=15。如果结果在控制步 7 可用，控制步 4、5、6 仍执行旧动作；控制步 7 直接执行新 chunk 的索引 3。随后可以在控制步 7 再次请求，新的实际 s=3。不会补执行已经过期的新动作 0、1、2。

延迟预测只用于生成约束，不再将其代入额外的 stride 公式，也不根据预测延迟切换成另一种调度策略。实际延迟下降时，实际请求间隔可随前次推理变短；保守的 d 预测则需等待较慢样本移出窗口后下降。Reset 会清空窗口，数字 d_init 则重新作为初始种子加载。

## 式 (2)–(5) 的前缀引导

模型调用链为：

```text
ModelClient.predict_chunk
  → WebsocketClientPolicy.predict_action
    → WebsocketPolicyServer._route_message
      → PolicyServerWrapper.predict_action
        → Qwen_PI.predict_action
          → LayerwiseFlowmatchingActionHead.predict_action
            → 每个 Euler 步的 ΠGDM 引导
```

客户端传输请求时刻剩余的旧模型动作原值 `[B,K,D]`。服务端使用该权重训练时的 `PolicyNormProcessor` 将动作前缀归一化，输出沿用原反归一化路径。前缀不额外二值化夹爪、不混合位置或旋转；LIBERO 最终夹爪编码仍由原来的环境适配代码处理。

在论文有效区域中，K=H-s、d≤K。式 (5) 为：

```text
W_i = 1                                   , i < d
W_i = c_i · (exp(c_i)-1)/(e-1)             , d ≤ i < H-s
W_i = 0                                   , i ≥ H-s
c_i = (H-s-i)/(H-s-d+1)
```

W 在动作维上广播；旧动作右补齐到 H，作为 Y。每个采样时刻 τ 执行：

```text
f(A_tau) = A_tau + (1-τ) · v(A_tau, observation, τ)
e = (Y - f(A_tau)) · W
g = J_f(A_tau)^T · e
r² = (1-τ)² / (τ² + (1-τ)²)
λ = min(β, (1-τ)/(τ · r²))
A_next = A_tau + (1/n) · (v + λ · g)
```

τ=0 时以 β 作为截断后的引导系数。通过完整 endpoint VJP 对当前动作求导，包括速度场对动作的导数；不更新模型参数，不对 VLM 回传梯度。前缀误差可以影响未掩码的后续动作，保证约束发生在生成过程中。

论文第 3.2 节说明有限步 ΠGDM 并不保证生成前缀逐元素严格相等。实际已经承诺的动作仍由旧 chunk 执行；返回后直接替换生成结果，不再增加输出平均、插值、SLERP 或生成后前缀覆盖。

## 模型支持与有效范围之外的保护

QwenPI + LayerwiseFM 已接入所需速度场和动作 VJP。OFT 等直接回归模型没有该采样过程，其他尚未接入的 Flow Matching 框架也不声明支持。它们开启 RTC 会报错，应使用 `RTC_ENABLED=false` 保持原有测评。不存在“关闭 inpainting 仍作为 RTC”的模式。

论文连续供给的工作区域为 `d ≤ s ≤ H-d`。突发延迟或过短 H 可能使真实运行超出该区域；不能通过压低延迟估计、重放动作或输出插值宣称仍满足论文保证。部署层保留以下保护：

* 首批等待或缓存耗尽时执行机械臂零增量，并保持最后夹爪命令。
* 返回时若 `a-r ≥ H`，丢弃整批过期动作，使用最新观测重试。
* 若预测 d 超过剩余前缀长度，告警并只约束可用的旧动作；保留真实估计与真实对齐索引。这属于越界运行的保护，不是论文有效区域内的新算法。
* 请求失败、超时和 Episode 清理失败向外抛出。结束、提前成功和 reset 都清理在途状态；旧 Episode 的结果不能进入下一轮。

## 日志与测试

日志记录请求耗时 `inference_ms`、实际控制周期延迟 `delay_steps`、预测延迟 `estimated_delay_steps`、实际执行长度 `execution_horizons`、等待帧、缓存耗尽和过期 chunk，并保留 LIBERO 成功率。`execution_horizons` 对应每次提交时的实际 s；初始无前缀请求记为 0，不算论文的正常重规划阶段。请求日志还记录 s_min 和 H。

客户端模型请求耗时包含预处理和网络，服务端另记录 `server_inference_ms`。终止后 drain 的墙钟时间不换算成仿真控制步，不污染延迟预测窗口。仿真跟不上实时控制频率时会告警。

```bash
python -m unittest discover -s tests -p 'test_*rtc*.py'
python -m unittest discover -s tests -p 'test_libero_realtime.py'
bash -n examples/LIBERO/eval_files/eval_libero.sh
```

本次完整验证 65 项测试全部通过，无跳过；Python 编译、Shell 语法和 `git diff --check` 均通过。测试覆盖 Algorithm 1 的实测 s/d、时间索引、单在途推理、旧动作继续执行、直接替换、完整 VJP 与 mask、归一化、超时、reset、提前成功、CLI 和 RTC 关闭后的回归行为。单元测试不能代替目标 GPU、实际权重和 LIBERO 成功率测量；显存和推理耗时仍需在目标部署环境验证。
