# LIBERO RTC 异步控制

实现依据为仓库内的 [RTC 论文](RTC.pdf)第 3 节、式 (2)–(5) 及 Algorithm 1。RTC 默认关闭，不修改训练、权重、训练数据、图像预处理、state 输入、夹爪编码或 LIBERO 环境配置。

## 原有等待帧的原因

原有调用链：

```text
eval_libero._run_episode
  → ModelClient.needs_inference(policy_step)
  → executor.submit(ModelClient.step, example, policy_step)
    → WebsocketClientPolicy.predict_action
      → WebsocketPolicyServer._route_message
        → PolicyServerWrapper.predict_action
          → framework.predict_action
```

`ModelClient.needs_inference()` 仅在没有缓存，或 `policy_step % action_chunk_size == 0` 时请求模型。因此下一次推理是在旧 chunk 消耗完后才发起。`_run_episode()` 在 Future 尚未完成时没有可消费的动作，执行六维机械臂零增量并保留上一夹爪命令。`policy_step` 只在消费预测动作时递增，仿真控制时间仍在递增，等待帧由此产生。RTC 关闭时继续走这条路径。

## 配置与启动

先按原流程启动模型服务，然后使用原来的脚本：

```bash
LIBERO_HOME=/path/to/LIBERO \
LIBERO_PYTHON=/path/to/python \
CKPT=/path/to/checkpoints/steps_100000_pytorch_model.pt \
HOST=127.0.0.1 PORT=6694 \
RTC_ENABLED=true RTC_STRIDE=4 RTC_DELAY=2 RTC_BETA=0.5 \
RTC_INPAINTING=true RTC_GUIDANCE_BETA=5.0 RTC_TIMEOUT=10 \
bash examples/LIBERO/eval_files/eval_libero.sh
```

| 含义 | Python 参数 | 脚本环境变量 | 默认值 |
| --- | --- | --- | --- |
| 开关 | `rtc_enabled` | `RTC_ENABLED` | `False` |
| H，预测 chunk 长度 | 服务端 metadata `action_chunk_size` | 无客户端覆盖 | 从权重配置获取 |
| s，最小请求间隔，单位控制周期 | `rtc_stride` | `RTC_STRIDE` | `4` |
| d，初始延迟估计，单位控制周期 | `rtc_delay` | `RTC_DELAY` | `2` |
| β，输出融合时旧动作的初始权重 | `rtc_beta` | `RTC_BETA` | `0.5` |
| 采样阶段前缀约束 | `rtc_inpainting` | `RTC_INPAINTING` | `True` |
| 论文式 (2) 的 guidance 上限 β | `rtc_guidance_beta` | `RTC_GUIDANCE_BETA` | `5.0` |
| 单次请求超时，秒 | `rtc_timeout` | `RTC_TIMEOUT` | `10.0` |

`rtc_beta` 与论文的 guidance clipping β 是两个不同的参数。前者控制额外的输出融合；后者控制采样时的前缀引导强度。仅评估论文式采样约束时可以设 `RTC_BETA=0`。

RTC 需要 `REALTIME=true`、`0 < s < H`、`0 ≤ d < H`、`0 ≤ rtc_beta ≤ 1`，guidance 上限和 timeout 必须为有限正数。常规连续控制还需要延迟小于旧 chunk 的剩余时间；论文有效工作区域为 `d ≤ s ≤ H-d`。不满足连续供给条件时不能保证无等待帧，程序执行下文的安全保持策略。

对 OFT 或未声明前缀采样能力的模型，使用：

```bash
RTC_ENABLED=true RTC_INPAINTING=false \
LIBERO_HOME=/path/to/LIBERO \
bash examples/LIBERO/eval_files/eval_libero.sh
```

这一模式提供异步 chunk 对齐和输出融合，不属于完整的 flow inpainting RTC。请求不支持的前缀约束时会明确报错，不会静默退化。

## 控制时间与实际时序

全部时间索引采用 episode 内的整数控制步。主线程在控制步 `t` 读取观测，在执行该步动作前处理已完成的推理。只有主线程可以调用 `env.step()`、读取 LIBERO 渲染缓存或生成视频帧。工作线程只运行纯模型请求。

RTC 路径为 `eval_libero._run_rtc_episode()` → `rtc_controller.RTCController.next_action(example, control_step)` → 工作线程中的 `ModelClient.predict_chunk()`。`RTCConfig` 验证配置，控制器管理本 episode 的 chunk、Future 和统计；原有 `ModelClient.step()` 缓存不参与 RTC。

需要区分：

* `t`：全局控制步，即将执行的动作时刻。
* `r`：推理提交时的全局控制步，也是正常后续 chunk 的时间起点。
* `a`：主线程首次发现推理已完成的控制步。
* `a-r`：该次请求实际经历的控制周期数，可能与估计 d 不同。
* `t-r`：新 chunk 在当前控制步对应的模型动作索引。

首个 chunk 尚未返回时安全保持。因为此时没有旧动作前缀，首个 chunk 在返回时建立执行起点，然后从动作 0 开始；这是静止启动的初始化步骤。所有后续请求严格以提交控制步为 chunk 起点。

例：`H=16, s=4`，首个 chunk 起点为 `t=3`。在 `t=7` 发起下次请求，旧 chunk 索引为 4。若请求在 `t=9` 返回，期间 `t=7,8` 继续执行旧动作索引 4、5；切换时新动作从索引 `9-7=2` 开始，并与旧动作索引 6 对齐。新动作索引 0、1 已经过期，永不补执行。下一次请求最早发生在上一请求起点经过 s 步之后，并且始终最多只有一个请求在途。

返回时若 `t-r ≥ H`，整个 chunk 都已过期，必须丢弃并用最新观测重试。延迟历史窗口用初始 d 初始化，每次返回加入实际控制周期延迟，最多保留 10 项；下一次请求采用窗口最大值。若估计超过剩余前缀容量，则对 mask 长度截断并警告。延迟估计仅用于指导尚未返回时预计执行的动作前缀，不参与动作对齐。若一次请求延迟已达到 s，收到结果后可立即发起下一次请求，此时实际请求间隔会大于等于配置的最小 s。

## 输出重叠融合

设新 chunk 返回时仍有 L 个与旧 chunk 时间对齐的未来动作，`j=0,…,L-1`。旧动作权重及融合为：

```text
w_old(j) = rtc_beta · (L-j)/L
a(j) = w_old(j) · a_old(j) + (1-w_old(j)) · a_new(j)
```

权重只用于重叠的未来区间，过期动作不参与融合，重叠之后直接使用新动作。

* 位置：对同一控制时刻的三维增量线性融合。
* 旋转：LIBERO OSC_POSE 接收每周期的旋转增量控制坐标，因此在该增量/切空间内线性融合；不是对绝对欧拉角做平均，也不是把控制器缩放过的命令当作绝对姿态做 SLERP。
* 夹爪：保持原来的 open/close 编码和二值化阈值。按较大权重选择旧或新命令，`w_old ≥ 0.5` 时选旧值；不产生夹爪中间命令。

输出融合只是连续性启发式，不能证明任意多模态动作平均仍然有效。真正的前缀条件生成在以下 PI Flow Matching 采样中完成。

## PI Action Prefix Inpainting

QwenPI 调用链继续进入：

```text
Qwen_PI.predict_action
  → LayerwiseFlowmatchingActionHead.predict_action
    → 每个 Flow Matching Euler 采样步骤中的 ΠGDM 引导
```

请求捕获旧 chunk 在请求时刻起的剩余动作作为 `prefix_actions`，维度为 `[B,K,D]`。K 为实际重叠长度，不硬编码 H。前缀传输使用环境动作坐标，由服务端 `PolicyNormProcessor` 按该权重原有训练变换归一化，随后才进入模型。模型输出仍通过原有反归一化路径转换回环境动作。前缀中的夹爪使用模型原始编码，不使用 LIBERO 最终 `-1/+1` 命令编码。

右侧补齐到 H 后，按论文式 (5) 构造每个时刻的权重，并广播到动作维：

```text
d' = min(estimated_delay, K)
W_i = 1                                      , i < d'
W_i = c_i · (exp(c_i)-1)/(e-1)                , d' ≤ i < K
W_i = 0                                      , i ≥ K
c_i = (K-i)/(K-d'+1)
```

其中 K 对应论文的 `H-s`。对每个采样时间 `τ`：

```text
A_hat = A_tau + (1-τ) · v(A_tau, observation, τ)
e = (prefix_target - A_hat) · W
g = VJP(A_hat, A_tau, e)
r² = (1-τ)² / (τ² + (1-τ)²)
λ = min(rtc_guidance_beta, (1-τ)/(τ · r²))
A_next = A_tau + Δτ · (v + λ · g)
```

`τ=0` 时将无穷上限取为 `rtc_guidance_beta`。VJP 在采样阶段通过自动微分计算，仅对当前动作轨迹求导，不更新模型参数。图像和 state 的处理沿用原路径。关闭前缀约束时仍使用原始 Euler 采样。

这实现了论文的训练外 ΠGDM 前缀约束，不是生成结束后覆盖动作。论文也指出有限采样步和受限 guidance 下，生成的前缀不保证与旧前缀逐元素完全相等；已承诺的动作仍由客户端旧 chunk 执行，不会拿新生成的过期前缀替换已经执行的动作。

当前能力范围：

| 模型 | 异步对齐/输出融合 | 采样阶段前缀约束 |
| --- | --- | --- |
| QwenPI + LayerwiseFM Action Head | 支持 | 支持论文式 (2)–(5) |
| QwenOFT 等直接回归模型 | 支持 | 不支持：没有迭代 flow 采样速度场及其动作 VJP |
| 其他 Flow Matching head/framework | 支持 | 尚未接入：需为其采样循环和条件特征启用动作 VJP，并显式声明能力 |

## 异常、终止与日志

首帧等待或旧 chunk 耗尽时，使用机械臂零增量并保持最后夹爪命令。绝不无限重复旧 chunk 的最后一个运动动作。推理错误和超时向上抛出，不能当作空结果忽略。

episode 的 Future、chunk、时间索引均单独创建，结束或提前成功后清理在途请求；工作线程不修改跨 episode 的动作缓存。清理完成后才允许下个 episode 使用连接，防止上个 episode 的结果进入下一轮。

超时同时覆盖主线程轮询期限、工作线程实际完成时间和 WebSocket 接收等待。即使结果在较晚的控制步已完成，也不能绕过请求总期限；按时完成但被较晚轮询的结果仍以真实控制索引判断是否过期。reset 清理失败后禁止复用该控制器，连接接收超时后也禁止复用。

日志包含请求耗时 `inference_ms`（毫秒）、实际控制周期延迟 `delay_steps`、等待帧 `wait_steps`、已消费的融合动作数 `blend_actions`、缓存耗尽次数 `buffer_exhaustions`、过期 chunk 数 `stale_chunks` 和请求数 `requests`，并保留原有 LIBERO 任务和总体成功率。控制循环落后实时周期时会警告，因为这时相同墙钟延迟对应的控制周期数会减少。

`inference_ms` 为客户端完整模型请求耗时；服务端同时记录 `server_inference_ms`，不含客户端预处理和网络往返。episode 结束后 drain 完成的请求单独记录耗时和终止前实际控制周期数，不将清理墙钟时间换算为仿真控制步，也不加入用于后续切换的 `delay_steps`。

## 验证范围

测试使用可控时钟、模拟模型/环境和 CPU 小型 flow 速度场验证：提前请求、在途继续消费、不同延迟的全局索引、融合公式、过期跳过、超时、reset/提前终止隔离、工作线程边界，以及 RTC 关闭后的既有行为。

```bash
python -m unittest discover -s tests -p 'test_*rtc*.py'
python -m unittest discover -s tests -p 'test_libero_realtime.py'
bash -n examples/LIBERO/eval_files/eval_libero.sh
```

本次实现验证：43 项测试通过，包括 15 项调度/episode 测试、11 项前缀采样测试、8 项协议/归一化测试和原有 9 项实时控制测试。Shell 语法检查、Python 编译检查、`git diff --check` 和真实 tyro CLI 参数解析通过。验证环境缺少的 `msgpack`/`tyro` 仅临时安装到 `/tmp/vla-rtc-deps`，未修改项目环境；评测依赖安装方式沿用 README。

单元测试不等同于实际 LIBERO 成功率或 GPU 性能测量。前缀 VJP 会增加推理显存和延迟，需要在目标 GPU、实际模型权重和 LIBERO 安装中测量并设置 s、d 和 timeout。若目标 GPU attention 内核不支持所需的反向传播，应选择支持动作 VJP 的 attention 后端，不能以输出拼接代替采样约束。
