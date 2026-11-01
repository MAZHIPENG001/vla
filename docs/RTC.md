# LIBERO RTC 异步控制

实现依据为仓库内的 [RTC 论文](RTC.pdf)第 3 节、式 (2)–(5) 及 Algorithm 1。Python 参数默认关闭 RTC；现有 `eval_libero.sh` 默认启用 RTC。开启后 stride 和 delay 默认根据推理的实际控制周期延迟自动推导，无需手动指定。不修改训练、权重、训练数据、图像预处理、state 输入、夹爪编码或 LIBERO 环境配置。

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
RTC_ENABLED=true RTC_BETA=0.5 \
RTC_INPAINTING=true RTC_GUIDANCE_BETA=5.0 RTC_TIMEOUT=10 \
bash examples/LIBERO/eval_files/eval_libero.sh
```

| 含义 | Python 参数 | 脚本环境变量 | 默认值 |
| --- | --- | --- | --- |
| 开关 | `rtc_enabled` | `RTC_ENABLED` | Python `False`；脚本 `true` |
| H，预测 chunk 长度 | 服务端 metadata `action_chunk_size` | 无客户端覆盖 | 从权重配置获取 |
| s，最小请求间隔，单位控制周期 | `rtc_stride` | `RTC_STRIDE` | `auto`，或手动正整数 |
| d，延迟估计，单位控制周期 | `rtc_delay` | `RTC_DELAY` | `auto`，或手动初始非负整数 |
| β，输出融合时旧动作的初始权重 | `rtc_beta` | `RTC_BETA` | `0.5` |
| 采样阶段前缀约束 | `rtc_inpainting` | `RTC_INPAINTING` | `True` |
| 论文式 (2) 的 guidance 上限 β | `rtc_guidance_beta` | `RTC_GUIDANCE_BETA` | `5.0` |
| 单次请求超时，秒 | `rtc_timeout` | `RTC_TIMEOUT` | `10.0` |

`rtc_beta` 与论文的 guidance clipping β 是两个不同的参数。前者控制额外的输出融合；后者控制采样时的前缀引导强度。仅评估论文式采样约束时可以设 `RTC_BETA=0`。

RTC 需要 `REALTIME=true`、`0 ≤ rtc_beta ≤ 1`，guidance 上限和 timeout 必须为有限正数。手动指定的 s 必须满足 `0 < s < H`，手动初始 d 必须满足 `0 ≤ d < H`；自动估计如实保留实际延迟，即使测得 d 已超过 H，也不会把它伪装成较小延迟。

### 自动推导 stride 和 delay

`RTC_STRIDE`、`RTC_DELAY` 默认为 `auto`，也可以显式设置：

```bash
RTC_STRIDE=auto RTC_DELAY=auto bash examples/LIBERO/eval_files/eval_libero.sh
```

自动模式的冷启动先提交一个没有动作前缀的普通请求，等待期间安全保持。第一批结果在主线程被发现可用时，测量 `d_k = return_control_step - request_control_step`。此后始终保存最近 10 次观测到的控制周期延迟，用其最大值作为下一次的 `d_hat`，同时用于前缀 inpainting 的 mask 与自动请求间隔。冷启动前不需要猜测推理耗时；实际延迟覆盖客户端预处理、网络、服务端推理和主线程发现结果的时间，不使用服务端毫秒数替代真实控制步。

自动 stride 采用优先快速响应的策略：

```text
candidate = max(1, d_hat)
s = candidate  , candidate + d_hat ≤ H
s = 1          , candidate + d_hat > H
```

常规情况下该选择满足论文工作区域 `d_hat ≤ s ≤ H-d_hat`（零周期延迟时最少隔一个控制周期）。例如 H=16、最近实测最大延迟为 3 时，自动使用 s=3。延迟升高会立即调整，延迟下降则在较慢样本移出 10 项窗口后调整。

如果 `candidate + d_hat > H`，单个在途请求的串行推理无法保证连续供应动作。控制器会警告、将请求门槛降为 1，尽早利用最新观测补充动作；不会重复提交在途请求，也不宣称能够消除由此产生的等待帧。缓存耗尽时仍执行安全保持。每次真正提交前还会检查剩余前缀容量，动作索引始终以实际控制步对齐。

保留手动与混合模式：

```bash
RTC_STRIDE=4 RTC_DELAY=2 bash examples/LIBERO/eval_files/eval_libero.sh
RTC_STRIDE=4 RTC_DELAY=auto bash examples/LIBERO/eval_files/eval_libero.sh
RTC_STRIDE=auto RTC_DELAY=2 bash examples/LIBERO/eval_files/eval_libero.sh
```

手动 stride 保持指定的最小请求间隔。手动 delay 沿用原行为：只作为 10 项历史窗口的初始种子，后续继续根据实测延迟更新，不固定实际动作对齐偏移。Episode reset 会清空历史；自动 delay 重新等待首批测量，手动 delay 重新加载该种子。

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

例：服务端 `H=16`、手动指定 `s=4`，首个 chunk 起点为 `t=3`。在 `t=7` 发起下次请求，旧 chunk 索引为 4。若请求在 `t=9` 返回，期间 `t=7,8` 继续执行旧动作索引 4、5；切换时新动作从索引 `9-7=2` 开始，并与旧动作索引 6 对齐。新动作索引 0、1 已经过期，永不补执行。下一次请求最早发生在上一请求起点经过当前 s 步之后，并且始终最多只有一个请求在途；自动模式会随延迟窗口更新 s。

返回时若 `t-r ≥ H`，整个 chunk 都已过期，必须丢弃并用最新观测重试。若延迟估计超过剩余前缀容量，则对 mask 长度截断并警告。延迟估计用于前缀约束与请求调度，不参与动作对齐。若一次请求延迟已达到当前 s，收到结果后可立即发起下一次请求，此时实际请求间隔受真实推理耗时限制，可能大于 s。

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

每次请求还记录生效的延迟估计 `estimated_delay_steps` 和请求门槛 `request_strides`，并记录配置模式、当前估计、stride 和 H。`request_strides` 是计划门槛，不代表在途推理一定能在该间隔内完成；实际延迟仍以 `delay_steps` 为准。

`inference_ms` 为客户端完整模型请求耗时；服务端同时记录 `server_inference_ms`，不含客户端预处理和网络往返。episode 结束后 drain 完成的请求单独记录耗时和终止前实际控制周期数，不将清理墙钟时间换算为仿真控制步，也不加入用于后续切换的 `delay_steps`。

## 验证范围

测试使用可控时钟、模拟模型/环境和 CPU 小型 flow 速度场验证：提前请求、在途继续消费、不同延迟的全局索引、融合公式、过期跳过、超时、reset/提前终止隔离、工作线程边界，以及 RTC 关闭后的既有行为。

```bash
python -m unittest discover -s tests -p 'test_*rtc*.py'
python -m unittest discover -s tests -p 'test_libero_realtime.py'
bash -n examples/LIBERO/eval_files/eval_libero.sh
```

本次验证 58 项测试全部通过且无跳过（原有 43 项、12 项自动调度、3 项 CLI），Python 编译、Shell 语法和 `git diff --check` 通过。自动模式覆盖冷启动测量、零周期延迟、延迟变化与窗口淘汰、超出 horizon、reset 历史清理、手动/混合模式和真实 tyro 参数解析。Shell 参数检查拒绝无效的 auto/数值输入。验证环境缺少的 `msgpack`/`tyro` 仅临时安装到 `/tmp/vla-rtc-deps`，未修改项目环境；评测依赖安装方式沿用 README。

单元测试不等同于实际 LIBERO 成功率或 GPU 性能测量。前缀 VJP 会增加推理显存和延迟，自动模式会随这些变化调整；仍需在目标 GPU、实际模型权重和 LIBERO 安装中测量成功率与耗时并选择 timeout。自动 stride 是优先响应速度的调度策略，不能仅凭耗时推导任务成功率最优的执行频率。若目标 GPU attention 内核不支持所需的反向传播，应选择支持动作 VJP 的 attention 后端，不能以输出拼接代替采样约束。
