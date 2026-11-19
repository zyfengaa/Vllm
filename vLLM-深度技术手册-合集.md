# vLLM 深度技术手册（2023–2026）

> **合集版**：由分章文件自动拼接，按 `00-总览.md` → `08-附录.md` 顺序；建议以分章文件为准。

---

# 第 1 章 发展史：五个时代与 102 个版本

> [← 总览](00-总览.md) ｜ 下一章：[02-宏观架构](02-宏观架构.md)

### 1.1 一页纸时间线

| 时间 | 版本区间 | 时代主题 | 一句话概括 | 里程碑 |
|---|---|---|---|---|
| 2023.06–2023.12 | 0.0.1 → 0.2.6 | **起源：分页显存管理** | 用操作系统虚拟内存的思路管理 KV Cache | PagedAttention 论文、continuous batching、OpenAI 兼容 API、AWQ/GPTQ |
| 2024.01–2024.06 | 0.3.0 → 0.4.3 | **生产化** | 从"跑得快"到"跑得稳、跑得省" | 自动前缀缓存、Chunked Prefill、投机解码、多模态、自定义 AllReduce |
| 2024.07–2024.12 | 0.5.0 → 0.6.6 | **吞吐战争** | 向 CPU 开销、调度串行、kernel 质量要性能 | 吞吐 2×、多步调度、异步输出处理、FlashInfer FP8 KV |
| 2025.01–2025.05 | 0.7.0 → 0.9.x | **V1 重写** | 为"规模化 + 编译"重新设计引擎 | V1 Alpha、torch.compile 默认、V1 默认、PyTorch 2.7/CUDA 12.8、Blackwell、DP/EP/PD |
| 2025.07–2025.12 | 0.10 → 0.13 | **V0 退役 + 异步化** | 删掉旧引擎，调度与执行解耦 | 异步调度（实验→默认）、FULL_AND_PIECEWISE CUDA Graph、gRPC、`--max-model-len auto` |
| 2026.01–2026.05 | 0.14 → 0.21 | **架构换代** | Transformer v5、CUDA 13、DeepSeek V4、低精度 KV | FA4 作为 MLA prefill 默认、TurboQuant 2-bit KV、在线量化、Rust 前端 |
| 2026.06–2026.11 | 0.22 → 0.34 | **Runner V2 与 KV 基础设施** | 执行器重写、KV 分层、去 PagedAttention | MRv2 默认、PagedAttention 移除、多级 KV Offload、HiSparse、block_size=64、可扩展 KV |

> 完整的 102 个版本日期表见 [附录 B](08-附录.md#b-完整版本时间线)。

### 1.2 时代一（2023.06–2023.12）：PagedAttention 的诞生

#### 1.2.1 当年的真问题

2023 年上半年，LLM 推理的主流做法是**静态批处理**：把 N 条请求对齐后一起前向。两个致命浪费：

1. **显存碎片**：每条请求的 KV Cache 必须**预先按 `max_model_len` 连续分配**。实际生成长度往往只有几百 token，却要占满 4K/8K 的连续显存；不同请求之间长度不一致，造成外部碎片。
2. **批内空转**：一条请求生成结束后，整个 batch 要等最慢的一条（`EOS` 不会提前释放槽位）。

论文给出的实测数字（LLaMA-13B / A100）：现有系统**只有 20%–38% 的 KV Cache 显存真正存放有效 token**，其余是预留与碎片。这直接决定了**并发上限**，而并发上限就是吞吐上限。

#### 1.2.2 PagedAttention 的三个关键设计

**（1）把 KV Cache 切成固定大小的 block**

```
逻辑序列:  [t0 t1 ... t15][t16 ... t31][t32 ... t47]
物理 block:   #7            #2            #19        ← 不连续
block_table:  [7, 2, 19]                             ← 每条序列一张表
```

- 物理 block 大小 = `block_size` 个 token（0.1–0.33 默认 16，**0.34 起默认 64**）。
- 一条序列的 KV 分散在任意物理块中，靠 **block_table**（逻辑块号 → 物理块号）寻址。
- 内部碎片最多 `block_size-1` 个 token 的浪费，把"按最大长度的外部碎片"变成"按块粒度的内部碎片"，**利用率从 20–38% 提升到 90%+**。

**（2）按需分配 + 按块追加**

生成第 t 个 token 时，只有跨过块边界才需要新块。这使得"显存按 token 增长"成为可能，并发数不再由 `max_model_len` 决定，而由**实际在途 token 数**决定。

**（3）写时复制（Copy-on-Write）**

并行采样（`n>1`）、beam search、前缀共享时，多条逻辑序列可以**共享同一批物理块**，只有发生写入的块才复制。这是 prefix caching 的思想雏形，也让 `n=8` 的采样显存从 8× 降到接近 1×。

> **面试要点**：PagedAttention 不是"一种 attention 算法"，而是**KV Cache 的内存管理方案**。它与 FlashAttention 正交：前者决定"KV 存在哪、怎么找"，后者决定"怎么算得快"。这一点在 0.25 版本"删除 PagedAttention 实现"之后尤其要讲清楚——被删的是**原始 CUDA kernel 实现**，块式 KV 管理作为**架构抽象**一直存在。

#### 1.2.3 当年的代码结构（V0 基因）

```
vllm/
├── engine/llm_engine.py          # 同步引擎，step() 驱动
├── core/scheduler.py             # 调度：prefill 优先、decode 优先两套策略
├── core/block_manager.py         # BlockSpaceManager：逻辑块 ↔ 物理块
├── worker/worker.py              # 执行者（TP rank）
├── worker/cache_engine.py        # 开显存池：按 num_gpu_blocks 分配 KV
└── model_executor/               # 模型与算子
```

这套结构的特征是：**调度器与执行器同步、单进程内 step 循环、block manager 只管块不管前缀哈希**。后面三年所有的重构，本质都是在拆这套结构。

#### 1.2.4 这个时代的版本要点

| 版本 | 日期 | 关键变化 | 为什么重要 |
|---|---|---|---|
| 0.0.1 | 2023-06-19 | 首个 PyPI 包（sdist） | 开源起点，论文配套代码 |
| 0.1.4 | 2023-08-25 | 首个 Linux wheel（cp38–cp311） | 从源码编译走向 `pip install` |
| 0.2.0 | 2023-09-28 | 多卡 TP、量化（AWQ/GPTQ/SqueezeLLM）、PagedAttention V2 kernel | 具备"上生产"的最小能力集 |
| 0.2.3 | 2023-12-03 | wheel 体积大幅下降（依赖拆分） | 镜像与分发成本下降 |
| 0.2.6 | 2023-12-17 | 稳定性与模型覆盖 | 年末成为事实标准 |

### 1.3 时代二（2024.01–2024.06）：从论文到生产

#### 1.3.1 三个必须解决的生产问题

1. **多轮对话/系统提示重复计算**：同一段 system prompt 每条请求都重算 prefill，浪费 30%–70% 的算力。
2. **长 prompt 阻塞 decode**：一条 32K 的 prefill 会让同批所有 decode 请求的 ITL 抖动到秒级。
3. **小模型/低并发下 GPU 打不满**：需要投机解码、需要更激进的批处理。

#### 1.3.2 自动前缀缓存（APC）

**做法**：以 block 为单位对 token 序列做**链式哈希**，哈希命中即可直接复用物理块，跳过对应 prefill。

```
block_hash[i] = hash(block_hash[i-1], tokens[i*B:(i+1)*B], 额外因素)
```

**关键工程细节**（高频面试题）：

- 哈希必须包含**影响该段 KV 的一切因素**：模型权重版本、LoRA id、多模态输入哈希、`block_size`、attention 类型（full/sliding window）、甚至 TP 拓扑（某些实现会带）。
- 只有**满块**才可缓存（partial block 在 0.25 之后才支持 `cache_partial_block`，见 §3.1.4）。
- 淘汰策略是 **LRU + 引用计数**：被运行中请求引用的块不能驱逐；空闲块按最后访问时间排队。
- 命中率是**首 token 延迟（TTFT）的第一影响因子**，但**不是越高越好**：哈希计算、块查找、以及在极端情况下"为了命中而延长序列驻留"都会带来成本。

#### 1.3.3 Chunked Prefill

**做法**：把一条长 prompt 的 prefill 切成多个 chunk，每个 step 只处理 `max_num_batched_tokens` 预算内的部分，与 decode 请求**混批**。

**收益**：

- ITL（inter-token latency）稳定：decode 不会被长 prefill 卡住。
- 显存峰值下降：不需要一次算完 128K 的激活。

**代价**：

- 长 prompt 的 TTFT 变长（prefill 被摊薄）。
- 需要处理"部分 prefill 的 KV 已写入、但 attention 元数据要标记为 chunked"的复杂性。
- 与投机解码、结构化输出、prefix caching 的交互是 bug 高发区。

**调参直觉**：`max_num_batched_tokens` 是"prefill 吞吐 / decode 延迟"的总阀门。**在线服务**常用 2048–8192（重延迟），**离线批处理**可到 16384–32768（重吞吐）。vLLM 在 **0.28.0 把默认值从 8192 提到 16384**，反映的是硬件与调度成熟后的默认取向变化。

#### 1.3.4 这个时代的其他拼图

| 能力 | 版本 | 影响 |
|---|---|---|
| 投机解码（speculative decoding） | 0.3.x 起步、0.4.x 推进 | 低并发下 1.5–2.5× 加速；高并发下可能负收益 |
| LLaVA 多模态 | 0.4.0 | 开启 VLM 服务时代，也带来 encoder cache 管理难题 |
| 自定义 AllReduce | 0.4.0 | 小消息 TP 通信优化，是 TPOT 的关键 |
| Prometheus 指标 | 0.4.x | 有了"可观测性"，才有"可运维" |
| `json_object` / 结构化输出雏形 | 0.4.0 | 后续 xgrammar/guidance 生态的起点 |

### 1.4 时代三（2024.07–2024.12）：吞吐战争与后端生态

#### 1.4.1 0.6.0 的"2× 吞吐"从哪来

官方口径是相对 0.5.3 的 **2 倍吞吐**。拆开看，三个来源：

1. **多步调度（multi-step scheduling）**：一次调度连续跑 N 个 decode step（`--num-scheduler-steps 8`），把 CPU 侧的调度/采样/ detokenize 开销摊薄到 1/8。**本质是用"调度前瞻性"换 CPU 效率**，代价是抢占/中止的响应变慢。
2. **异步输出处理（async output processor）**：把"构造输出数据结构、detokenize、流式回调"与 GPU 计算重叠，官方给出 **~12%** 吞吐提升。
3. **Kernel 质量**：FlashInfer 后端用于 FP8 KV Cache、投机解码的 rejection sampling 优化。

> **方法论**：这三条正好对应推理系统的三层开销——**调度层（CPU）、数据层（拷贝/序列化）、计算层（kernel）**。任何"吞吐上不去"的排查，都应先判断瓶颈在哪一层（见 §5.1）。

#### 1.4.2 生态扩张

- **硬件**：TPU、Intel XPU、AWS Neuron/Inferentia、Gaudi、ROCm。多硬件支持倒逼出 **Platform 抽象**（`vllm/platforms/`），这是后来国产卡接入的标准入口。
- **模型**：Qwen2 系列、DeepSeek-V2（MLA + MoE）、Llama 3.1、Mistral、Phi-3.5-MoE、Gemma 2。
- **MLA 的冲击**：DeepSeek-V2 的 latent attention 让 KV Cache 形状从 `(2, H, D)` 变成 `(512+64)` 的压缩 latent，**块管理、attention backend、量化全部要重写**。这是 vLLM 第一次为"非标准 attention"做架构级适配。

#### 1.4.3 V0 的结构性瓶颈（V1 重写的动因）

到 0.6.x，V0 已经积累了大量历史包袱：

| 瓶颈 | 表现 | 根因 |
|---|---|---|
| CPU 开销高 | 小模型/高 QPS 下 GPU 利用率 < 50% | 每 step 的 Python 对象构造、张量拼接、`torch.cat` 满天飞 |
| 调度与执行串行 | 调度时 GPU 空闲，执行时 CPU 空闲 | 单线程 step 循环 |
| prefill/decode 两套路径 | 行为不一致、bug 双份 | `_schedule_prefills` / `_schedule_default` 分叉 |
| 抢占只有 swap | swap 到 CPU 再换回，PCIe 带宽打满 | KV 块管理不支持"重算"作为一等公民 |
| 前缀缓存耦合 | 与 block manager 强耦合，难支持多级/远程 | 哈希逻辑嵌在 BlockSpaceManager 里 |

### 1.5 时代四（2025.01–2025.05）：V1 引擎重写

#### 1.5.1 V1 的设计目标（官方博客口径 + 工程解读）

1. **统一调度路径**：prefill 与 decode 用同一套 token budget 逻辑，取消两套代码。
2. **降低 CPU 开销**：`InputBatch`/`InputBuffers` 持久化、避免每 step 重建；引入 `torch.compile` 把 Python 开销编译掉。
3. **请求级 KV 管理**：KV 的分配/释放/抢占以 request 为单位，支持 recompute 作为默认抢占手段（**V1 去掉了 swap**）。
4. **可插拔**：scheduler、attention backend、KV connector 全部接口化。
5. **为规模化铺路**：DP/EP/PD 分离在 V1 上才可能干净实现。

#### 1.5.2 V1 的关键结构变化

| 维度 | V0 | V1 |
|---|---|---|
| 引擎进程模型 | 单进程 step 循环（可选 MQ 引擎） | **EngineCore 独立进程** + ZMQ/IPC，前端（tokenize/detokenize/HTTP）与核心解耦 |
| 抢占 | swap 为主 | **recompute 为主**（丢弃 KV，重新 prefill） |
| 前缀缓存 | 可选，默认关 | **默认开**，与块管理解耦 |
| CUDA Graph | 全图为主，动态 batch 受限 | **piecewise + full** 混合 |
| 编译 | 手写融合算子 | `torch.compile` + 自定义 op + 图切分 |
| 指标 | 引擎内统计 | EngineCore 通过消息上报，支持多前端 |

#### 1.5.3 版本节奏

- **0.7.0（2025-01-27）**：V1 Alpha（`VLLM_USE_V1=1`）；`torch.compile` 全面集成并在 V1 默认开启；新增 `LLM.sleep/wake_up`（为 RLHF 权重热更新铺路）。
- **0.8.0（2025-03-18）**：**V1 成为默认引擎**（V0 保留兼容）；V1 支持结构化输出、LoRA、PP 增强；引入**可插拔调度器**。
- **0.9.0（2025-05-28）**：PyTorch 2.7 + 默认 CUDA 12.8（破坏性升级）；Blackwell 初步优化；**DP/EP/PD 三件套起步**。

> **面试高频**：「V1 为什么去掉 swap？」
> 因为 swap 的收益依赖"被换出的序列很快回来"，但在真实负载里，被抢占的请求往往要等很久；而 swap 要占用 CPU 内存 + PCIe 带宽，还会阻塞其他请求的 KV 读写。V1 选择 **recompute**：直接丢弃 KV，恢复时重新 prefill。代价是 prefill 算力，收益是**实现简单、显存零额外占用、与 prefix caching 天然协同**（重算时大概率能命中前缀缓存）。这也是"显存换带宽"到"算力换带宽"的一次典型权衡。

### 1.6 时代五（2025.07–2026.11）：规模化、V2 Runner 与去 PagedAttention

#### 1.6.1 0.10–0.13：V0 退役与异步化

| 版本 | 关键变化 | 工程含义 |
|---|---|---|
| 0.10.0（2025-07-25） | 开始删除 V0 代码；**实验性异步调度** `--async-scheduling` | 调度与 GPU 执行重叠，CPU 不再是瓶颈 |
| 0.11.0（2025-10-04） | **V0 引擎彻底移除**；CUDA Graph 默认改为 `FULL_AND_PIECEWISE` | 只维护一条代码路径；细粒度 MoE 也能全图捕获 |
| 0.12.0（2025-12-03） | PyTorch 2.9 / CUDA 12.9；EAGLE 多步 CUDA Graph；ROCm 扩展 | 升级节奏跟随 PyTorch；AMD 成为一等公民 |
| 0.13.0（2025-12-19） | 稳定性与模型覆盖 | — |

**异步调度是什么**（0.14 起默认，必考）：

```
同步：  [调度 step N] → [GPU 执行 step N] → [取回输出] → [调度 step N+1]
异步：  [调度 step N] → [GPU 执行 step N] ↘
                [调度 step N+1]（用上一轮结果的预测） → [GPU 执行 step N+1]
```

调度器在 GPU 执行本轮时就开始准备下一轮输入，**CPU 与 GPU 真正重叠**。代价是调度器必须处理"输出尚未返回"的不确定性（提前假设采样结果、必要时回滚），因此对投机解码、结构化输出、抢占逻辑都有额外约束。官方在 0.14 明确：**PP、CPU 后端、部分 spec decode 组合暂不支持**。

#### 1.6.2 0.14–0.21：架构换代

| 版本 | 关键变化 | 为什么重要 |
|---|---|---|
| 0.14.0（2026-01-20） | **异步调度默认开启**；gRPC 入口；`--max-model-len auto` | 消除 OOM 启动失败；服务协议不再只有 REST |
| 0.16.0（2026-02-26） | 异步调度 + PP 完整支持（官方称 **+30.8% E2E 吞吐、+31.8% TPOT**）；Realtime WebSocket API | 大模型多机部署的默认组合成型 |
| 0.18.0（2026-03-20） | gRPC 正式；**无 GPU 的 Render 服务**（`vllm launch render`）；KV offload 改进；弹性 EP | 多模态预处理与 GPU 解耦；CPU 内存成为 KV 二级缓存 |
| 0.20.0（2026-04-27） | **DeepSeek V4 支持**；CUDA 13 默认；Transformers v5；**FA4 成为 MLA prefill 默认**；**TurboQuant 2-bit KV**；在线量化前端 | 低精度 KV 从实验走向默认；KV 容量再 ×4 |
| 0.22.0（2026-05-29） | **MRv2 对 Qwen3 dense 默认**；Rust 前端；多级 KV offload | 执行器开始换代；前端性能成为独立战场 |

#### 1.6.3 0.22–0.34：Model Runner V2 与 KV 基础设施

这是"当前最新技术栈"的核心，务必逐条理解：

| 版本 | 关键变化 | 面试价值 |
|---|---|---|
| 0.24.0（2026-06-30） | MRv2 默认支持量化模型；Streaming Parser Engine 统一工具调用/推理解析；DeepEP v2；不再内部设置 `CUDA_VISIBLE_DEVICES`（改 `device_ids`） | 解析器从"每模型一个"走向统一框架 |
| **0.25.0（2026-07-11）** | **MRv2 成为所有 dense 模型默认**；**PagedAttention 原始实现被删除**；TLI 异构词表投机解码；DSpark/DFlash drafter | 旧 attention 实现退场，块管理抽象保留 |
| 0.26.0（2026-08-10） | 支持**按 KV cache group 选择不同 attention backend**；sliding window 成为显式后端能力 | 混合模型（full + SWA + Mamba）的必需品 |
| 0.28.0（2026-08-26） | Decode Context Parallel（DCP）；**`max_num_batched_tokens` 默认 8192→16384**；Blackwell CUDA Graph 捕获上限 1024 | 长上下文解码的并行新维度 |
| 0.30.0（2026-09-22） | **Fast Start**（`--load-format ipc_cache`）；**HiSparse**（把稀疏 MLA 的 KV 页溢出到 pinned host memory）；Watermarking；PCP+DCP | 启动时间从分钟级降到秒级；显存不足时的"软降级"方案 |
| 0.32.0（2026-10-18） | **NVFP4 KV Cache 覆盖 SM8x/SM9x/SM12x**；GLM-5.x 在 SM100 默认 FP8 KV；`--api-server-count` 多 API worker 负载均衡（短请求 +43–85% req/s） | 低精度 KV 进入主流；前端横向扩展 |
| **0.34.0（2026-11-16）** | **可扩展 KV 默认**（按 warmup 后真实空闲显存定容，官方建议移除 `gpu_memory_utilization`）；**MRv1 彻底删除**；**block_size 默认 64**；1,061 commits / 412 contributors | "参数调优"范式变化：从手动预算到自动定容 |

#### 1.6.4 Model Runner V2 到底改了什么

MRv2 不是"V1 引擎的小改"，而是**执行层的重写**，目标是让 runner 成为"可编排的执行图"：

| 能力 | MRv1 的限制 | MRv2 的做法 |
|---|---|---|
| 输入构造 | 每 step 重建张量、大量 host→device 拷贝 | `InputBatch`/`Buffer` 持久化 + 增量更新（`prepare_inputs`） |
| 采样 | 与执行耦合 | `sample`/`sample_tokens`/`postprocess_sampled` 分离，支持异步 |
| 投机解码 | 各方法各自为政 | `v1/worker/gpu/spec_decode/` 统一 drafter 接口（EAGLE/DFlash/DSpark/MTP…） |
| RLHF 热更新 | 需重启 | `reload_weights` / `update_config` / sleep-mode |
| 并行 | TP/PP 逻辑散落 | `pp_utils`/`cp_utils`/`dp_utils`/`eplb_utils` 模块化 |
| 图捕获 | 单一模式 | `cudagraph_utils` + 按 batch 形状的 UBatching |

**为什么要重写**：V1 引擎把调度做干净之后，**runner 成了新的 CPU 瓶颈与复杂度中心**（每 step 的 Python 开销、spec decode 分支、多模态、MoE 路由、EPLB、PP 气泡）。MRv2 用"状态化 + 分阶段 + 可插拔 drafter"把这团乱麻拆开。

### 1.7 五条技术主线（看懂版本史的正确姿势）

不要按版本号死记，按**五条主线**记，每条线都有自己的"目标函数"和"代价"：

| 主线 | 目标 | 关键手段 | 代价/边界 | 代表版本 |
|---|---|---|---|---|
| **① KV 显存管理** | 单位显存装更多 token | 分页 → 前缀缓存 → 低精度 KV（FP8/NVFP4/2bit）→ 分层 offload → 可扩展 KV | 精度、查找开销、PCIe 带宽 | 0.1 / 0.4 / 0.20 / 0.30 / 0.34 |
| **② 调度与并发** | 让 GPU 永不空转 | continuous batching → chunked prefill → 多步调度 → 异步调度 → SLO 感知调度 | 延迟抖动、实现复杂度 | 0.1 / 0.4 / 0.6 / 0.14 |
| **③ 执行与编译** | 降低每 step 的 CPU/launch 开销 | CUDA Graph → piecewise/full → torch.compile → MRv2 → IR | 动态形状、捕获失败、编译时间 | 0.4 / 0.7 / 0.11 / 0.25 / 0.20(IR) |
| **④ 并行与通信** | 单卡放不下、单机不够快 | TP → PP → DP → EP(+EPLB) → CP/DCP/PCP | 通信墙、负载不均、拓扑约束 | 0.2 / 0.9 / 0.16 / 0.28 / 0.30 |
| **⑤ 服务化与生态** | 从库到平台 | OpenAI API → 指标 → 结构化输出 → gRPC/Rust 前端 → PD 分离 → 多级缓存 | 协议兼容、运维复杂度 | 0.1 / 0.4 / 0.14 / 0.22 / 0.30 |

**用这条框架回答"vLLM 这几年的演进"**，比背版本号高一个层级：每一代都是在**某条主线遇到物理墙**时，换一种权衡方式。

### 1.8 升级纪律：版本差异是线上事故的第一来源

vLLM 的破坏性变更密度很高（PyTorch/CUDA 大版本、默认值、API 删除）。工程上建议：

1. **锁定镜像 + 锁定 commit**，不要 `pip install -U vllm`。
2. **升级前读 Release Notes 的 Breaking Changes**（近两年的典型：0.9 PyTorch 2.7/CUDA 12.8、0.10 V0 开始删除、0.11 V0 移除、0.14 async 默认、0.20 CUDA 13、0.25 PagedAttention 移除、0.28 bitsandbytes 转插件、0.32 移除 cascade attention、0.34 block_size 64）。
3. **建立三层回归**：正确性（固定 prompt 的 greedy 输出逐 token 比对）→ 性能（固定 workload 的 TTFT/TPOT/吞吐）→ 稳定性（4h 长跑 + 抢占注入）。
4. **灰度维度**：同一模型双版本并行，按流量比例切换；关注 **prefix cache 命中率**与 **preemption 次数**这两个"隐性回归指标"，它们比吞吐更早暴露问题。

**升级检查清单（可直接贴到 PR 模板）**：

```text
[ ] 镜像 digest 是否锁定？CUDA / PyTorch / Python 版本是否匹配驱动？
[ ] 是否命中 Breaking Changes？（默认值、参数删除、API 改名）
[ ] 模型结构是否被重命名（如 deepseek_v4 独立包）？自定义注册是否失效？
[ ] 自定义 attention backend / 量化 kernel 是否仍被选中？（打印实际后端名）
[ ] CUDA Graph 是否仍能捕获？（启动日志有无 fallback 到 eager）
[ ] 固定 prompt 输出是否逐 token 一致？
[ ] 压测：TTFT p99 / TPOT p99 / 吞吐 / prefix 命中率 / 抢占次数 是否在阈值内？
[ ] 4 小时长跑无 OOM、无句柄泄漏、无 NCCL 超时？
```

---

> [返回总览](00-总览.md) ｜ 下一篇：[02-宏观架构](02-宏观架构.md)


# 第 2 章 宏观架构

> [← 总览](00-总览.md) ｜ 下一章：[03a-核心机制-KV与调度](03a-核心机制-KV与调度.md)

### 2.1 分层视图

```
┌──────────────────────────────────────────────────────────────────────┐
│ 接入层  OpenAI REST / Anthropic Messages / gRPC / Rust 前端 / Python API │
│        (tokenize、chat template、模板渲染、流式 SSE、鉴权、限流)          │
├──────────────────────────────────────────────────────────────────────┤
│ 引擎层  LLMEngine / AsyncLLM（请求编排、输出处理、detokenize、指标）      │
│        └─ EngineCoreClient ──(ZMQ / IPC)──> EngineCore                  │
├──────────────────────────────────────────────────────────────────────┤
│ 核心层  Scheduler（调度）  KVCacheManager（块分配/前缀缓存/抢占）         │
│        StructuredOutputManager  KVConnector（P/D 与 offload）           │
├──────────────────────────────────────────────────────────────────────┤
│ 执行层  Executor（uni-proc / multiproc / ray）→ Worker(TP/PP rank)      │
│        └─ GPUModelRunner（MRv2：prepare_inputs → execute → sample）      │
├──────────────────────────────────────────────────────────────────────┤
│ 算子层  AttentionBackend（FA/FlashInfer/MLA/Triton/TurboQuant…）        │
│        GEMM/MoE（CUTLASS/TRTLLM-GEN/CuTeDSL/Triton）、Sampler、通信算子   │
├──────────────────────────────────────────────────────────────────────┤
│ 硬件层  NVIDIA / AMD / Intel / TPU / 国产加速卡（Platform 插件）          │
└──────────────────────────────────────────────────────────────────────┘
```

**为什么 EngineCore 要独立进程**（V1 的核心决策之一）：

- **CPU 隔离**：tokenize/detokenize、HTTP 解析、JSON 序列化是 CPU 密集的，和调度循环抢 GIL 会直接抬高 TPOT。
- **多前端**：一个 EngineCore 可以挂多个 API server（`--api-server-count`，0.32 起进一步做 least-loaded 路由）。
- **故障隔离**：前端崩溃不丢 KV；核心崩溃可由 supervisor 重启。
- **代价**：多一次 IPC（序列化 + 拷贝），所以 V1 对 `SchedulerOutput`/`ModelRunnerOutput` 做了极致的**零拷贝/共享内存**优化，这也是 `tensor_ipc.py`、`serial_utils.py` 存在的原因。

### 2.2 进程与线程拓扑

以「单机 8 卡 TP=8 + 2 个 API worker」为例：

```
进程 0: API Server #1   ─┐
进程 1: API Server #2   ─┼─ ZMQ ──> 进程 2: EngineCore（调度主循环，单线程）
进程 3: Tokenizer 组    ─┘                 │
                                           ├─ 同进程内：Scheduler + KVCacheManager（纯 Python，无 GPU）
                                           │
                                           └─ Executor（multiproc）
                                                ├─ Worker rank0（GPU0，driver）
                                                ├─ Worker rank1 … rank7（各持一段模型分片）
                                                └─ 通信：NCCL（TP all-reduce）/ 共享内存（元数据）
```

关键点：

- **调度是单线程的**。所有"并发"都来自批处理，不是多线程调度器。这决定了：**任何调度路径上的 Python 阻塞（哈希计算、日志、锁、序列化）都会直接变成 GPU 空转**。
- **KV 元数据在 CPU，KV 张量在 GPU**。Scheduler 只操作 block id；真正写 KV 的是 attention backend（通过 `slot_mapping` 把 token 写到对应物理块）。
- **TP rank 之间只同步"要算什么"，不同步"调度决策"**：rank0 的 scheduler 产出 `SchedulerOutput`，广播给所有 rank 执行。

### 2.3 一次请求的完整生命周期

```
① HTTP 请求 → chat template 渲染 → tokenize → 构造 Request
② add_request()：入 waiting 队列（FCFS + priority）
③ schedule()：
     - 计算可复用的前缀块（get_computed_blocks）→ 命中的 token 直接跳过 prefill
     - allocate_slots()：为剩余 token 分配物理块
     - 组 SchedulerOutput：new_reqs / cached_reqs / num_scheduled_tokens / block_ids
④ execute_model()：
     - prepare_inputs：从持久化 buffer 增量拼 input_ids / positions / slot_mapping / block_table
     - forward：embedding → layers(attention + MLP/MoE) → logits
     - sample：greedy / top-p / top-k / logprobs / 结构化输出 mask
⑤ update_from_output()：
     - 追加 token，判断 stop（EOS / max_tokens / stop strings / stop_token_ids）
     - 若跨块：allocate_slots 追加新块；若显存不足：抢占（preempt）其他请求
     - 产出 EngineCoreOutput → detokenize → SSE 流式返回
⑥ 完成：free 块、更新 prefix cache、上报指标（TTFT/TPOT/tokens）
```

这条链路上**每一个箭头都是可优化点**，也是排障时的定位坐标：

| 环节 | 典型耗时占比（经验值） | 常见问题 |
|---|---|---|
| tokenize + 模板渲染 | 1–5%（长 prompt 可到 10%+） | Jinja 模板慢、tokenizer 单线程 |
| 调度 | 2–10%（小模型高 QPS 可到 30%） | Python 开销、哈希、抢占风暴 |
| forward | 60–85% | kernel 选择、图捕获、通信 |
| sample | 1–10% | 词表大（150K+）、结构化输出 mask |
| detokenize + SSE | 1–5% | 逐 token flush、JSON 序列化 |

### 2.4 组件职责与代码地图（0.25.1 实测）

| 组件 | 真实路径 | 职责 | 扩展方式 |
|---|---|---|---|
| 引擎入口 | `vllm/v1/engine/llm_engine.py`、`async_llm.py` | 请求编排、输出处理 | 一般不改，改上层 |
| 核心循环 | `vllm/v1/engine/core.py`、`core_client.py` | EngineCore 主循环、IPC | 谨慎 |
| 调度器 | `vllm/v1/core/sched/scheduler.py`（`class Scheduler`）、`async_scheduler.py`、`interface.py` | 批组成、抢占、KV 连接器协调 | **可插拔**（`scheduler_cls`） |
| KV 管理 | `vllm/v1/core/kv_cache_manager.py`（`KVCacheManager`）、`block_pool.py`（`BlockPool`）、`kv_cache_utils.py` | 块分配、哈希、淘汰 | 改这里要极谨慎 |
| KV 规格 | `vllm/v1/kv_cache_interface.py`、`kv_cache_spec_registry.py` | 每层的 KV 形状/类型（Full/MLA/Mamba/SWA） | 新模型结构需扩展 |
| 注意力 | `vllm/v1/attention/backends/*.py`、`registry.py` | attention 实现与选择 | **可插拔**（`register_backend`） |
| 执行器 | `vllm/v1/executor/{uniproc,multiproc,ray}_executor.py` | 拉起 worker、广播指令 | 多机/自定义用 Ray |
| Worker | `vllm/v1/worker/gpu_worker.py`、`worker_base.py` | 设备初始化、显存池、profile | 换硬件时重写 |
| Runner（MRv2） | `vllm/v1/worker/gpu/model_runner.py`（`class GPUModelRunner`） | 输入构造、执行、采样 | 复杂定制入口 |
| Runner（MRv1，≤0.33） | `vllm/v1/worker/gpu_model_runner.py` | 旧执行路径 | 0.34 已删除 |
| 投机解码 | `vllm/v1/spec_decode/{eagle,medusa,ngram_proposer,dflash,draft_model,...}.py` | draft/verify | 新 drafter 可注册 |
| 结构化输出 | `vllm/v1/structured_output/` | 语法/JSON schema → bitmask | 后端可选（xgrammar/guidance） |
| KV 连接器 | `vllm/v1/kv_offload/`、`vllm/distributed/kv_transfer/` | P/D 与 offload | **插件** |
| 配置 | `vllm/config/*.py`（`cache.py`/`scheduler.py`/`parallel.py`/`speculative.py`…） | 所有参数定义与校验 | 加参数改这里 |

### 2.5 V0 / V1 / MRv2 的对照（一张表讲清"现在是什么"）

| 关注点 | V0（≤0.10，已删除） | V1（0.8+，当前） | MRv2（0.25+ 默认，0.34 唯一） |
|---|---|---|---|
| 调度单位 | 序列 + 双路径（prefill/decode） | **token budget 统一路径** | 同 V1 |
| KV 抢占 | swap / recompute | recompute | recompute + 更细粒度 |
| 输入构造 | 每 step 重建 | `InputBatch` 持久化 | `InputBatch` + `Buffer` + 增量 |
| 图捕获 | 全图为主 | piecewise + full | UBatching、按形状分桶 |
| 采样 | 同步 | 同步/异步输出 | 采样阶段化（sample → postprocess） |
| 扩展点 | 少 | scheduler/backend/connector | + drafter / 自定义 runner |

---

> 上一篇：[01-发展史](01-发展史.md) ｜ [返回总览](00-总览.md) ｜ 下一篇：[03a-核心机制-KV与调度](03a-核心机制-KV与调度.md)


# 第 3 章 核心机制（上）：KV Cache、调度与前缀缓存

> [← 总览](00-总览.md) ｜ 下一章：[03b-核心机制-执行与并行](03b-核心机制-执行与并行.md)

### 3.1 KV Cache：从 PagedAttention 到"可扩展 KV"

#### 3.1.1 数据结构：逻辑块、物理块、块表、槽位映射

```
KVCacheConfig
 ├─ num_blocks: int                      # 物理块总数
 ├─ kv_cache_tensors: list[KVCacheTensor]  # 每层的 K/V 张量（可能多 group）
 └─ kv_cache_groups: list[KVCacheGroupSpec]
        └─ 每 group: layer_names + KVCacheSpec

KVCacheSpec（每层的"KV 几何"）
 ├─ FullAttentionSpec: block_size, num_kv_heads, head_size, dtype
 ├─ MLAAttentionSpec:  latent dim（如 512 + rope 64）
 ├─ SlidingWindowSpec: window_size
 └─ MambaSpec: state shape, block_size

运行时（每个 step）
 ├─ block_table: [num_reqs, max_num_blocks]  int32   # 逻辑→物理
 ├─ slot_mapping: [num_tokens]               int64   # 每个 token 写到哪个 (block, offset)
 └─ seq_lens / query_start_loc / positions
```

**slot_mapping 是理解 vLLM 的钥匙**：它把"第 i 个 token 的 KV"映射到"物理块 b 的第 o 个槽位"，attention kernel 用 `block_table + slot_mapping` 完成 scatter/gather。**所有 KV 量化、offload、P/D 传输，本质都是在这两个结构上做文章。**

#### 3.1.2 显存容量怎么算（必须会手算）

**每 token 的 KV 字节数**：

```
MHA/GQA:  bytes = 2 × L × H_kv × D × dtype_bytes
MLA:      bytes = L × (D_latent + D_rope) × dtype_bytes      # 只有一份 latent，不乘 2
Mamba:    不是 per-token，而是 per-state（固定大小 × 层数）
```

**总容量**：

```
num_gpu_blocks = floor(可用KV显存 / (block_size × per_token_bytes))
最大并发 token 数 ≈ num_gpu_blocks × block_size
```

**例子**：Qwen3-32B，40 层，GQA 8 个 KV head，head_dim 128，BF16，TP=4（KV head 也被切 4 份 → 每卡 2 个 KV head）：

```
per_token = 2 × 40 × 2 × 128 × 2 B = 40,960 B ≈ 40 KB/token
80GB 卡，权重 16GB/卡，可用 KV ≈ (80×0.9 − 16 − 激活3) ≈ 53 GB
num_tokens ≈ 53e9 / 40960 ≈ 1.29M token
block_size=16 → 约 80,000 个块；block_size=64 → 约 20,000 个块
```

> 注意：**块大小不改变总 token 容量，只改变碎片率与元数据开销**。block_size 小 → 尾部浪费少、block_table 长；block_size 大 → 元数据少、attention 内循环更友好、但尾部最多浪费 63 token/序列。0.34 把默认从 16 提到 64，是在"平均序列长度变长 + kernel 偏好大块"的趋势下做的取舍。

**MLA 的账**（以 DeepSeek 系为例）：每层每 token 只存 `512（latent）+ 64（rope）` 个元素，BF16 下 ≈ 1152 B/层，61 层 ≈ 70 KB/token——**和同级 GQA 模型相当，但因为不需要重复存 K/V，长上下文优势明显**。

#### 3.1.3 分配与回收：`KVCacheManager` 的最小语义

0.25.1 的核心方法（实测签名存在）：

```python
km = KVCacheManager(...)
blocks, num_computed = km.get_computed_blocks(request)   # 前缀命中
new_blocks = km.allocate_slots(request, num_new_tokens, num_lookahead_tokens=...)
km.cache_blocks(request, num_computed_tokens)            # 落前缀缓存
km.free(request)                                          # 请求结束
km.evict_blocks(block_ids)                                # 显式驱逐（如 offload）
```

`allocate_slots` 的语义要点：

1. **先扣已命中块**，再按"从 `num_computed_tokens` 到 `num_computed_tokens + num_new_tokens`"需要的新块数分配。
2. 分配失败 → **返回 None**，由 Scheduler 触发**抢占**（而不是抛异常）。
3. 支持 **lookahead**（投机解码要预留 draft token 的槽位）。
4. 跨块时**触发一次 block 拷贝语义**（V1 中通过 `slot_mapping` 直接写，不需要显式 memcpy）。

#### 3.1.4 前缀缓存的两个实现层次

**（1）整块缓存（经典）**：只有凑满 `block_size` 的块才进哈希表。

**（2）部分块缓存（0.25 起 `cache_partial_block`）**：允许把"最后一个不满的块"也缓存，用 `(parent_hash, start, tokens)` 生成 partial hash。

为什么重要：在 **RAG/多轮对话**里，prompt 长度往往不是 block_size 的整数倍，整块缓存会浪费最后 1–63 个 token 的复用机会；在**长 system prompt + 短问题**场景，partial 缓存能把命中率从 ~85% 拉到 ~95%+（经验值，取决于长度分布）。

**代价**：哈希计算量上升、块内容可能被后续 token 覆盖（需要 copy-on-write 或重新计算），因此实现上要处理"partial 块被写满后转正"的边界。

#### 3.1.5 抢占（Preemption）：V1 的核心权衡

```
allocate_slots 失败
   └─> Scheduler 选一个"受害者"（通常是最后到达 / 最低优先级 / 运行中最长的）
         └─> 释放其全部块，状态置为 PREEMPTED，已算 token 数清零
               └─> 放回 waiting 队列头部
                     └─> 下一轮重新 prefill（可能命中前缀缓存，代价降低）
```

**为什么 V1 敢用 recompute 取代 swap**：

| 维度 | swap | recompute |
|---|---|---|
| 额外显存 | 需要等量 CPU pinned 内存 | 0 |
| 带宽 | PCIe 双向（KV 很大时是瓶颈） | 0（但消耗算力） |
| 恢复延迟 | 取决于 swap 量 | 取决于重算长度（有前缀缓存时很快） |
| 实现复杂度 | 高（异步 swap、块状态机） | 低 |
| 与 prefix cache | 协同差 | 天然协同 |

**工程结论**：**抢占次数是比吞吐更敏感的容量指标**。线上应监控 `vllm:num_preemptions_total`，稳态下应接近 0；持续增长说明 KV 预算不足或 `max_num_seqs` 过大。

#### 3.1.6 0.34 的"可扩展 KV"（Extensible KV Cache）

官方口径：**按 warmup 后真实空闲显存自动定容**，建议移除 `gpu_memory_utilization`。

**解决的真问题**：`gpu_memory_utilization=0.9` 是一个**启动前的静态猜测**，它必须同时覆盖：

- 权重（已知）
- 激活峰值（**未知**，取决于 batch 形状、序列长度、是否 chunked、是否 CUDA Graph）
- CUDA Graph 捕获占用的显存（**未知**，取决于捕获的 batch size 集合）
- NCCL/通信 buffer、CUDA context、碎片

猜大了 → 启动 OOM；猜小了 → KV 少、并发低。0.34 改为**先跑 warmup（含图捕获），测量真实余量，再按剩余显存分配 KV**，把"用户猜"变成"系统测"。

**对调参的影响**：

- `--gpu-memory-utilization` 从"主参数"降级为"兼容/上限约束"。
- 显存调优的主要抓手变成：**`--max-model-len`（或 auto）、`--kv-cache-dtype`、`--max-num-seqs`、block_size、是否开启 prefix caching**。
- 多进程/多卡场景下，**rank 间显存不均衡**（MoE 专家分布）会让"自动定容"取最小值，因此 EP 场景仍需关注 EPLB。

### 3.2 调度器

#### 3.2.1 调度的目标函数

调度器每个 step 要解一个**带约束的装箱问题**：

```
maximize  Σ 有效token吞吐
s.t.      Σ num_scheduled_tokens ≤ max_num_batched_tokens     （计算预算）
          num_running_seqs ≤ max_num_seqs                       （并发预算）
          KV 块需求 ≤ 空闲块                                     （显存预算）
          每个 running 请求本 step 至少推进 1 token（除非被抢占）
          公平性/优先级/SLO 约束
```

vLLM 的解法是**贪心 + 分层**：先保证所有 running 请求的 decode（每人 1 token），剩余预算给 waiting 的 prefill（可被 chunk）。

#### 3.2.2 `schedule()` 主干（概念伪代码，对照 0.25.1 的方法名）

```python
def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
    # 0) 处理被抢占/远端 KV 就绪的请求，决定是否提升回 running
    self._try_promote_blocked_waiting_request(...)

    # 1) 先排 running：每条 1 个 decode token（投机解码是 1 + num_spec_tokens）
    for req in running:
        num_tokens = 1 + req.num_spec_tokens
        if budget_left < num_tokens: preempt(req); continue
        blocks = kv.allocate_slots(req, num_tokens, lookahead=...)
        if blocks is None: preempt(req); continue      # 显存不足 → 抢占
        scheduled[req] = num_tokens

    # 2) 再排 waiting：prefill，按 chunk 切分（chunked prefill）
    while waiting and budget_left > 0 and seqs_left > 0:
        req = self._select_waiting_queue_for_scheduling()  # FCFS/优先级
        computed, num_new = kv.get_computed_blocks(req)    # 前缀命中
        num_new = min(req.num_tokens - computed, budget_left, chunk_limit)
        blocks = kv.allocate_slots(req, num_new)
        if blocks is None: break                           # 留给下一 step
        scheduled[req] = num_new

    # 3) 组装输出：新请求 / 缓存命中请求 / 增量 token / 块表
    self._update_after_schedule(out)
    return out          # → Executor → GPU
```

#### 3.2.3 三个必懂的子问题

**（1）为什么 decode 优先？**
decode 的 token 是"已在途的承诺"，卡住会直接抬高 TPOT（用户可感知的卡顿）；prefill 是"新承诺"，晚一个 step 只影响 TTFT 的几毫秒。**交互式服务必须 decode 优先**；离线批处理可以把比例调向 prefill。

**（2）chunk 大小怎么定？**
`max_num_batched_tokens` 是总预算，chunk 的实际大小 = 预算 − decode 占用。经验配置：

| 场景 | max_num_batched_tokens | max_num_seqs | 说明 |
|---|---|---|---|
| 低延迟 API（短 prompt，TTFT 敏感） | 2048–4096 | 128–256 | 小 batch 降排队，decode 优先 |
| 通用在线服务 | 8192 | 256 | vLLM 长期默认 |
| 长上下文/RAG | 16384–32768 | 64–128 | 大 chunk 提 prefill 吞吐，需盯 ITL |
| 离线批量 | 32768+ | 512+ | 吞吐最大化，允许 TTFT 变差 |

**（3）抢占选谁？**
默认倾向"最后进入 running 的请求"（LIFO 式），理由是它最可能还没产生用户可见输出，且重算成本最低。生产中如果需要 SLO 保障，应改成**按优先级/剩余预算/已生成 token 数**加权，这就是 §6.4 的自定义调度器实战。

#### 3.2.4 异步调度（Async Scheduling）

**机制**：调度器在 GPU 执行 step N 时，基于 `ModelRunnerOutput` 的"预期"准备 step N+1。为此需要：

- 调度器**不阻塞等待**输出，改用事件/回执；
- 对"提前假设"的部分（如采样结果、停止判断）设计**回滚或幂等**；
- 与结构化输出（grammar bitmask 依赖上一步 token）、投机解码（draft 依赖上一步输出）强耦合，因此这些组合被逐步放开（0.14 默认、0.16 支持 PP、0.28 支持 draft model）。

**收益来源**：把每 step 的 CPU 时间藏进 GPU 时间里。**CPU 越忙（小模型、短请求、高 QPS、多 LoRA、复杂模板），收益越大**；大模型长 decode（GPU 时间远超 CPU）收益趋近于 0。

**风险**：一旦某步调度假设错误，会表现为"输出轻微错乱/重复/吞 token"，且**只在特定并发下复现**。压测时要专门跑"高并发 + 结构化输出 + 投机解码"的组合回归。

#### 3.2.5 Pause / Resume（RLHF 与运维）

- **Pause**：停止接受新请求、把在途请求的 KV 保留（或按策略丢弃）、返回可恢复状态。
- **Resume**：`update_config` + `reload_weights` 后继续，无需重启引擎。
- 典型用途：RLHF 训练-推理权重同步（0.16 起 NCCL 权重同步、0.30 Fast Start）、线上热更新模型、故障切换。

### 3.3 Prefix Caching

#### 3.3.1 哈希链的设计

```python
# 概念实现（vLLM 用 blake3/sha256 变体 + 自定义 tuple hash）
def hash_block(parent_hash, token_ids, extra: tuple) -> BlockHash:
    # extra 通常包含：LoRA id、多模态输入 hash、cache_salt、block 在 group 中的位置
    return H(parent_hash, tuple(token_ids), extra)
```

**必须进哈希的因素**（漏一个就是"串味"事故）：

| 因素 | 为什么 | 事故形态 |
|---|---|---|
| 模型权重版本 | 同结构不同权重，KV 不可复用 | 输出胡言乱语 |
| LoRA / adapter id | 不同 adapter 的 KV 不同 | 语气/知识串台 |
| 多模态输入哈希 | 图片/音频变化但 prompt 相同 | 答非所问 |
| `block_size` | 块边界不同 | 命中错位 |
| attention 类型/窗口 | SWA 的 KV 只覆盖窗口 | 长文丢上下文 |
| TP/PP 拓扑 | 分片方式不同，物理块布局不同 | 跨实例共享时崩溃 |
| `cache_salt`（安全） | 多租户隔离，防止跨租户探测 | 隐私泄露 |

#### 3.3.2 命中、复用、淘汰

```
请求到达
  → 逐块算 hash（O(prompt_len)，可与 tokenize 流水化）
  → BlockPool.get_cached_block(hash)：命中则引用计数 +1，touch LRU
  → 命中的 token 数 = 命中块数 × block_size（partial 块另算）
  → 剩余部分进入正常 prefill
  → 请求结束时 cache_blocks()：把新算的满块写入哈希表
  → 空闲块不足时：从 free queue 头部淘汰（LRU），引用计数 >0 的块跳过
```

**关键指标**（Prometheus）：

```
vllm:gpu_prefix_cache_queries_total
vllm:gpu_prefix_cache_hits_total
命中率 = hits / queries        # 注意这是"块级"，不是"token 级"
vllm:gpu_cache_usage_perc      # KV 占用率
vllm:num_preemptions_total     # 抢占次数
```

#### 3.3.3 命中率上不去的 8 个原因（排障清单）

1. **prompt 太短**：不足一个块（block_size=16 时 <16 token 基本无收益）。
2. **前缀不稳定**：模板里带时间戳、随机 ID、`system` 里插入用户昵称 → 第一块就不同。
3. **多租户混跑**：不同租户的 system prompt 不同，缓存互相挤占（用 `cache_salt` 隔离，但要接受命中率下降）。
4. **KV 预算太小**：块被频繁淘汰，命中率被 LRU 稀释。
5. **请求顺序随机**：先来长尾请求把缓存冲掉（可用"保留区间/分区"缓解）。
6. **block_size 过大**：短前缀根本凑不满块（0.34 默认 64 后更明显，短 prompt 场景可显式调小）。
7. **模型/LoRA 切换**：哈希带 adapter id，切换即失效。
8. **多模态**：图片 hash 每次都不同（如带签名的 URL 被下载后重新编码）。

#### 3.3.4 前缀缓存的"暗坑"

- **与 chunked prefill 的交互**：chunk 边界不必对齐块边界，但"部分算完的块"不能进缓存，否则会把半截 KV 当完整块复用。V1 用 `num_computed_tokens` 严格区分"已算"与"已缓存"。
- **与投机解码的交互**：draft token 的 KV 可能被写入，但若验证失败必须能**回滚**（V1 通过 `num_computed_tokens` 与块释放实现）。
- **与 offload 的交互**：offload 到 CPU/磁盘的块被淘汰回 GPU 时，哈希表要能"重新挂载"，否则命中率归零（0.30 的 tiered KV 就是干这个的）。
- **安全性**：prefix cache 是**跨请求共享**的，天然存在侧信道（通过响应时间探测他人 prompt）。多租户必须开 `cache_salt` 或关闭跨租户共享。

---

> 上一篇：[02-宏观架构](02-宏观架构.md) ｜ [返回总览](00-总览.md) ｜ 下一篇：[03b-核心机制-执行与并行](03b-核心机制-执行与并行.md)


# 第 3 章 核心机制（中）：Attention 后端、执行、投机与并行

> [← 总览](00-总览.md) ｜ 下一章：[03c-核心机制-量化与服务化](03c-核心机制-量化与服务化.md)

### 3.4 Attention 后端：vLLM 最复杂的"可插拔层"

#### 3.4.1 为什么需要这么多后端

attention 是**唯一同时被以下因素撕扯**的算子：

- **形状**：prefill（长 query、因果 mask）vs decode（q_len=1、KV 很长）→ kernel 完全不同
- **KV 布局**：分页块、MLA latent、滑动窗口、Mamba state
- **精度**：BF16 / FP8 KV / NVFP4 KV / 2-bit KV
- **硬件**：SM80/89/90/100/120、ROCm gfx950、XPU、国产卡
- **特性**：投机解码的 tree mask、chunked prefill 的 prefix、多模态双向注意力

所以 vLLM 把 attention 抽象成 **backend + metadata + builder** 三件套，由 registry 按"平台 + 模型 + 配置"自动选择。

#### 3.4.2 后端能力矩阵（0.25–0.34 口径）

| 后端 | 典型场景 | 分页 KV | MLA | FP8/NVFP4 KV | 投机/tree | 备注 |
|---|---|---|---|---|---|---|
| **FlashAttention 2/3/4** | NVIDIA 主力 prefill/decode | ✅ | FA4 支持 MLA prefill（0.20 默认） | ✅ | ✅ | FA4 在 SM90+ 上支持 head_dim 512 |
| **FlashInfer** | 低延迟 decode、FP8、Blackwell | ✅ | ✅（含 ragged prefill） | ✅ | ✅ | 0.9 起 Blackwell 首选，需匹配 wheel |
| **FlashMLA** | DeepSeek 系 decode | ✅ | ✅ | fp8_ds_mla | ✅ | SM90/SM100 专用，稀疏变体 |
| **Triton MLA / Cutlass MLA** | 兼容/回退 | ✅ | ✅ | 部分 | ✅ | 可移植性优先 |
| **Triton Attention** | 无 FA 的硬件/自定义 | ✅ | ❌ | 部分 | 部分 | 性能一般，胜在能改 |
| **FlexAttention** | 自定义 mask/稀疏 | ✅ | ❌ | 部分 | 部分 | 研究友好 |
| **TurboQuant** | 2-bit KV（0.20+） | ✅ | 部分 | ✅ | — | 4× 容量，精度需评估 |
| **Mamba/GDN/Linear/ShortConv** | 混合/线性模型 | 状态制 | — | — | — | 与 full attention 混层 |
| **CPU Attn** | 无 GPU 部署 | ✅ | ❌ | — | — | 吞吐极低，仅兜底 |

> 0.26 起支持**按 KV cache group 选不同 backend**：一个模型里 full attention 层用 FA、SWA 层用 FlashInfer、Mamba 层用 GDN——这是混合架构（Qwen3-Next、Kimi 系）能跑的前提。

#### 3.4.3 选择逻辑与覆盖方式

优先级大致是：**显式 env/参数 > 模型/平台推荐 > 自动探测 > 回退**。

```bash
# 强制指定（排障/对比时最有用）
export VLLM_ATTENTION_BACKEND=FLASHINFER      # 或 FLASH_ATTN / TRITON_ATTN / FLASHMLA ...
```

**排障口诀**：性能不对先看日志里实际选中的 backend（启动日志会打印 `Using ... attention backend`），不要假设它用了你以为的那个。

#### 3.4.4 自定义 backend 的接口形状（0.25.1 实测）

`vllm/v1/attention/backends/registry.py` 提供 `register_backend()`；每个 backend 需要：

```python
class MyBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size) -> tuple: ...
    @staticmethod
    def get_supported_head_sizes() -> list[int]: ...
    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]: ...
    @classmethod
    def supports_sliding_window(cls) -> bool: ...
```

配套的 `AttentionMetadataBuilder.build()` 每 step 从 `SchedulerOutput` 里取 block_table / slot_mapping / seq_lens，组装成 kernel 需要的描述符。**写 backend 的 80% 工作量在 metadata 与边界条件**（chunked prefill、投机多 query、Sliding Window），kernel 本身往往只有 20%。

### 3.5 CUDA Graph：把 launch 开销压到 0

#### 3.5.1 为什么 decode 必须上图

decode 每 step 只算 1 个 token，GPU 计算时间可能只有 **200–800 μs**，而一个 70B 模型每 step 要发 **数千个 kernel**，每个 kernel launch 3–10 μs，CPU 侧总开销轻易超过 GPU 计算时间 → **GPU 空等**。

CUDA Graph 把整段前向"录制"成一张图，之后一次 `graph.replay()` 发完所有 kernel，**launch 开销从 O(kernel 数) 降到 O(1)**。

#### 3.5.2 vLLM 的两种图模式

| 模式 | 捕获范围 | 适用 | 限制 |
|---|---|---|---|
| **PIECEWISE**（分段） | 把模型按 attention 边界切成若干段，每段一张图；attention 动态执行 | 形状变化大、MoE、动态 batch | 段间仍有 launch 开销 |
| **FULL**（全图） | 整个 decode step 一张图 | 静态形状、dense 模型 | batch 必须 pad 到捕获尺寸；控制流不能变 |
| **FULL_AND_PIECEWISE**（0.11 起默认） | 能全图就全图，否则分段 | 通用 | 捕获时间与显存更高 |

**捕获尺寸**：vLLM 会枚举一组 batch size（如 `[1,2,4,8,...,max_num_seqs]`），为每个尺寸捕获一张图；运行时把实际 batch **pad 到最近的捕获尺寸**。所以：

- `max_num_seqs` 越大、捕获列表越密，显存与启动时间越高；
- **padding 浪费**会让"实际 batch=1 但按 8 跑"，小并发下吞吐反而变差；
- 0.28 把 Blackwell 的捕获上限提到 **1024**，0.30 通过冻结 GC 把捕获时间从 12s 降到 2s（H200）。

#### 3.5.3 什么时候图会失效（线上高频）

1. **动态控制流**：模型里有 `if tensor.sum() > 0` 之类的 data-dependent 分支。
2. **变长 all-reduce**：TP 通信若按动态 shape 分配 buffer，捕获会失败（需 `--disable-custom-all-reduce` 或固定 buffer）。
3. **投机解码 + 动态 draft 长度**：多 token 验证的 query 长度变化（0.25 起才逐步支持全图）。
4. **LoRA 动态切换**、**多模态 encoder**（0.28 才支持 encoder CUDA graph）。
5. **自定义 kernel 未注册为 graph-safe**（内部有 host 同步、`cudaMemcpy`、动态分配）。
6. **结构化输出的 mask 在 GPU 侧动态生成**（0.14 才支持与 async 组合）。

**排查**：启动加 `VLLM_LOGGING_LEVEL=DEBUG`，搜索 `cudagraph` / `fallback`；或对比 `--enforce-eager` 与默认的吞吐差（差 2× 以上说明图没生效）。

#### 3.5.4 权衡表

| 配置 | 吞吐 | 启动时间 | 显存 | 适用 |
|---|---|---|---|---|
| `--enforce-eager` | 低（-20%~-50%） | 最快 | 最省 | 调试、形状极不规则 |
| PIECEWISE | 中 | 中 | 中 | MoE、动态模型 |
| FULL | 高 | 慢 | 高 | dense、固定负载 |
| FULL_AND_PIECEWISE | 高 | 慢 | 高 | 默认 |

### 3.6 torch.compile 与图级优化

#### 3.6.1 它在 vLLM 里到底做什么

不是"把整个模型丢给 Inductor"，而是**有边界的编译**：

1. **切分**：attention、通信、采样等算子被标记为**自定义 op**（`torch.ops.vllm.*`），编译器不越过它们。
2. **融合**：RMSNorm + residual、RoPE、SiLU×gate、量化 GEMM 的 epilogue 融合。
3. **消除 Python 开销**：把 `InputBatch` 的组装逻辑编译成静态图（这是 V1 启动快、step 稳的关键）。
4. **内存规划**：复用中间激活，降低峰值。

#### 3.6.2 编译等级与代价

```bash
vllm serve <model> -O3          # 最高优化（默认在 V1 开启）
vllm serve <model> -O0          # 关闭编译（排障用）
```

| 等级 | 行为 | 启动时间 | 风险 |
|---|---|---|---|
| O0 | 纯 eager | 快 | 慢 |
| O1/O2 | 选择性编译 | 中 | 中 |
| O3 | 全量 Dynamo 编译 + 融合 | 慢（首次可达数分钟） | 图断裂、编译失败回退 |

**工程经验**：

- **编译缓存要持久化**（`VLLM_CACHE_ROOT`），否则每次重启都重新编译，K8s 滚动发布会被 readiness 探针打死。
- **图断裂（graph break）**是性能杀手，用 `TORCH_LOGS=graph_breaks` 定位；常见原因：`.item()`、动态 shape、第三方库不支持。
- **0.20 起引入 vLLM IR**（先做 rms_norm 等算子），方向是"减少对 Inductor 的依赖、让自定义 kernel 可组合"——面试可以提"vLLM 正在从 torch.compile 使用者变成自己的编译器"。

### 3.7 采样与结构化输出

#### 3.7.1 采样流水线

```
logits (B, V)
  → logits processors（按请求）：repetition/frequency/presence penalty、min_tokens、
    logit_bias、bad_words、grammar bitmask、logprobs 计算
  → 温度 / top-k / top-p / min-p
  → 采样（greedy / multinomial / beam / rejection sampling for spec decode）
  → 输出 token + logprobs
```

**性能陷阱**：

- **词表大**（150K+）时，penalty 与 top-p 的 sort 是显存带宽瓶颈；vLLM 用 fused kernel + 按请求分组处理。
- **`logprobs=k`** 会显著增加开销（需要 top-k 排序 + 回传），RLHF 场景常见。
- **结构化输出 bitmask** 是 `(B, V)` 的 bool/uint32 位图，**每 step 都要 apply**；语法复杂时 mask 计算可能比模型前向还慢（CPU 侧 FSM 推进）。优化手段：xgrammar 编译成 GPU 友好的 mask、缓存 FSM 状态、jump-forward（确定的前缀直接跳过采样）。

#### 3.7.2 结构化输出（JSON Schema / 正则 / 语法）

- 后端：**xgrammar**（默认，编译式）、**guidance**（解释式）、outlines。
- 原理：把 schema 编译成 **FSM/下推自动机**，每步计算"允许的 token 集合"，转成 bitmask 屏蔽非法 logits。
- **保证**：输出 100% 合法（不是"概率上更可能合法"）。代价是**可能改变分布**（在合法集合内重新归一化）。
- **常见坑**：schema 过于复杂导致编译超时/内存爆炸；`anyOf` 嵌套；与 speculative decoding 组合时 draft 也必须合法（0.16 起支持）。

### 3.8 投机解码：收益、代价与选型

#### 3.8.1 基本模型

```
draft 阶段：小模型/额外头/检索，猜 γ 个 token（成本 ≈ γ × c_draft）
verify 阶段：大模型一次前向验证 γ+1 个 token（成本 ≈ 1 × c_target，与 γ 弱相关）
接受：从左到右匹配，接受 k 个则本 step 前进 k+1 个 token
```

**期望加速比**（忽略 draft 成本）：

```
speedup ≈ (1 - α^(γ+1)) / ((1 - α) × (1 + γ × c_draft/c_target))
α = 单 token 接受率
```

**结论**：

- α 高（>0.8）、γ 适中（3–5）、draft 便宜 → 加速明显（1.5–2.5×）。
- **高并发下收益消失**：因为 verify 让每 step 的 token 数变多、batch 变大，系统从"延迟受限"变成"吞吐受限"，而投机解码**增加了总计算量**。
- **显存换速度**：draft 模型/额外头要占显存，长上下文时 KV 也要额外一份。

#### 3.8.2 方法谱系

| 方法 | 原理 | 优点 | 缺点 | 适用 |
|---|---|---|---|---|
| **N-gram / Prompt lookup** | 从 prompt 里检索重复片段 | 零训练、零额外显存 | 只对"复制型"任务有效（RAG、代码补全） | 文档问答、续写 |
| **Suffix decoding** | 后缀自动机匹配历史 | 比 n-gram 更灵活 | 命中率依赖文本重复度 | 对话、模板化输出 |
| **Medusa** | 在 hidden state 上接多个解码头 | 单模型、无 draft 模型 | 头数固定、接受率中等 | 通用加速 |
| **EAGLE-1/2/3** | 用小自回归头在**特征层**预测，再验证 | 接受率高（0.7–0.9） | 需训练/对齐，显存 + | 主流生产方案 |
| **MTP**（DeepSeek 系） | 模型自带多 token 预测层 | 与训练一致、接受率高 | 依赖模型结构 | DeepSeek/Qwen 系 |
| **DFlash / DSpark** | 新一代 drafter（局部卷积 + 候选选择 / 置信度调度验证） | 长上下文与高并发下更稳 | 较新，生态少 | 2026 新部署 |
| **TLI**（0.25） | 异构词表 draft→target 映射 | 可用不同 tokenizer 的小模型 | 映射损失 | 跨模型族加速 |

#### 3.8.3 调参与"负收益"判断

```bash
--speculative-config '{"method":"eagle3","model":"<draft>","num_speculative_tokens":3}'
--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'
```

**判断是否该开**：

1. 先测 **α**（日志/指标里有 acceptance rate），α < 0.6 基本不值得。
2. 看**负载形态**：低并发（batch ≤ 8）、长输出、有重复文本 → 值得；高并发吞吐型 → 通常不值得。
3. 看**显存余量**：draft 模型 + 额外 KV 可能挤掉并发。
4. 看**是否与其它特性冲突**：结构化输出、异步调度、PP 的组合要逐个验证（0.14–0.28 逐步放开）。

**动态投机**（0.25+）：根据在线接受率**自动调整 γ**（0.28 `adaptive speculative token budget`，0.30 `online acceptance estimator`）。这是 2026 年的默认姿势：**不要手调 γ，让系统自己收敛**。

### 3.9 并行策略：TP / PP / DP / EP / CP

#### 3.9.1 五种并行的本质

| 并行 | 切什么 | 通信 | 适用 | 主要瓶颈 |
|---|---|---|---|---|
| **TP** | 层内权重（头/维度） | 每层 2 次 all-reduce | 单机多卡、模型放不下 | 通信延迟、小消息效率 |
| **PP** | 层（stage） | 只在 stage 边界传激活 | 多机、超大模型 | 流水气泡、负载不均 |
| **DP** | 请求（副本） | 几乎无（除 MoE） | 提升吞吐 | 显存复制、负载均衡 |
| **EP** | 专家（MoE） | all-to-all（每层） | MoE 大模型 | 通信量、专家负载不均 |
| **CP/DCP/PCP** | 序列长度 | 注意力环状/分块通信 | 超长上下文 | 通信 + 实现复杂度 |

#### 3.9.2 通信量速算（面试常考）

**TP all-reduce**（每层两次，前向）：

```
单次消息量 = B × S × H × dtype_bytes
每层两次 → 2 × B×S×H×2B（BF16）= 4·B·S·H 字节
L 层总计 ≈ 4·L·B·S·H 字节
```

以 70B（L=80, H=8192）、B=32、S=1（decode）、BF16 为例：`4×80×32×1×8192 ≈ 84 MB/step`。在 NVLink（~300–900 GB/s）上约 0.1–0.3 ms，可接受；在 PCIe（~25 GB/s/卡）上要 3+ ms，**直接吃掉 decode 预算**。

> **这就是"TP 只适合单机 NVLink"的定量原因**，也是 vLLM 提供 **custom all-reduce**（0.4 起）的原因：小消息用 NCCL 效率低，自研 kernel 可省 20–40%。

**PP**：stage 间只传 `B×S×H`，量小，但**气泡**是主要成本。1F1B 调度下气泡比例 ≈ `(P-1)/(M+P-1)`（M = micro-batch 数）。异步调度 + PP（0.16）通过重叠调度与执行把气泡进一步压小。

**EP**：all-to-all 的量与"每 token 选中的专家数 × hidden"成正比，DeepSeek 类模型每层 2 次 all-to-all，是**跨机部署的主要障碍**，因此有 DeepEP（专用 all-to-all kernel）、EPLB（专家重平衡）、Elastic EP（动态扩缩）。

#### 3.9.3 选型决策树

```
模型能放进单卡？
├─ 能 → 单卡 DP（多副本）最优；若单请求延迟不达标 → TP=2/4 试
└─ 不能
   ├─ 单机 8 卡能放下？
   │   ├─ 能 → TP=8（NVLink）优先；MoE → TP + EP 混合
   │   └─ 不能 → PP=跨机 + TP=机内
   └─ MoE 且专家极多 → EP（跨机需 DeepEP/InfiniBand）+ EPLB
长上下文（>128K）解码慢 → 追加 DCP（decode context parallel）
超长 prefill（>256K）→ PCP（prefill context parallel）或 PD 分离
```

**经验配置表**：

| 模型规模 | 单机 8×H100/H200 | 多机 |
|---|---|---|
| 7B–14B | TP=1/2 + 多副本 | 不需要 |
| 32B–72B | TP=4/8 | PP=2×TP=8（若跨机） |
| 100B+ dense | TP=8 | PP=2–4 + TP=8 |
| MoE（DeepSeek 类） | TP=8 + EP=8（机内） | EP=16/32 + DeepEP + EPLB |
| 长上下文推理 | TP=8 + DCP=2/4 | PD 分离 + PCP |

#### 3.9.4 并行带来的"非性能"问题

- **确定性**：TP/EP 的归约顺序不同 → 同 prompt 不同输出（greedy 也可能变）。需要可复现时用 `--enforce-eager` + batch invariant 模式（0.22 起有 batch-invariant 优化）。
- **负载不均**：EP 场景某些专家热、某些冷 → EPLB 动态迁移专家副本；MoE 路由的 token 分布漂移会导致"同配置今天快明天慢"。
- **拓扑敏感**：跨 NUMA、跨 PCIe switch、跨 leaf-spine 的性能差异可达 30%+。部署前必须跑一次 `all_reduce_perf` / `alltoall_perf` 基线。

---

> 上一篇：[03a-核心机制-KV与调度](03a-核心机制-KV与调度.md) ｜ [返回总览](00-总览.md) ｜ 下一篇：[03c-核心机制-量化与服务化](03c-核心机制-量化与服务化.md)


# 第 3 章 核心机制（下）：量化、分离式服务与启动优化

> [← 总览](00-总览.md) ｜ 下一章：[04-性能与调参](04-性能与调参.md)

### 3.10 量化：从权重到 KV Cache 的全栈低精度

#### 3.10.1 先分清三种量化

| 类型 | 量化对象 | 收益 | 风险 | 典型方案 |
|---|---|---|---|---|
| **权重-only（W4A16/W8A16）** | 模型权重 | 显存 ↓、decode 带宽 ↓（decode 是带宽瓶颈） | 精度损失集中在 outlier 通道 | AWQ、GPTQ、Marlin、GGUF |
| **权重+激活（W8A8/FP8）** | 权重 + 激活 | prefill 算力利用率 ↑（用低精度 TensorCore） | 激活 outlier 更难处理 | FP8 per-tensor/per-block、SmoothQuant、MXFP8 |
| **KV Cache 量化** | K/V | **直接决定并发上限**（KV 占大头） | 长上下文精度衰减、注意力分数失真 | FP8 KV、NVFP4 KV、TurboQuant 2-bit |

> **一句话**：decode 阶段 GPU 在"搬权重 + 搬 KV"，所以**权重量化与 KV 量化直接换算成吞吐**；prefill 阶段在"算 FLOPs"，所以**激活量化才提 prefill**。这两个结论能解释 90% 的"量化后为什么没变快"。

#### 3.10.2 格式与硬件对应（2026 口径）

| 格式 | 位宽/块 | 硬件 | 典型吞吐收益 | 备注 |
|---|---|---|---|---|
| FP8 E4M3（per-tensor / per-block） | 8 bit | SM89+（H100/L40S）、SM100 | prefill 1.5–2×，decode 1.3–1.6× | 生态最成熟 |
| MXFP8 | 8 bit / 32 元素共享指数 | SM100+ | 接近 FP8 | OCP 标准，跨厂商 |
| **NVFP4** | 4 bit / 16 元素 + FP8 scale | SM100/SM120 | decode 最高 ~2× | 0.24 起 MoE、0.32 起 KV |
| INT8 W8A8 | 8 bit | 广泛 | 1.2–1.5× | 需要校准 |
| INT4 W4A16（AWQ/GPTQ/Marlin） | 4 bit group=128 | 广泛 | decode 1.5–2×（低并发） | 最普及的"省显存"方案 |
| TurboQuant | 2 bit KV | SM90+ | KV 容量 ×4 | 精度需逐模型验证 |
| GGUF（Q4_K 等） | 混合 | CPU/GPU | 部署方便 | 性能不如原生 kernel |

#### 3.10.3 KV Cache 量化的"账"与"坑"

**账**：KV 从 BF16 → FP8，容量 ×2、带宽 ÷2，**并发直接翻倍**。对一个 32K 上下文的重度 RAG 服务，这往往比换卡更划算。

**坑**：

1. **scale 的粒度**：per-tensor 最省事但精度差；per-head/per-token 更好但要额外 kernel 支持。FP8 KV 在长上下文上容易出现"注意力弥散"（回答变泛）。
2. **K 与 V 的敏感度不同**：K 影响 softmax 分数（对 outlier 敏感），V 影响加权和（相对鲁棒）。实践里常见"K 用更高精度、V 用低精度"的混合方案。
3. **校准数据**：静态 scale 需要与线上分布匹配；用通用语料校准、上线跑代码/数学，会翻车。
4. **与投机解码/结构化输出的交互**：低精度 KV 让 verify 的 logits 偏移，接受率下降。
5. **精度验证必须是任务级**：perplexity 变化 <1% 不代表任务不退化。要跑 **长上下文 needle、代码单测通过率、JSON 合法率、数学准确率**。

#### 3.10.4 "同一种量化格式，不同 kernel"的血泪教训

真实事故模式（国产卡适配常见）：checkpoint 标称 AWQ，但内部打包是 **GEMM 布局**还是 **WNA16 布局**决定了必须用哪个 kernel。选错的后果不是报错，而是**输出退化成 `!!!!!` 之类的乱码**——因为 shape 对得上、dtype 对得上，只是权重被按错误的 packing 解读。

**防御性工程**（可直接落地）：

```python
# 1) 加载后做"金标准"自检：固定 prompt + greedy + 前 32 token 逐字比对
GOLDEN = [("1+1=", "2"), ("中国的首都是", "北京")]
for prompt, expect in GOLDEN:
    out = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=8))
    assert expect in out, f"quant kernel mismatch? got={out!r}"

# 2) 记录实际命中的量化方法（不要相信你的配置，相信运行时）
print(getattr(model.quant_config, "get_name", lambda: "?")())

# 3) 断言权重布局：读 checkpoint 元数据里的 quantization_config 字段
#    （group_size / zero_point / version / checkpoint_format）与 kernel 注册表比对
```

**原则**：**量化路径必须有"可观测的正确性断言"**。没有金标准自检的量化上线，等于在赌。

#### 3.10.5 在线量化（0.20+）

`--quantization fp8` 之类可以在**加载时**把 BF16 权重转成低精度，省去离线转换步骤。适合：

- 拿不到官方量化版；
- 想在 A/B 中快速验证低精度收益；
- 显存紧张但不想改 checkpoint。

代价：启动变慢（要做 scale 计算）、精度取决于运行时校准策略（`quantization_config.targets` 可只量化部分层）。

### 3.11 分离式服务（P/D Disaggregation）

#### 3.11.1 为什么要拆

| 阶段 | 瓶颈 | 最优并行 | 对 batch 的偏好 |
|---|---|---|---|
| Prefill | **算力**（FLOPs） | TP 大、PP 少 | 大 batch、chunk 大 |
| Decode | **显存带宽** | TP 适中、EP（MoE） | 大 batch 摊薄权重读取 |

把两者放同一实例，会互相拖累：长 prefill 抢占 decode 的 SLO；decode 又让 prefill 的算力利用率上不去。**拆开之后可以分别扩缩容**，并且 prefill 实例可以把 KV 直接传给 decode 实例。

#### 3.11.2 架构与数据流

```
Client → Router（前缀感知 / 负载感知）
            ├─> Prefill 实例（P）：算 KV，写入本地块
            │        └─ KV Connector ──(RDMA / NVLink / CXL / TCP)──┐
            └─> Decode 实例（D）：从远端拉 KV，继续生成              │
                     └─ KV Connector <──────────────────────────────┘
```

- **连接器生态**：LMCache（缓存/复用）、Mooncake（KVCache 中心化存储）、NIXL（NVIDIA 推理传输库）、FlexKV、HiSparse（0.30）。
- **路由**：`vllm router`、llm-d、Dynamo、AIBrix、Kthena 等；关键是**前缀感知路由**（把同一前缀的请求打到同一 D 实例，命中本地缓存）。
- **0.18+**：`vllm launch render` 把多模态预处理也拆出去，形成 **P / D / Render 三分离**。

#### 3.11.3 成本模型（决定"该不该拆"）

KV 传输量 = `prompt_tokens × per_token_bytes`。

以 70B（GQA，~320 KB/token，BF16，TP=8 后每卡 40 KB）为例，8K prompt：

```
总 KV ≈ 8000 × 320 KB ≈ 2.5 GB
100 Gbps 网卡（12.5 GB/s）→ 0.2 s   ← 直接加到 TTFT
400 Gbps（50 GB/s）→ 50 ms
NVLink（~450 GB/s）→ 5.6 ms
```

**结论**：

- **短 prompt（<2K）不值得拆**：传输开销占比太高。
- **长 prompt（>8K）+ 短输出**：拆分收益最大（TTFT 由 P 决定，D 只做 decode）。
- **必须有 layer-wise 流水传输**（算完一层传一层），否则串行传输会把收益吃光。
- **必须有失败恢复**（0.25 起有 KV Load Failure Recovery 测试）：远端 KV 拉取失败要能回退到本地重算。

#### 3.11.4 拓扑选择

| 拓扑 | 说明 | 适用 |
|---|---|---|
| 1P1D | 最简单 | 验证阶段 |
| xPyD | P、D 独立扩缩容 | 生产主流 |
| P-EP / D-TP | P 用 EP 摊算力，D 用 TP 降延迟 | MoE 大模型 |
| 同机 P/D | 共享 NVLink，零网络 | 单机内解耦 |
| 跨机 P/D | RDMA | 集群级 |

### 3.12 KV Offloading / HiSparse：把"冷 KV"请出显存

#### 3.12.1 分层缓存

```
GPU HBM（热，纳秒级）
  └─ CPU DRAM / pinned（温，~10–50 μs）
       └─ 本地 NVMe（冷，~100 μs–1 ms）
            └─ 远端对象存储（Mooncake Store / S3，~ms–10 ms）
```

**动机**：前缀缓存的命中率受限于 **KV 显存容量**。把"长期不变但访问稀疏"的前缀（如企业知识库的固定文档、长 system prompt）下沉到 CPU/磁盘，可以让 GPU 只保留热块，**等效扩大缓存容量**。

**收益场景**：多轮长对话、固定知识库 RAG、多租户共享模板。

**代价**：换入换出的 PCIe/NVMe 带宽；调度器要感知"块在哪一层"（tier-aware scheduling）；失败恢复复杂。

**关键实现点（0.18–0.30 演进）**：

- 只 offload **高频复用**的块（0.18 "smart CPU offloading"），避免无效搬运。
- **多级（multi-tier）**框架 + 可插拔二级存储（0.22、0.28 `module_path`）。
- 与 **DP 副本**协同（0.26 DP-replica-aware tiering）。
- 指标：offload 命中率、换入延迟、PCIe 利用率（0.26 起有专门 metrics）。

#### 3.12.2 HiSparse（0.30）

针对**稀疏 MLA decode**（DeepSeek 系）：GPU 显存紧张时，把 KV 页"溢出"到 pinned host memory，decode 时按 top-k 需要从 host 取；同时维护一个 **per-request 的 GPU 热缓冲**承接 miss。

**本质**：用"稀疏访问 + 主机内存"换显存，属于**软降级**方案——当显存不足以支撑目标上下文时，避免直接 OOM，而是用延迟换容量。

### 3.13 启动优化与权重加载

#### 3.13.1 冷启动时间都花在哪

| 阶段 | 典型耗时（70B，经验值） | 优化手段 |
|---|---|---|
| import + CUDA context | 5–20 s | 预编译镜像、预热 |
| 权重加载（本地 NVMe） | 30–120 s | mmap + 多线程、fastsafetensors、分片并行 |
| 权重加载（网络/对象存储） | 分钟级 | runai streamer、tensorizer、本地缓存 |
| 在线量化 | 10–60 s | 预量化 checkpoint |
| torch.compile | 30 s–5 min | 持久化编译缓存、O2 降级 |
| CUDA Graph 捕获 | 10–60 s | 减少捕获尺寸、0.30 GC 冻结（12s→2s） |
| FlashInfer autotune | 10–30 s | `vllm preload --preload-autotune`（0.32） |
| warmup / profile_run | 5–20 s | 减少 dummy batch 规模 |

#### 3.13.2 Fast Start（0.30）

**思路**：启动一个**常驻的 per-GPU 权重缓存守护进程**，把"已经量化、已经按 TP 切好"的权重留在 GPU/主机内存里；重启引擎时通过 **CUDA IPC** 直接映射（`--load-format ipc_cache`），不再从磁盘读。

**收益**：重启从分钟级降到秒级。对**弹性扩缩容、滚动发布、RLHF 频繁换权重**是质变。

#### 3.13.3 加载格式速查

| `--load-format` | 说明 | 适用 |
|---|---|---|
| `auto` | 自动判断（safetensors/pytorch） | 默认 |
| `safetensors` | 最稳 | 本地盘 |
| `runai_streamer` | 流式读取，支持 S3/GCS | 对象存储 |
| `tensorizer` | 序列化加速 | 自建 |
| `fastsafetensors` | 多线程/GPU 直读 | 大模型提速 |
| `ipc_cache` | Fast Start | 频繁重启 |
| `dummy` | 随机权重，不加载 | 性能压测/图捕获验证 |
| `bitsandbytes` | 0.28 起转为插件 | 旧方案 |

---

> 上一篇：[03b-核心机制-执行与并行](03b-核心机制-执行与并行.md) ｜ [返回总览](00-总览.md) ｜ 下一篇：[04-性能与调参](04-性能与调参.md)


# 第 4 章 性能模型、容量规划与调参

> [← 总览](00-总览.md) ｜ 下一章：[05-故障诊断](05-故障诊断.md)

### 4.1 指标体系：先把"快"定义清楚

| 指标 | 定义 | 谁在乎 | 常见误区 |
|---|---|---|---|
| **TTFT** | 首 token 时间 = 排队 + prefill + 首采样 | 用户感知、RAG | 只看均值不看 p99 |
| **TPOT** | 每输出 token 时间（decode 稳态） | 长输出、reasoning | 把 ITL 当 TPOT |
| **ITL** | 相邻 token 间隔（含被抢占/插队） | 流式体验 | 忽略 prefill 插队导致的尖刺 |
| **E2E** | 端到端 = TTFT + TPOT×(out_len-1) | SLA 定级 | 用平均长度算容量 |
| **吞吐** | req/s 或 output tok/s | 成本 | 混用 input/output token |
| **Goodput** | 满足 SLO 的吞吐 | 生产核心 | 只报峰值吞吐 |
| **MFU/MBU** | 算力/带宽利用率 | 性能工程师 | 脱离 workload 谈 |

**两个关键关系**：

```
并发数 ≈ QPS × 平均延迟（Little's Law）
吞吐 ≤ 并发数 / 平均每请求耗时
```

**压测口径必须写清**：闭循环（固定并发，client 等响应再发下一条）还是开循环（固定 QPS / Poisson 到达）。**闭循环会系统性低估排队延迟**，线上故障往往来自开循环下的雪崩。推荐 `vllm bench serve --request-rate <qps>` 做开循环，`--max-concurrency` 做闭循环，两者都跑。

### 4.2 容量规划三步法

#### Step 1：算显存与 KV 预算

```
可用 KV 显存 ≈ 总显存 × 利用率 − 权重 − 激活峰值 − 图/通信/碎片预留
每 token KV = 2 × L × H_kv × D × dtype（MLA 见 §3.1.2）
最大在途 token ≈ 可用 KV 显存 / 每 token KV
```

#### Step 2：由业务形态定并发

```
目标并发会话数 × 平均上下文长度 ≤ 最大在途 token × 0.8（留余量）
```

#### Step 3：由 roofline 估吞吐上限

**decode（带宽受限）**：

```
单 step 时间 ≈ (权重字节 + 本 step 读的 KV 字节) / 有效带宽
稳态 output tok/s ≈ batch_size / 单 step 时间
```

例：70B W4（35 GB 权重）在 3 TB/s HBM 上：

```
batch=1：35 GB / 3 TB/s ≈ 11.7 ms/token → 85 tok/s（单请求）
batch=32：仍是 ~11.7 ms/step（权重只读一次）→ 32/0.0117 ≈ 2700 tok/s
```

**prefill（算力受限）**：

```
prefill FLOPs ≈ 2 × 参数量 × prompt_tokens
TTFT ≈ FLOPs / (峰值算力 × MFU) + 排队
```

这两条式子能快速判断"该加卡还是该量化"：

- 若 decode 带宽利用率 > 70% → 量化权重/KV 或加卡；
- 若 prefill MFU < 30% → 调大 chunk、换 FA 后端、检查 TP 通信。

### 4.3 参数调优手册（0.25–0.34 口径）

> 默认值随版本变化，**以 `vllm serve --help` 与你实际版本为准**；下表给语义与调法。

| 参数 | 作用 | 调大 | 调小 | 典型值 | 坑 |
|---|---|---|---|---|---|
| `--gpu-memory-utilization` | KV 预算比例（0.34 起弱化） | 更多 KV/并发 | 防 OOM | 0.85–0.92 | 与激活峰值打架；0.34 建议不设 |
| `--max-model-len` | 最大上下文 | 支持长文 | 省 KV、防 OOM | 按业务 p99 设 | 设太大直接 OOM 起不来；0.14+ 可 `auto` |
| `--max-num-seqs` | 单 step 最大序列数 | 吞吐 | 延迟 | 64–512 | 过大→抢占、ITL 抖动 |
| `--max-num-batched-tokens` | 单 step token 预算 | prefill 吞吐 | TTFT/ITL | 2048–32768 | 0.28 默认 16384 |
| `--block-size` | KV 块大小 | 元数据少、kernel 友好 | 尾部浪费少 | 16 / 32 / 64 | 0.34 默认 64；改它会影响 prefix 命中粒度 |
| `--kv-cache-dtype` | KV 精度 | — | 容量/带宽 | `auto`/`fp8`/`nvfp4` | 精度回归必须任务级验证 |
| `--enable-prefix-caching` | 前缀缓存 | 多轮/RAG 命中 | 省哈希开销 | 开 | V1 默认开；多租户注意隔离 |
| `--enable-chunked-prefill` | 分块 prefill | 长文不阻塞 decode | 短 prompt 无需 | 开 | 与 spec decode/结构化输出交互要测 |
| `--async-scheduling` | 调度/执行重叠 | CPU 瓶颈时 | 兼容性优先 | 开（0.14+ 默认） | PP/部分 spec 组合不支持 |
| `-O3` / `--compilation-config` | torch.compile | 吞吐 | 启动时间 | O3 | 图断裂会白编译 |
| `--enforce-eager` | 禁用图 | 排障 | 性能 | 默认关 | 关图后吞吐掉 20–50% |
| `--tensor-parallel-size` | TP | 单卡放不下 | 通信开销 | 2/4/8 | 必须整除 attention head 数 |
| `--pipeline-parallel-size` | PP | 跨机大模型 | 气泡 | 2/4 | 需要足够 micro-batch |
| `--data-parallel-size` | DP | 吞吐 | 显存 | 多副本 | MoE 时与 EP 组合 |
| `--enable-expert-parallel` | EP | MoE 显存/算力 | 通信 | MoE 必开 | 跨机需 DeepEP/IB |
| `--speculative-config` | 投机解码 | 低并发延迟 | 高并发吞吐 | 视 α 而定 | α<0.6 别开 |
| `--quantization` | 量化方法 | 省显存 | 精度 | fp8/awq/gptq | 在线量化启动慢 |
| `--load-format` | 权重加载 | 启动速度 | — | auto/runai/ipc_cache | 网络盘必配流式 |
| `--disable-custom-all-reduce` | 关闭自研 allreduce | 排障 | 小消息性能 | 默认开自研 | 拓扑异常时先关它验证 |
| `--api-server-count` | API 进程数 | 短请求 QPS | 内存 | 2–4 | 0.32 起做 least-loaded |
| `--enable-lora` / `--max-loras` | 多 LoRA | 多租户定制 | 显存/调度复杂度 | 按需 | LoRA 会影响前缀缓存哈希 |
| `--swap-space` | V0 的 CPU swap | — | — | V1 已无 | 旧文档残留，别照抄 |
| `--enable-scale-out` | 暴露横向扩展端点 | 分布式路由 | 安全面 | 按需 | 0.30 起需显式开启 |

**调参的优先级顺序**（省时间）：

```
1. 先把 max_model_len / KV dtype 定下来（决定容量天花板）
2. 再定 max_num_seqs 与 max_num_batched_tokens（决定延迟/吞吐平衡）
3. 然后开 prefix caching / chunked prefill（决定真实负载效率）
4. 再看是否需要 spec decode / 量化 / 并行调整
5. 最后才是 CUDA Graph / compile / 前端进程数（决定常数开销）
```

### 4.4 场景配方（8 个高频场景）

#### 场景 A：高并发短请求（分类/抽取/客服首响）

- **目标**：QPS 最大、TTFT p99 < 300 ms、输出 < 64 token。
- **形态**：7B–14B 小模型，多副本 DP。

```bash
vllm serve Qwen3-8B \
  --max-model-len 8192 --max-num-seqs 512 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching --async-scheduling \
  --api-server-count 4 --disable-log-requests
```

- **代码级动作**：
  1. 把 **system prompt 稳定性**当一等公民（模板里禁止时间戳/随机 ID），命中率可从 20% → 80%+。
  2. 关闭 `logprobs`（除非业务需要），否则每步多一次 top-k 排序。
  3. 若用 K8s，配 **prefix-aware 路由**（同前缀请求打到同副本）。
- **验证**：`vllm bench serve --request-rate 200 --num-prompts 5000`，看 goodput 与 p99 TTFT。

#### 场景 B：长文档 RAG（长 prompt、短输出）

- **目标**：TTFT 可控、KV 不爆、命中率高。
- **关键**：chunked prefill + prefix cache + KV 低精度 +（可选）PD 分离。

```bash
vllm serve Qwen3-32B --tensor-parallel-size 4 \
  --max-model-len 65536 --max-num-batched-tokens 16384 --max-num-seqs 64 \
  --kv-cache-dtype fp8 --enable-prefix-caching --enable-chunked-prefill
```

- **代码级动作**：
  1. 文档分块时**把稳定前缀（指令+文档头）放在最前**，变化部分放最后，最大化前缀命中。
  2. 若同一文档被反复问答，考虑 **KV offload 到 CPU**（§3.12），把文档 KV 常驻。
  3. 超过 64K 时上 **PD 分离**：P 用大 chunk 快速算 KV，D 用 TP 做 decode。
- **验证**：TTFT 分布（p50/p99）、prefix 命中率、`num_preemptions`。

#### 场景 C：长输出推理（R1 类，输出 2K–32K token）

- **目标**：TPOT 稳定、不被抢占打断、输出不截断。
- **关键**：足够 KV、关掉高并发、谨慎开投机。

```bash
vllm serve DeepSeek-V4-Flash --tensor-parallel-size 8 --enable-expert-parallel \
  --max-model-len 131072 --kv-cache-dtype fp8 \
  --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --async-scheduling
```

- **代码级动作**：
  1. **不要把 max_num_seqs 开大**：长输出请求会互相抢占，导致"生成到一半被丢弃重算"（用户看到卡顿）。
  2. 监控 **preemption**：长输出场景任何抢占都是灾难。
  3. 若单请求 TPOT 不达标，上 **DCP**（decode context parallel）或提高 TP。
- **验证**：TPOT p99、抢占次数、输出完成率。

#### 场景 D：多轮对话 / Agent（tool calling）

- **目标**：多轮命中、工具调用解析正确、流式低延迟。
- **关键**：prefix cache + 会话亲和 + 流式 parser。

```bash
vllm serve Qwen3-32B --max-model-len 32768 \
  --enable-prefix-caching --enable-auto-tool-choice \
  --tool-call-parser hermes --reasoning-parser qwen3
```

- **代码级动作**：
  1. **会话粘性路由**：同一 session 打到同一副本，命中率从"跨副本随机"提升到 90%+。
  2. 用 0.24+ 的 **Streaming Parser Engine**（统一 tool/reasoning 解析），避免自写正则。
  3. 长对话要**截断策略**（保留 system + 最近 N 轮 + 摘要），否则 KV 线性增长。
- **验证**：多轮命中率、工具调用 JSON 合法率、p99 TTFT。

#### 场景 E：代码补全（超低延迟、短输出）

- **目标**：ITL < 30 ms、首 token < 100 ms。
- **关键**：小模型 + n-gram 投机 + 小 batch + TP=1。

```bash
vllm serve Qwen3-Coder-7B --max-model-len 8192 --max-num-seqs 32 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'
```

- **代码级动作**：
  1. 代码补全的**接受率天然高**（大量重复/模板代码），n-gram 投机性价比最高。
  2. 关闭不必要的 `logprobs`；用 **SSE 立即 flush**。
  3. 考虑 **FIM（fill-in-middle）** 模板与 KV 复用（同一文件前缀复用）。
- **验证**：ITL p99、接受率、单请求 tok/s。

#### 场景 F：MoE 大模型（DeepSeek / Qwen-MoE）

- **目标**：显存放下、专家负载均衡、跨机通信可控。
- **关键**：EP + EPLB + MLA 后端 + FP8 KV。

```bash
vllm serve DeepSeek-V4 --tensor-parallel-size 8 --enable-expert-parallel \
  --enable-eplb --kv-cache-dtype fp8 --trust-remote-code
```

- **代码级动作**：
  1. 确认 **MLA 走 FlashMLA/FlashInfer-MLA**（看启动日志），否则 decode 会慢 2×。
  2. 打开 **EPLB**（专家并行负载均衡），否则热专家会成为长尾。
  3. 跨机 EP 需要 **DeepEP**；没有 IB/RoCE 时不要跨机 EP。
- **验证**：每卡专家负载方差、all-to-all 耗时、TPOT p99。

#### 场景 G：多模态（VLM）

- **目标**：图像编码不拖累文本 decode、显存可控。
- **关键**：encoder cache、图像 token 裁剪、render 分离。

```bash
vllm serve Qwen3-VL-32B --max-model-len 32768 \
  --limit-mm-per-prompt '{"image":4}' --mm-processor-kwargs '{"max_pixels":1003520}'
```

- **代码级动作**：
  1. **限制每请求图片数与分辨率**（`max_pixels`），否则单张 4K 图就能吃满激活。
  2. 用 **encoder cache**（同图复用）与 **vision DP**（0.11+）。
  3. 高并发图片预处理用 **`vllm launch render`**（0.18）拆到 CPU 集群。
- **验证**：图像编码耗时占比、显存峰值、首 token 延迟。

#### 场景 H：离线批处理（数据合成 / 评测 / 蒸馏）

- **目标**：纯吞吐最大化，延迟无所谓。
- **关键**：大 batch、大 chunk、忽略 TTFT。

```bash
vllm serve Qwen3-32B --max-num-seqs 1024 --max-num-batched-tokens 32768 \
  --enable-prefix-caching --gpu-memory-utilization 0.95
```

- **代码级动作**：
  1. 用 **offline `LLM` API + 批量提交**，而不是 HTTP 逐条打。
  2. 排序：**按长度分桶**提交，减少 padding 与抢占。
  3. 关掉流式、关掉 logprobs（除非需要）。
- **验证**：tokens/s、GPU 利用率、抢占次数（应接近 0，否则说明 KV 不足）。

---

> 上一篇：[03c-核心机制-量化与服务化](03c-核心机制-量化与服务化.md) ｜ [返回总览](00-总览.md) ｜ 下一篇：[05-故障诊断](05-故障诊断.md)


# 第 5 章 线上问题诊断手册

> [← 总览](00-总览.md) ｜ 下一章：[06-改代码实战](06-改代码实战.md)

### 5.1 五分钟定位流程

```
① 现象是什么？
   ├─ 起不来（启动失败/超时） ──────────────→ A 组
   ├─ 能起但慢（TTFT/TPOT/吞吐不达标）────→ B 组
   ├─ 能起但错（乱码/截断/重复/精度）────→ C 组
   └─ 跑着跑着坏（越来越慢/超时/断流）────→ D 组
② 先看三个指标（有 Prometheus 就看板，没有就日志）
   KV 使用率 / 抢占次数 / 前缀命中率
③ 再对比配置：与"上次正常版本"diff（镜像、模型、参数、拓扑）
④ 最后才怀疑模型与 kernel
```

**核心判据表**：

| 观察 | 指向 |
|---|---|
| KV 使用率长期 ~100% + 抢占增长 | KV 不足（max_model_len/max_num_seqs/量化） |
| KV 使用率低 + GPU 利用率低 | CPU 瓶颈（调度/图/前端）或 batch 太小 |
| TTFT 高但 TPOT 正常 | 排队 / prefill / tokenize / 前缀未命中 |
| TTFT 正常但 TPOT 抖动 | 抢占、chunk 插队、异步调度假设错误 |
| GPU 利用率高但吞吐低 | kernel 选错 / 精度过高 / 通信等待 |
| 输出乱码 | 量化 kernel 不匹配 / 模板错误 / 前缀串味 |

### 5.2 速查表（20 条）

| # | 现象 | 最可能根因 | 30 秒验证 | 修复 |
|---|---|---|---|---|
| 1 | 启动 OOM | `max_model_len` 过大 / 激活峰值 / 图捕获 | 降 `--max-model-len` 重试 | 调小、量化、`--max-model-len auto` |
| 2 | 启动卡住无日志 | NCCL 建链 / 权重下载 / 编译 | `py-spy dump` 看栈 | 检查网络/IB、预下载、缓存编译 |
| 3 | CUDA Graph 捕获失败 | 动态形状/自定义 kernel | 日志搜 `cudagraph`、加 `--enforce-eager` | 关图定位、修 kernel/固定形状 |
| 4 | 吞吐只有预期一半 | 图没生效 / backend 选错 | 对比 `--enforce-eager`、看日志 backend | 修图捕获、指定 backend |
| 5 | TTFT 高 | 排队 + prefill 长 | 看 queue time | 加副本、chunk、PD 分离 |
| 6 | ITL 尖刺 | 长 prefill 插队 / 抢占 | 看 preemption 与 chunk | 调 `max_num_batched_tokens`、保 decode |
| 7 | 抢占风暴 | KV 不足 + 并发过大 | `num_preemptions_total` 斜率 | 降 `max_num_seqs`、量化 KV、加卡 |
| 8 | 输出乱码/`!!!!!` | 量化 kernel 与 checkpoint 布局不匹配 | 金标准 prompt 比对 | 换对应 kernel、加自检 |
| 9 | 回答"串味" | 前缀缓存跨租户共享 | 关 prefix caching 复测 | `cache_salt`、租户隔离 |
| 10 | 精度下降 | FP8/NVFP4 KV 或权重量化 | 切回 BF16 对比 | 任务级评估、混合精度 |
| 11 | 多模态 OOM | 图片分辨率/数量过高 | 限制 `max_pixels` | 裁剪、encoder cache |
| 12 | 结构化输出卡死 | 语法过复杂 / FSM 状态爆炸 | 简化 schema | 拆 schema、换后端、设超时 |
| 13 | 投机解码变慢 | α 低 / 高并发 | 看接受率 | 关掉或换 n-gram |
| 14 | NCCL 超时 | 拓扑/网卡/防火墙 | `all_reduce_perf` | 修网络、`NCCL_DEBUG=INFO` |
| 15 | 多卡结果不一致 | 归约顺序/非确定性 kernel | `VLLM_BATCH_INVARIANT=1` | 确定性模式、接受差异 |
| 16 | 越跑越慢 | 前缀缓存膨胀 / 多模态缓存泄漏 | 重启对比 | 限缓存、`reset_prefix_cache` |
| 17 | SSE 断流 | 代理缓冲 / 超时 | curl 直连对比 | 关 proxy buffering、调超时 |
| 18 | API 5xx/超时 | 前端进程不足 / 序列化慢 | `--api-server-count` 加压 | 多 worker、Rust 前端 |
| 19 | 升级后性能回退 | 默认值变化（block_size/async/chunk） | diff 默认参数 | 显式固定参数 |
| 20 | 关停很慢 | 请求排空 / 图析构 | 看 shutdown 日志 | 优雅退出超时、SIGTERM 处理 |

### 5.3 重点故障详解

#### 5.3.1 启动 OOM：为什么"算得好好的"还会炸

启动阶段的显存组成：

```
权重（含量化）
+ 激活峰值（profile_run 用 dummy batch 估）
+ CUDA Graph 捕获（每个 batch size 一份，常被低估）
+ NCCL/通信 buffer
+ CUDA context + 碎片
+ KV Cache（剩下的才是它）
```

**高频原因**：

1. `--max-model-len` 按模型上限设（如 128K），但 KV 按最大长度预留 → 直接吃光。
2. **CUDA Graph 捕获尺寸过多**（`max_num_seqs=1024` 时每档都捕获），显存爆在"看不见"的地方。
3. 多模态模型：dummy 输入只测了文本，图片的激活峰值没测到。
4. TP 下 **rank 间不均衡**（MoE 专家落点不均），最热的 rank 先 OOM。

**处置**：

```bash
# 1) 先确认是哪种 OOM：启动即炸 vs 跑到一半炸
# 2) 保守启动
vllm serve <model> --max-model-len 32768 --gpu-memory-utilization 0.85 --enforce-eager
# 3) 逐步放开：先关图 → 开图 → 加长度 → 加并发
# 4) 0.34+ 直接用自动定容（移除 gpu_memory_utilization），让系统按真实余量分配
```

#### 5.3.2 吞吐不达标：三层排查法

```
Layer 1 计算层：GPU 利用率（nvidia-smi / dcgm）
   < 50% → 不是算力问题，往下
   > 85% → 算力/带宽到顶，只能量化/加卡/换 kernel

Layer 2 数据层：是否频繁 host↔device 拷贝、是否每 step 重建张量
   看 nsys：cudaMemcpy 占比、kernel 之间空隙

Layer 3 调度层：CPU 时间去哪了
   py-spy top --pid <engine_core_pid> → 看是否卡在 Python/hash/序列化
```

**常见结论**：

- 小模型 + 高 QPS：**CPU 调度是瓶颈** → 开 async scheduling、加 `--api-server-count`、减少日志。
- 大模型 + 低 QPS：**带宽是瓶颈** → 量化权重/KV、增大 batch。
- MoE：**all-to-all 或专家不均衡** → EP/EPLB/DeepEP。
- 长 prompt：**prefill 是瓶颈** → 大 chunk、FA 后端、PD 分离。

#### 5.3.3 输出乱码：最危险的一类

**症状**：模型能跑、不报错，但输出是 `!!!!!`、重复字符、或语义完全崩坏。

**根因排序**：

1. **量化 kernel 与 checkpoint 打包格式不匹配**（尤其国产卡/自定义后端）。shape/dtype 都对，语义错。
2. **chat template 错误**（少 special token、BOS 重复、system 位置错）。
3. **RoPE scaling / YaRN 配置不一致**（长上下文外推）。
4. **前缀缓存串味**（哈希漏了 LoRA/多模态/租户）。
5. **KV 量化 scale 错误**（校准数据不匹配、per-head scale 维度错）。

**定位手法**：

```bash
# ① 关掉所有优化，最小复现
vllm serve <model> --enforce-eager --no-enable-prefix-caching --kv-cache-dtype auto
# ② 固定 greedy，比对 transformers 同 prompt 输出
# ③ 若 transformers 正常而 vllm 异常 → 是 vllm 侧（模板/权重加载/kernel）
# ④ 二分：只开量化 / 只开前缀缓存 / 只开图
```

#### 5.3.4 抢占风暴

**症状**：`num_preemptions_total` 持续增长，TPOT p99 飙升，用户看到"回答到一半卡住又重来"。

**根因**：KV 预算 < 在途 token 需求。触发条件通常是：

- `max_num_seqs` 设得比 KV 能支撑的并发大；
- 请求长度分布有长尾（少数 32K 请求吃掉大量块）；
- 前缀缓存把块"占住"（引用计数未释放或缓存过多）。

**处置**：

```bash
--max-num-seqs 128          # 先砍并发
--max-model-len 32768       # 砍上限
--kv-cache-dtype fp8        # 容量 ×2
# 或加卡 / 开 PD 分离（把长 prefill 挪走）
```

**监控建议**：把 `preemptions / requests` 作为**容量水位线**，>2% 就该扩容。

#### 5.3.5 NCCL / 多机问题

```bash
# 基线：先证明硬件与拓扑没问题
all_reduce_perf -b 8 -e 8G -f 2 -g 8
# 打开 NCCL 日志
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET vllm serve ...
# 常见坑
# - 容器内 IB 设备未挂载（--device=/dev/infiniband）
# - 多网卡未指定 NCCL_SOCKET_IFNAME
# - 防火墙阻断临时端口
# - 跨机 MTU 不一致
```

### 5.4 常态化观测（上线必配）

**四个黄金指标**：

```
1. TTFT p50/p99（按 prompt 长度分桶）
2. TPOT p50/p99（按输出长度分桶）
3. KV 使用率 + 抢占率
4. 前缀缓存命中率（按租户/场景）
```

**告警建议**：

| 指标 | 阈值 | 含义 |
|---|---|---|
| `num_preemptions_total` 5min 增量 > 请求数 2% | 容量不足 | 扩容或降并发 |
| `gpu_cache_usage_perc` > 0.95 持续 10min | KV 见顶 | 同上 |
| TTFT p99 > SLO × 1.5 | 排队/长文 | 查 queue time |
| prefix 命中率环比跌 20% | 模板/路由变了 | 查 prompt 构造 |
| `num_requests_waiting` 持续 > 0 | 过载 | 限流/扩容 |

---

> 上一篇：[04-性能与调参](04-性能与调参.md) ｜ [返回总览](00-总览.md) ｜ 下一篇：[06-改代码实战](06-改代码实战.md)


# 第 6 章 改代码：扩展点与实战

> [← 总览](00-总览.md) ｜ 下一章：[07-面试题库](07-面试题库.md)

> **前置纪律**：优先用**官方扩展点**（插件、注册表、配置），其次 **out-of-tree 包 + monkeypatch**，**最后**才 fork。直接改 `site-packages` 里的 vLLM 是运维灾难（升级即丢失、问题难复现）。国产卡适配的成熟做法就是"**独立插件包**"（如 `vllm_ecu` 那样：platform/attention/kernels/quantization/compilation/worker/patch 分目录）。

### 6.1 源码阅读路线（按依赖顺序，2 天能通）

```
Day 1
  vllm/config/*.py                     # 先搞清参数如何变成配置对象
  vllm/v1/engine/core.py               # EngineCore 主循环：step() 长什么样
  vllm/v1/core/sched/scheduler.py      # schedule() 主干
  vllm/v1/core/kv_cache_manager.py     # allocate_slots / get_computed_blocks
  vllm/v1/core/block_pool.py           # 哈希表 + LRU + 引用计数
Day 2
  vllm/v1/worker/gpu/model_runner.py   # prepare_inputs → execute_model → sample
  vllm/v1/attention/backends/registry.py + flash_attn.py
  vllm/model_executor/models/<model>.py # 一个具体模型的 forward
  vllm/v1/spec_decode/eagle.py         # 投机解码如何插入
```

**读源码的三个技巧**：

1. **从日志反推**：启动日志里每行 `INFO` 都对应一个初始化决策，顺着日志找代码最快。
2. **打印数据结构**：`SchedulerOutput`、`ModelRunnerOutput`、`KVCacheBlocks` 打一次 repr，比读十页代码管用。
3. **用 `--enforce-eager -O0` 跑最小模型**，打断点看每个 step 的输入变化。

### 6.2 扩展点矩阵

| 想做什么 | 扩展点 | 入口 | 侵入性 |
|---|---|---|---|
| 支持新硬件 | **Platform 插件** | `vllm.platform_plugins` entry point | 低（独立包） |
| 新 attention kernel | **AttentionBackend** | `register_backend()` | 低 |
| 新量化格式 | **QuantizationConfig** | `register_quantization_config()` | 中 |
| 新模型结构 | **ModelRegistry** | `register_model()` | 低 |
| 改调度策略 | **scheduler_cls** | `--scheduler-cls` | 中 |
| P/D 传输 | **KVConnector** | `kv_connector` 配置 | 中 |
| 自定义 logits 处理 | **LogitsProcessor** | 请求级 API | 低 |
| 新解析器（tool/reasoning） | **Parser 注册** | `--tool-call-parser` / `--reasoning-parser` | 低 |
| 权重加载格式 | **LoadFormat / loader** | `--load-format` | 中 |
| 全局 monkeypatch | **general_plugins** | `vllm.general_plugins` entry point | 高（谨慎） |

### 6.3 实战 A：自定义 Attention Backend

**场景**：国产卡只有 Triton，但你需要支持 sliding window + FP8 KV。

```python
# my_backend/attn.py
import torch
from vllm.v1.attention.backends.registry import register_backend
from vllm.attention.backends.abstract import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder

class MyAttnMetadata(AttentionMetadata):
    # 每 step 由 builder 填充
    block_table: torch.Tensor          # [num_reqs, max_blocks]
    slot_mapping: torch.Tensor         # [num_tokens]
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    max_query_len: int

class MyAttnMetadataBuilder(AttentionMetadataBuilder):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        # 从 SchedulerOutput 的块表/槽位构造 kernel 描述符
        return MyAttnMetadata(...)

class MyAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MY_ATTN"

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 80, 96, 112, 128, 256]

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
        # 注意布局要与 kernel 的索引方式一致（[2, blocks, kv_heads, block, dim] 最常见）
        return (2, num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_builder_cls():
        return MyAttnMetadataBuilder

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

register_backend("MY_ATTN", MyAttentionBackend)   # 之后用 VLLM_ATTENTION_BACKEND=MY_ATTN 选择
```

**验收清单**：

- [ ] `get_kv_cache_shape` 与 kernel 的 stride 假设一致（**最常见的静默错误来源**）
- [ ] chunked prefill：`query_start_loc` 不连续时能正确切分
- [ ] 投机解码：`max_query_len > 1` 时 mask 正确（tree/causal）
- [ ] sliding window：窗口外的块被正确跳过
- [ ] 与 prefix caching 组合：`block_table` 含共享块时不重复写
- [ ] 数值对齐：与 FlashAttention 对比 `max_abs_diff < 2e-2`（BF16）

### 6.4 实战 B：改调度策略（SLO / 优先级 / 抢占）

#### 方案 1（推荐）：应用层准入控制

不改 vLLM，在 API 网关做**分级队列 + 并发配额**：

```python
# 伪代码：按优先级分配并发令牌
class Admission:
    def __init__(self, quotas={"gold": 64, "silver": 32, "bronze": 16}):
        self.sem = {k: asyncio.Semaphore(v) for k, v in quotas.items()}

    async def handle(self, req, tier):
        async with self.sem[tier]:          # 超卖保护
            return await vllm_client.chat(req)
```

优点：零侵入、可灰度、易回滚。绝大多数"要 SLO"的需求到这里就够了。

#### 方案 2：自定义 Scheduler（进阶）

0.25.1 已确认支持 `--scheduler-cls`（`SchedulerConfig.scheduler_cls`，可传类或 `module:Class` 字符串）。

**先看清结构**：受害者选择发生在 `Scheduler.schedule()` 内部——`allocate_slots()` 返回 `None` 时才调用 `_preempt_request()`。所以"换一种抢占策略"的本质是**在 schedule() 里插入自己的选择点**，只重写 `_preempt_request()` 是改不了"谁被牺牲"的（它只负责释放块与状态回滚）。

**做法**：从源码复制一份 `schedule()`，只在开头插入"主动让位"逻辑，其余完全走父类：

```python
# my_sched.py
import time
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

class PriorityScheduler(Scheduler):
    """高优先级请求等待且 KV 紧张时，主动让低优先级请求让位。"""

    def _pick_victim(self, running: list[Request]) -> Request | None:
        if not running:
            return None
        # 值越小越先被牺牲：低优先级 → 已算 token 越少（重算越便宜）→ 越晚到达
        return min(running, key=lambda r: (
            getattr(r, "priority", 0),
            r.num_computed_tokens,
            -getattr(r, "arrival_time", 0),
        ))

    def schedule(self, throttle_prefills: bool = False):
        # ① KV 水位过高且还有等待者 → 主动抢占一个"最不痛"的 running 请求
        if self.waiting and self.kv_cache_manager.usage > 0.9:
            victim = self._pick_victim(self.running)
            if victim is not None:
                self._preempt_request(victim, time.monotonic())
        # ② 其余逻辑全部交给父类，保证预算/块状态/连接器顺序不被破坏
        return super().schedule(throttle_prefills=throttle_prefills)
```

```bash
vllm serve <model> --scheduler-cls my_sched:PriorityScheduler
```

> 注意：`self.waiting` 在部分版本里是 `RequestQueue` 而非 list，`usage` 的访问方式也可能不同——**以实际源码为准**。这段代码的价值是说明"策略插入点在哪"，不是让你逐字抄。

**必须遵守的不变量**（改错会直接死锁/漏 token）：

1. 每个 running 请求每 step **至少推进 1 token**，否则会饿死。
2. `num_scheduled_tokens` 总和 ≤ `max_num_batched_tokens`。
3. 被抢占请求的块必须**全部释放**，且 `num_computed_tokens` 归零。
4. `SchedulerOutput` 里的 `block_ids` 必须与 `KVCacheManager` 状态一致。
5. 与 KV connector 的 `update_state_after_alloc` 顺序不能乱。

### 6.5 实战 C：接入新模型（out-of-tree）

```python
# my_models/qwen_custom.py
import torch
from torch import nn
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.layers.logits_processor import LogitsProcessor

class MyQwenForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        self.model = Qwen3Model(vllm_config=vllm_config, prefix=prefix)
        self.lm_head = nn.Linear(...)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden = self.model(input_ids, positions, intermediate_tensors)
        return hidden

    def compute_logits(self, hidden_states, sampling_metadata):
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

    def load_weights(self, weights):
        # 用 AutoWeightsLoader 处理名字映射/分片/量化 scale
        from vllm.model_executor.model_loader.weight_utils import AutoWeightsLoader
        return AutoWeightsLoader(self).load_weights(weights)
```

```python
# my_models/__init__.py
from vllm import ModelRegistry
ModelRegistry.register_model(
    "MyQwenForCausalLM", "my_models.qwen_custom:MyQwenForCausalLM"
)
```

```toml
# pyproject.toml：让 vllm 自动加载你的注册逻辑
[project.entry-points."vllm.general_plugins"]
my_models = "my_models:register"
```

**踩坑点**：

- `config.json` 的 `architectures` 必须与注册名一致，或用 `--hf-overrides '{"architectures":["MyQwenForCausalLM"]}'`。
- 权重名映射错 → 表现为"能加载但输出退化"，务必用 `load_weights` 的返回值检查未加载参数。
- 自定义 attention/MoE 时，**KV cache spec 要显式声明**（`get_kv_cache_spec`），否则块形状对不上。

### 6.6 实战 D：非 NVIDIA 平台插件（以国产加速卡为例）

**目标**：让 `vllm` 在自定义设备上跑起来，且**不改 site-packages**。

典型的插件包结构（参考 `vllm_ecu` 的分层）：

```
my_accel_vllm/
├── __init__.py            # register_platform_plugins() → 返回 Platform 类路径
├── platform/device.py     # Platform 子类：设备属性、backend 选择、可用性
├── worker/worker.py       # Worker：显存池、memcpy、图捕获适配
├── attention/             # 自定义 attention backend（如 Triton/手写 kernel）
├── kernels/               # 自定义算子（RMSNorm/RoPE/量化 GEMM/MoE）
├── quantization/          # 量化格式注册（AWQ/GPTQ 变体）
├── compilation/           # 图/编译适配
├── distributed/           # 通信（NCCL 替代/网卡绑定）
└── patch/                 # 版本锁定的 monkeypatch（带版本断言）
```

```python
# __init__.py
def register_platform_plugins() -> str:
    """vLLM 加载平台插件时调用，返回 Platform 实现的完整类路径。"""
    return "my_accel_vllm.platform.device:MyAccelPlatform"

def register_quantization_plugins() -> None:
    from my_accel_vllm.quantization import register_all
    register_all()
```

```toml
[project.entry-points."vllm.platform_plugins"]
my_accel = "my_accel_vllm:register_platform_plugins"
[project.entry-points."vllm.general_plugins"]
my_accel_quant = "my_accel_vllm:register_quantization_plugins"
```

**平台适配的六个必答题**：

| 问题 | 影响 |
|---|---|
| `device_type` / 设备枚举 API | vLLM 用 `current_platform` 分支，错了直接 import 失败 |
| 显存查询与分配器（是否支持 caching allocator / 统一内存） | 决定 KV 池与图捕获能否工作 |
| 是否有等价 NCCL 的集合通信库 | 决定 TP/EP 是否可用 |
| 支持哪些 dtype / 是否有 FP8/NVFP4 tensor core | 决定量化路线 |
| 自定义 kernel 的 stream / event 语义 | 决定异步调度能否开 |
| 版本绑定策略 | 框架升级时插件是否失效 |

**血泪经验（务必内建）**：

1. **kernel 选择必须"按 checkpoint 格式"而不是"按名字"**：同名 AWQ 有 GEMM/WNA16 两种打包，选错**不报错、只输出乱码**。落地手段：读取 checkpoint 的 `quantization_config` 字段（`checkpoint_format`/`version`/`group_size`）做显式分派，并在启动时跑金标准自检。
2. **不要动厂商安装目录**：所有适配放在插件包里，用 `patch/` 做带版本断言的 monkeypatch（`assert vllm.__version__ == "0.25.1"`），升级时不静默生效。
3. **建立"算子级"回归**：每个自定义 kernel 都要有 PyTorch 参考实现的数值对比测试（BF16 容差 1e-2，FP8 容差 5e-2），并覆盖边界 shape（head_dim=64/128、seq=1、seq=8192）。

### 6.7 实战 E：KV Connector（P/D 与 Offload）

```python
# 接口形状以 vllm/distributed/kv_transfer/kv_connector/v1/abstract.py 为准
class MyConnector(KVConnectorBase_V1):
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """返回可从远端命中的 token 数（P/D 时为 prompt 长度）。"""
        return self.store.lookup(request.request_id)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        ...  # 记录要拉取的块，触发异步传输

    def build_connector_meta(self, scheduler_output):
        return MyMeta(...)   # 随 SchedulerOutput 下发到 worker

    def start_load_kv(self, metadata): ...
    def wait_for_layer_load(self, layer_name): ...   # 逐层流水，隐藏传输延迟
    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kw): ...
    def wait_for_save(self): ...
    def request_finished(self, request, block_ids): ...
```

**验收**：

- [ ] 传输失败能回退到本地重算（0.25+ 有专门测试用例可参考）
- [ ] 逐层流水（不是算完整个 prefill 再传）
- [ ] 块生命周期与 `KVCacheManager` 引用计数一致（过早释放 → 脏读）
- [ ] 指标：传输字节、传输耗时、命中率、失败率

### 6.8 实战 F：接入自定义量化格式

```python
from vllm.model_executor.layers.quantization import register_quantization_config

@register_quantization_config("my_int4")
class MyInt4Config(QuantizationConfig):
    def get_name(self) -> str: return "my_int4"
    def get_supported_act_dtypes(self): return [torch.bfloat16, torch.float16]
    def get_min_capability(self): return 70
    @classmethod
    def get_config_filenames(cls): return ["quantize_config.json"]
    @classmethod
    def from_config(cls, config): ...

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            return MyInt4LinearMethod(self)
        return None
```

**必须配套的东西**（少一个就会在某个模型上崩）：

1. **权重加载钩子**：处理 `qweight/qzeros/scales/g_idx` 的打包与转置。
2. **shape 断言**：`in_features % group_size == 0`、`packed_dim` 正确。
3. **回退路径**：kernel 不支持时用 dequant + F.linear（能跑但慢，保证不崩）。
4. **数值自检**：金标准 prompt + 与 BF16 的 KL/一致性对比。

### 6.9 调试工具链

```bash
# 环境与版本
vllm collect-env

# 日志
VLLM_LOGGING_LEVEL=DEBUG
VLLM_TRACE_FUNCTION=1          # 打印调用栈（很慢，只用于定位）

# 图/编译
TORCH_LOGS=graph_breaks,recompiles
VLLM_DEBUG_DUMP_PATH=/tmp/dump  # dump 张量

# 正确性
VLLM_GPU_SYNC_CHECK=warn       # 检查未同步的读写（定位异步 bug）
VLLM_BATCH_INVARIANT=1         # 批不变（可复现）

# 性能
nsys profile -t cuda,nvtx,osrt -o prof vllm serve ...
py-spy top --pid <EngineCore pid>
nvidia-smi dmon -s u,m          # 利用率与显存
```

**性能剖析的正确顺序**：

```
1. 先看"step 时间"分布：scheduler / forward / sample / detokenize
2. 再看 forward 内部：attention vs MLP/MoE vs 通信
3. 最后看 kernel：是否选错、是否低效、是否有空隙
```

### 6.10 打补丁的正确姿势

```python
# my_accel_vllm/patch/__init__.py
import vllm
from packaging.version import parse

SUPPORTED = {"0.25.1"}

def apply():
    if vllm.__version__ not in SUPPORTED:
        # 版本不匹配：告警但不 patch，避免"静默行为改变"
        import warnings
        warnings.warn(f"patch skipped: vllm {vllm.__version__} not in {SUPPORTED}")
        return

    from vllm.v1.worker.gpu_worker import Worker
    _orig = Worker.determine_available_memory
    def patched(self):
        return _orig(self) - self.requested_memory  # 你的设备有额外常驻
    Worker.determine_available_memory = patched
```

**原则**：**补丁要有版本断言、要有开关、要有日志**。否则半年后没人知道线上跑的是"哪个魔改版"。

---

> 上一篇：[05-故障诊断](05-故障诊断.md) ｜ [返回总览](00-总览.md) ｜ 下一篇：[07-面试题库](07-面试题库.md)


# 第 7 章 面试题库

> [← 总览](00-总览.md) ｜ 下一章：[08-附录](08-附录.md)

> 用法：先自己答，再看"要点"。每题给 2–6 行**必须说到的点**，括号里是常见追问。

### 7.1 架构与演进

**Q1. vLLM 最核心的创新是什么？**
三点：① **PagedAttention**——把 KV Cache 按固定块分页管理，消除外部碎片，显存利用率从 20–38% 提到 90%+；② **continuous batching**——以 token 为粒度动态组批，请求完成即释放槽位；③ **统一的服务栈**——OpenAI 兼容 API + 调度 + 分布式执行 + 多硬件后端。一句话：**它把"推理"做成了带内存管理与调度的系统**。

**Q2. 0.25 版本"删除 PagedAttention"是什么意思？**
删的是**原始 PagedAttention CUDA kernel 实现**（V0 时代的 attention 算子），不是块式 KV 管理架构。块表、slot_mapping、前缀缓存、块级分配全部保留，只是 attention 计算统一交给 FlashAttention/FlashInfer/FlashMLA 等后端完成。（追问：那 PagedAttention 论文还有意义吗？——有，它定义的是**内存管理抽象**，今天所有后端都在这个抽象上工作。）

**Q3. V0 与 V1 的核心区别？为什么重写？**
V0：prefill/decode 双路径、调度与执行串行、每 step 重建张量、swap 抢占、CPU 开销高。V1：统一 token budget 调度、EngineCore 独立进程、持久化 InputBatch、recompute 抢占、torch.compile、可插拔调度器/后端/连接器。重写动因是**CPU 开销与代码分叉**：小模型高 QPS 下 GPU 利用率被 Python 拖到 50% 以下，且双路径导致 bug 要修两遍。

**Q4. EngineCore 为什么要独立进程？**
CPU 隔离（tokenize/HTTP/JSON 不抢 GIL）、多前端复用（`--api-server-count`）、故障隔离（前端崩不丢 KV）、为异步调度提供"调度与执行解耦"的结构基础。代价是多一次 IPC，所以 vLLM 对 `SchedulerOutput` 做了零拷贝/共享内存优化。

**Q5. V1 为什么去掉 swap？什么时候 swap 反而更好？**
swap 需要等量 CPU pinned 内存 + PCIe 双向带宽，而被抢占的请求往往很久才回来，收益低；recompute 实现简单、零额外显存，且重算时大概率命中前缀缓存。**swap 更好的场景**：KV 极大（重算成本 > 传输成本）、CPU 内存充裕、PCIe 空闲、请求很快恢复。

**Q6. Model Runner V2 解决什么问题？**
V1 把调度做干净后，runner 成为新的复杂度与 CPU 瓶颈中心（每 step Python 开销、spec decode 分支、多模态、MoE/EPLB、PP）。MRv2 用**状态化输入（InputBatch/Buffer）+ 分阶段执行（prepare/execute/sample）+ 统一 drafter 接口 + 模块化并行工具**重写，支持 RLHF 热更新、EPLB、图捕获分桶。0.34 起 MRv1 已删除。

**Q7. 异步调度怎么工作？风险是什么？**
调度器在 GPU 执行 step N 时就准备 step N+1，把 CPU 时间藏进 GPU 时间。风险：调度基于"尚未返回的输出"做假设，一旦假设错（停止判断、grammar bitmask、draft token），会输出错乱且**只在特定并发下复现**；因此 PP、CPU 后端、部分 spec decode 组合被逐步放开。收益与"CPU/GPU 时间比"正相关。

**Q8. vLLM 怎么做到支持这么多硬件？**
`Platform` 抽象 + 插件机制：`vllm.platform_plugins` entry point 返回平台类，平台负责设备枚举、显存、可用 backend、通信库、图捕获等；attention/quantization 各自有 registry。国产卡的标准做法是**独立插件包**（platform/worker/kernels/quantization/patch 分层），不修改 site-packages。

**Q9. 描述一次请求在 vLLM 中的完整路径。**
HTTP → chat template → tokenize → `add_request` 入 waiting → `schedule()`（前缀命中 + 块分配 + 组 SchedulerOutput）→ `execute_model`（prepare_inputs → forward → sample）→ `update_from_output`（追加 token、判停、必要时再分配/抢占）→ detokenize → SSE 流式返回 → 释放块并写入前缀缓存。

**Q10. 调度器是单线程的，会不会成为瓶颈？**
会，而且这是设计选择：并发来自批处理而非多线程。因此**调度路径上任何阻塞（哈希、日志、序列化、全局锁）都会直接变成 GPU 空转**。缓解手段：异步调度、把 tokenize/detokenize 拆到前端进程、编译掉 Python 开销、减少 per-step 对象构造。

**Q11. 为什么 V1 默认开前缀缓存？代价是什么？**
收益：多轮/RAG/Agent 场景 TTFT 与算力大幅下降。代价：哈希计算（O(prompt)）、块查找、缓存占用 KV、跨租户侧信道风险（需 `cache_salt`）、与 chunked prefill/投机解码的交互复杂度。

**Q12.（开放）如果让你设计 vLLM 下一代，你会改什么？**
参考方向：KV 作为一等存储对象（分层/共享/跨实例）、调度与执行进一步解耦（SLO 感知）、编译与 kernel 自动生成（IR）、前端与服务网格融合、以及"按 token 计费"的 QoS 控制。答出**权衡**比答出**功能**更重要。

### 7.2 KV Cache 与 PagedAttention

**Q13. 每 token 的 KV 显存怎么算？**
`2 × L × H_kv × D × dtype_bytes`（MHA/GQA）；MLA 为 `L × (D_latent + D_rope) × bytes`。例：40 层、8 KV head、head_dim 128、BF16 → 40 KB/token；32K 上下文约 1.3 GB/请求。

**Q14. block_size 怎么选？**
小（8/16）：尾部浪费少、前缀命中粒度细、block_table 长；大（32/64）：元数据少、kernel 内循环友好、尾部最多浪费 63 token。0.34 默认 64 是"序列变长 + kernel 偏好"的结果；短 prompt 场景可显式调小。

**Q15. block_table 与 slot_mapping 的区别？**
`block_table[req][logical_block] = physical_block`，是**序列级的地址翻译表**；`slot_mapping[token] = physical_block * block_size + offset`，是**token 级的写入位置**。前者用于 attention 读 KV，后者用于把新算的 KV 写到正确槽位。

**Q16. 前缀缓存哈希要包含哪些因素？漏了会怎样？**
必须包含：模型权重版本、LoRA/adapter id、多模态输入哈希、block_size、attention 类型/窗口、TP/PP 拓扑、租户 salt。漏了会"串味"——复用不属于自己的 KV，输出语义错乱且不报错。

**Q17. partial block caching 解决什么？**
只缓存满块时，prompt 长度非 block 整数倍的最后一段永远无法复用；partial 缓存用 `(parent_hash, start, tokens)` 把不满块也纳入，RAG/多轮场景命中率可显著提升（经验值 +5~10pt）。代价是哈希更复杂、块写满后要"转正"。

**Q18. 抢占有哪两种方式？V1 为什么选 recompute？**
swap（KV 换到 CPU）与 recompute（丢弃 KV 重新 prefill）。V1 选 recompute：零额外显存、实现简单、与前缀缓存协同；代价是重算算力。**swap 更适合 KV 巨大且 PCIe 空闲**的场景。

**Q19. 被抢占后 prefix cache 还有用吗？**
有。被释放的块如果仍在前缀缓存表中，重算时可以命中，从而把"重算成本"降到接近 0。这也是 recompute 在 V1 可行的关键前提。

**Q20. KV 量化对精度的影响怎么评估？**
不能只看 perplexity。要跑：长上下文 needle（KV 量化最敏感）、代码单测通过率、JSON 合法率、数学准确率，并与 BF16 做 KL/一致性对比。K 比 V 敏感，常见做法是 K 高精度、V 低精度。

**Q21. MLA 的 KV 为什么小？**
DeepSeek 系把 K/V 压缩成 latent（如 512）+ rope（64），每层只存一份 latent，不随 head 数增长；GQA 虽共享 KV head，但仍按 head 存。长上下文下 MLA 的每 token KV 更可控，代价是 attention 要"解压"，kernel 更复杂。

**Q22. 什么是多 KV group？**
一个模型里不同层的 KV 规格不同（full attention、sliding window、Mamba state），需要多个 KV cache group 分别管理、甚至可以按 group 选不同 attention backend（0.26+）。这是混合架构（Qwen3-Next、Kimi 系）能跑的前提。

**Q23. KV offload 的收益与代价？**
收益：用 CPU/磁盘扩大"等效 KV 容量"，提升前缀命中率，支撑长会话/多租户。代价：PCIe/NVMe 带宽与延迟、tier 感知调度复杂度、失败恢复。只值得对**高频复用**的块做。

**Q24. HiSparse 是什么？**
面向稀疏 MLA decode 的"主机内存 KV 层"：GPU 显存紧张时把 KV 页溢出到 pinned host memory，decode 时按 top-k 取，miss 由 per-request GPU 热缓冲兜底。本质是**用延迟换容量**的软降级。

**Q25. 显存不够时，按优先级有哪些手段？**
① 降 `max_model_len`；② KV 量化（FP8/NVFP4）；③ 权重 4bit 量化；④ 降 `max_num_seqs`；⑤ 前缀缓存/offload 调优；⑥ 加卡/TP/EP；⑦ PD 分离把 prefill 挪走。

**Q26. beam search / 并行采样怎么省 KV？**
多条候选共享已算前缀的物理块（引用计数 + 写时复制）：只有分叉后新写的块才复制。`n=8` 的显存从 8× 降到接近 1×（分叉前）。

### 7.3 调度与批处理

**Q27. continuous batching 相比 static batching 收益从哪来？**
静态批要等最慢请求且槽位固定；连续批在**每个 step 重新组批**，完成的请求立刻释放、新请求立刻进入，GPU 不空转。收益与"请求长度方差"正相关：方差越大，收益越大。

**Q28. chunked prefill 的收益与代价？**
收益：长 prefill 不再阻塞 decode，ITL 稳定；显存峰值下降。代价：长 prompt 的 TTFT 变长；实现复杂（partial KV、chunk 边界、与 spec/结构化输出的交互）。

**Q29. decode 优先还是 prefill 优先？**
交互式服务 decode 优先（在途承诺不能卡）；离线批处理可以 prefill 优先（吞吐最大化）。vLLM 默认 decode 优先，通过 token budget 剩余给 prefill。

**Q30. `max_num_seqs` 与 `max_num_batched_tokens` 的关系？**
前者限制**并发序列数**（影响 KV 与调度开销），后者限制**单 step token 总量**（影响计算量与延迟）。二者共同决定 batch 形状：`max_num_batched_tokens` 是硬预算，`max_num_seqs` 是并发上限。

**Q31. 调度器怎么选抢占受害者？**
默认倾向"最后进入 running"的请求（可能尚无用户可见输出、重算成本低）。生产中应按优先级/已生成 token/剩余预算加权，这需要自定义 `scheduler_cls`。

**Q32. 优先级调度怎么实现？**
两层：① 应用层准入控制（分级信号量/队列，零侵入，推荐）；② 自定义 Scheduler 重写 waiting 队列选择与抢占策略（需遵守不变量：每 step 至少推进 1 token、预算不超、块状态一致）。

**Q33. 为什么按长度排序/分桶能提吞吐？**
减少 padding 与抢占、让同一 batch 的序列长度接近，提高 kernel 效率与 KV 利用率。离线批处理必做，在线可用"短请求优先"策略（但要防饿死）。

**Q34. 一个 batch 里既有 prefill 又有 decode，attention 怎么算？**
V1 用**统一 attention 接口 + 元数据**描述混合 batch：prefill 段走 causal/chunked 路径，decode 段走 q_len=1 路径，kernel 内部按 `query_start_loc` 切分；部分后端用"分离调用 + 拼接"实现。

**Q35. 请求取消/超时怎么处理？**
客户端断开 → 前端发 abort → EngineCore 标记请求完成并释放块；超时由前端/网关控制。注意：**取消要及时释放 KV**，否则长尾请求会拖垮缓存。

**Q36. pause/resume 有什么用？**
RLHF 权重热更新（暂停→`update_config`/`reload_weights`→恢复）、模型热切换、故障转移、按需释放 GPU。要求引擎状态可序列化、在途请求可保留或安全丢弃。

### 7.4 性能与调优

**Q37. TTFT 由什么决定？怎么优化？**
排队时间 + prefill 计算 + 首 token 采样。优化：加副本/降排队、chunked prefill（对短请求不利，对长请求有利）、前缀缓存、FA/FlashInfer 后端、PD 分离、减少 tokenize 开销。

**Q38. TPOT/ITL 由什么决定？**
decode 每 step 时间 = 权重读取 + KV 读取 + 通信 + kernel launch。优化：权重量化（减带宽）、KV 量化（减带宽）、CUDA Graph（减 launch）、TP 通信优化（NVLink/custom allreduce）、避免抢占/插队。

**Q39. 吞吐与延迟的权衡曲线长什么样？**
典型"延迟-吞吐"曲线：小 batch 延迟低吞吐低；随 batch 增大吞吐上升、延迟上升；到 KV/带宽饱和后吞吐平台化、延迟陡增。**最优点在吞吐拐点前**；SLO 决定你停在曲线哪一段。

**Q40. 怎么判断 compute-bound 还是 memory-bound？**
decode：算 `算术强度 ≈ 2×参数量 / (权重字节 + KV字节)`，远小于硬件 ridge point → memory-bound。prefill：大矩阵乘 → compute-bound。工具：nsys 看 kernel 时间与 DRAM 吞吐、`nvidia-smi dmon` 看利用率。

**Q41. GPU 利用率低怎么查？**
分层：① 是否有图（对比 `--enforce-eager`）；② CPU 是否瓶颈（py-spy、async scheduling）；③ batch 是否太小（`max_num_seqs`、请求速率）；④ 是否有通信等待（NCCL 时间线）；⑤ 是否频繁抢占/换入换出。

**Q42. 压测怎么才可信？**
开环（固定 QPS/Poisson 到达）+ 真实长度分布 + 报告 p50/p95/p99 + 跑足够久（覆盖缓存稳态）+ 记录抢占/命中率。闭循环会低估排队延迟。

**Q43. goodput 是什么？**
满足 SLO（如 TTFT<500ms 且 TPOT<50ms）的**有效吞吐**。比峰值吞吐更能反映用户体验与成本。

**Q44. 容量规划怎么做？**
三步：算每 token KV → 算可用 KV 与最大在途 token → 用 Little's Law（并发=QPS×延迟）与 roofline（带宽/算力）反推所需副本数。留 20% 余量。

**Q45. 为什么 decode 增大 batch 吞吐近似线性？什么时候不线性？**
权重只需读一次、batch 内共享，batch 翻倍→吞吐翻倍（带宽不变时）。不线性：KV 读取随 batch 线性增长、超过 L2/HBM 带宽、通信/调度开销上升、preemption 出现。

**Q46. CUDA Graph 的收益与失效场景？**
收益：把数千次 kernel launch 压成一次，decode 小 batch 下可提 20–50%。失效：动态控制流、变长 all-reduce、动态 draft 长度、LoRA 切换、非 graph-safe 自定义 kernel；表现为回退 eager 或捕获失败。

### 7.5 并行与分布式

**Q47. TP 的通信量怎么算？为什么不宜跨机？**
每层两次 all-reduce，单次 `B×S×H×bytes`；decode 时 S=1，消息小、延迟主导。跨机 Ethernet/RoCE 的延迟与带宽（25–100 Gbps）远低于 NVLink（数百 GB/s），通信会吃掉 decode 预算。

**Q48. PP 的气泡怎么算？怎么减小？**
1F1B 下气泡比例 ≈ `(P-1)/(M+P-1)`（P stage 数，M micro-batch 数）。减小：增大 M、异步调度与 PP 重叠（0.16）、interleaved schedule、把 stage 切得更均衡。

**Q49. DP 与 TP 怎么选？**
能单卡放下 → 优先 DP（多副本，线性扩吞吐、零通信）；单卡放不下 → TP；TP 到 8 仍不够 → PP 跨机。DP 的代价是显存复制与负载均衡。

**Q50. EP 的 all-to-all 为什么贵？**
每层每 token 要把 hidden 发给选中的专家、再收回结果；跨机时是双向 all-to-all，量大且对网络抖动敏感。DeepSeek 类模型每层两次，是跨机部署的主要障碍，所以有 DeepEP/EPLB/Elastic EP。

**Q51. EPLB 是什么？**
Expert Parallel Load Balancing：统计各专家负载，动态复制/迁移热专家、调整放置，缓解"热专家排队、冷专家闲置"。MoE 生产部署必备。

**Q52. CP / DCP / PCP 的区别？**
CP（context parallel）：把序列切到多卡算 attention（环状通信）；DCP（decode context parallel）：decode 阶段按 KV 长度切分，降低单卡 KV 带宽压力；PCP（prefill context parallel）：prefill 阶段切分超长 prompt。三者解决"长上下文单卡放不下/算不动"。

**Q53. MoE 模型推荐什么拓扑？**
单机：TP=8 + EP=8（专家在机内 all-to-all，走 NVLink）；多机：EP 跨机 + DeepEP + EPLB，TP 只做机内；decode 用 PD 分离把 prefill 的 all-to-all 洪峰隔离。

**Q54. NCCL hang 怎么排查？**
`NCCL_DEBUG=INFO` 看卡在哪一步（init/connect/allreduce）；`all_reduce_perf` 验证链路；检查 IB 设备挂载、`NCCL_SOCKET_IFNAME`、防火墙、MTU、GPU 拓扑（`nvidia-smi topo -m`）。

**Q55. TP 为什么要求 head 数能被整除？**
每个 rank 分到 `H/T` 个 head，切分必须整数；GQA 下要求 KV head 数也能被整除（否则要复制 KV head）。不满足时 vLLM 会报错或自动调整。

**Q56. 多机推理怎么保证确定性？**
浮点归约顺序、kernel 选择、batch 组合都会影响结果。手段：`VLLM_BATCH_INVARIANT=1`（批不变）、固定 batch/拓扑、确定性 kernel、greedy 采样；完全确定性代价高，通常只要求"统计等价"。

### 7.6 量化

**Q57. W4A16 为什么能加速 decode？**
decode 是带宽受限：权重字节减半 → 每 step 读取时间减半 → 吞吐近似翻倍（低并发）。prefill 是计算受限，W4A16 不会加速 prefill（反而要 dequant 开销）。

**Q58. FP8 为什么对 prefill 帮助大？**
prefill 是大矩阵乘，FP8 TensorCore 峰值算力是 BF16 的 2×（H100 上 FP8 ~1979 TFLOPS vs BF16 ~989），且显存带宽占用减半。需要 per-tensor/per-block scale 控制精度。

**Q59. KV cache FP8 的精度风险？**
长上下文注意力分数失真、needle 任务退化、与投机解码的接受率下降。K 比 V 敏感；建议 per-head scale、任务级评估、必要时混合精度。

**Q60. AWQ 与 GPTQ 的区别？**
AWQ：按激活分布找显著通道，保护 1% 的 salient weights，无需反向传播，泛化好；GPTQ：逐层二阶（Hessian）误差补偿，量化误差更小但校准敏感。工程上二者都有成熟 kernel（Marlin 等）。

**Q61. 量化后精度掉了怎么定位？**
① 逐层对比（哪一层误差最大）；② 关掉 KV 量化只看权重；③ 换校准集；④ 检查 embedding/lm_head/首末层是否被量化；⑤ 用任务级指标（代码/数学/JSON）而非只看困惑度。

**Q62. NVFP4 与 MXFP4 的区别？**
NVFP4：16 元素共享一个 FP8 scale（更细）；MXFP4：32 元素共享一个 power-of-2 scale（更通用、跨厂商）。NVFP4 在 NVIDIA SM100 上精度更好，MXFP4 生态更开放。

**Q63. 在线量化 vs 离线量化？**
在线：加载时转换，省去离线步骤、可只量化部分层，但启动慢、依赖运行时校准；离线：精度可控、启动快，但要维护 checkpoint 与 kernel 匹配。

**Q64. 为什么量化 kernel 选错会输出乱码而不报错？**
因为 shape/dtype 都匹配，只是权重按错误布局（GEMM vs WNA16、group 排列）被解读，计算"合法"但语义错误。防御：按 checkpoint 元数据显式分派 kernel + 启动金标准自检 + 数值回归。

### 7.7 投机解码

**Q65. 投机解码为什么能加速？加速比公式？**
因为 target 模型一次前向可以**并行验证** γ 个 draft token，把"γ 次串行 decode"变成"1 次大前向"。期望加速 ≈ `(1-α^(γ+1))/((1-α)(1+γ·c_draft/c_target))`。

**Q66. 为什么必须做 rejection sampling？**
直接接受 draft token 会改变输出分布（偏向 draft 模型）。rejection sampling 保证**最终分布与 target 模型一致**（无偏），这是投机解码"免费加速"的理论前提。

**Q67. EAGLE 与 Medusa 的区别？**
Medusa：在 target 的 hidden state 上接多个解码头，并行预测多个位置；EAGLE：用小自回归头在**特征层**逐步预测，再用 target 验证，接受率更高但结构更复杂。EAGLE-3 进一步提升训练对齐。

**Q68. n-gram 投机适合什么场景？**
输出与输入高度重复：RAG 引用、代码补全、模板化生成、翻译。零训练、零额外显存，命中时收益极高；不适合开放式创作。

**Q69. 高并发下为什么负收益？**
verify 增加总计算量；高并发时系统已从"延迟受限"变为"吞吐受限"，多算的部分没有换来等比例的时间节省，还挤占了 KV 与带宽。

**Q70. 如何调 γ？**
按接受率与负载动态调：低并发/高 α 用 3–5；高并发降到 1–2 或关闭。0.25+ 支持动态预算（在线接受率估计），不要手调死值。

**Q71. 与结构化输出/前缀缓存如何交互？**
draft token 必须满足 grammar（否则验证必失败，浪费算力）；前缀缓存命中会改变 draft 的上下文，需要保证 draft 与 target 看到同样的 KV；这些组合是 bug 高发区，必须回归。

### 7.8 工程、排障与系统设计

**Q72. 线上吞吐突然掉一半，怎么排查？**
结构化回答：① 确认现象（哪个指标、哪些请求、何时开始）；② 对比变更（发布/配置/流量/模型）；③ 分层定位（GPU 利用率→图/backend；CPU→调度/前端；KV→抢占/命中率；网络→NCCL）；④ 复现（固定 workload 压测）；⑤ 回滚验证；⑥ 加监控防复发。

**Q73. 输出乱码怎么定位？**
先关一切优化（eager、无前缀缓存、KV auto）复现；再与 transformers 对比；二分开启量化/前缀缓存/图；检查 chat template 与 RoPE 配置；最后怀疑 kernel 布局（用金标准 prompt 与权重元数据核对）。

**Q74. 多租户怎么做隔离？**
请求级：`cache_salt` 隔离前缀缓存、按租户限流/配额、LoRA 隔离权重；资源级：独立实例或 MIG/分区；安全：prompt 不落盘、日志脱敏、输出水印（0.30 Watermarking）。

**Q75. 设计一个 1 万 QPS 短请求的推理平台。**
小模型（7B/8B）+ 多副本 DP + 每副本 `--api-server-count` 多前端 + 前缀缓存（模板稳定）+ 大 `max_num_seqs` + 关 logprobs + 网关层限流/路由 + 自动扩缩容（按 QPS/队列长度）+ 多级缓存（同 prompt 直接返回）。核心矛盾是 CPU/前端而非 GPU。

**Q76. 设计一个长文档问答服务。**
解析/分块 → embedding → 检索 → 拼 prompt（稳定前缀在前）→ 推理（长上下文：chunked prefill + FP8 KV + 前缀缓存 + PD 分离）→ 引用回填。优化重点：TTFT、KV 容量、文档级缓存复用。

**Q77. PD 分离什么时候值得？怎么算账？**
值得：长 prompt（>8K）+ 短输出、TTFT 敏感、P/D 资源需求差异大。算账：KV 传输量 = tokens × per-token bytes；对比"传输时间"与"同实例互相干扰造成的时间"。带宽不足时不要拆（见 §3.11）。

**Q78. 如何评估一次 vLLM 升级是否安全？**
三层回归（正确性/性能/稳定性）+ 固定 prompt 逐 token 比对 + 压测分位数 + 4h 长跑 + 灰度双版本 + 关注隐性指标（命中率、抢占、图回退）。见 §1.8 检查清单。

**Q79. 国产卡适配 vLLM 的关键难点？**
① 平台抽象与显存/流语义；② 集合通信库（NCCL 替代）；③ attention/量化 kernel 的正确性与性能；④ 图捕获与异步调度支持；⑤ checkpoint 打包格式与 kernel 匹配（静默错误）；⑥ 版本绑定与补丁管理。

**Q80. 你如何跟进 vLLM 的最新进展？**
固定节奏：读每次 Release Notes 的 Highlights + Breaking Changes；跟踪 5 条主线（KV/调度/执行/并行/服务化）而不是追版本号；对每个新特性问三个问题——解决什么瓶颈、代价是什么、我的负载是否受益。

---

### 7.9 面试前 15 分钟速览（Top 15）

| # | 问题 | 一句话答案 | 题号 |
|---|---|---|---|
| 1 | PagedAttention 解决什么 | KV 按固定块分页，消除外部碎片，利用率 20–38% → 90%+ | Q1 |
| 2 | 0.25"删除 PagedAttention"删的是什么 | 删原始 kernel，块式 KV 管理抽象保留 | Q2 |
| 3 | V1 为什么重写 | CPU 开销 + prefill/decode 双路径分叉 | Q3 |
| 4 | V1 为什么去掉 swap | 零额外显存、实现简单、与前缀缓存协同 | Q5 |
| 5 | chunked prefill 收益/代价 | 稳 ITL、降峰值；代价是长 prompt TTFT 变长 | Q28 |
| 6 | 前缀缓存哈希要含什么 | 权重版本/LoRA/多模态/block_size/窗口/拓扑/salt | Q16 |
| 7 | 抢占怎么选受害者 | 低优先级 → 重算便宜 → 晚到达 | Q31 |
| 8 | KV 容量怎么算 | 2×L×H_kv×D×bytes（MLA 只算 latent） | Q13 |
| 9 | TP 通信量 | 每层 2×B×S×H×bytes，跨机延迟吃死 decode | Q47 |
| 10 | EP 为什么贵 | 每层双向 all-to-all，对网络抖动敏感 | Q50 |
| 11 | 投机解码加速比 | (1-α^(γ+1))/((1-α)(1+γ·c_d/c_t))，高并发负收益 | Q65/Q69 |
| 12 | W4A16 为什么快 | decode 带宽受限，权重字节减半 | Q57 |
| 13 | CUDA Graph 收益/失效 | launch 压成一次；动态形状/变长通信会失效 | Q46 |
| 14 | PD 分离何时值得 | 长 prompt + 短输出；算 KV 传输时间 vs 干扰成本 | Q77 |
| 15 | 吞吐掉一半怎么查 | 变更对比 → 分层（GPU/CPU/KV/网络）→ 复现 → 回滚 | Q72 |

---

> 上一篇：[06-改代码实战](06-改代码实战.md) ｜ [返回总览](00-总览.md) ｜ 下一篇：[08-附录](08-附录.md)


# 第 8 章 附录

> [← 总览](00-总览.md)

### A. 术语表

| 术语 | 含义 |
|---|---|
| **PagedAttention** | 分块管理 KV Cache 的内存抽象（原始 kernel 已于 0.25 移除） |
| **block / block_table** | 物理 KV 块 / 逻辑块到物理块的映射表 |
| **slot_mapping** | 每个 token 写入 KV 的物理槽位 |
| **prefix caching** | 基于块哈希的前缀复用 |
| **chunked prefill** | 长 prefill 分块，与 decode 混批 |
| **preemption** | 显存不足时释放某请求的 KV（recompute/swap） |
| **TTFT / TPOT / ITL** | 首 token 时间 / 每输出 token 时间 / 相邻 token 间隔 |
| **goodput** | 满足 SLO 的有效吞吐 |
| **continuous batching** | 以 step 为单位动态组批 |
| **CUDA Graph** | 把 kernel 序列录制为图，一次 replay |
| **piecewise / full graph** | 分段捕获 / 整段捕获 |
| **TP / PP / DP / EP / CP** | 张量/流水/数据/专家/上下文并行 |
| **DCP / PCP** | 解码/预填充上下文并行 |
| **EPLB** | 专家并行负载均衡 |
| **DeepEP** | 面向 MoE 的高性能 all-to-all 通信库 |
| **MLA** | DeepSeek 系多头潜在注意力（压缩 KV） |
| **GQA / MQA** | 分组/多查询注意力（共享 KV head） |
| **MoE** | 混合专家模型 |
| **P/D disaggregation** | 预填充与解码分离部署 |
| **KV Connector** | P/D 与 offload 的 KV 传输抽象 |
| **HiSparse** | 稀疏 MLA 的 host 侧 KV 层 |
| **Fast Start / ipc_cache** | 权重常驻 + CUDA IPC 映射的快速启动 |
| **speculative decoding** | 投机解码（draft + verify） |
| **MTP** | 多 token 预测（模型自带 draft） |
| **DFlash / DSpark** | 2026 年新增的 draft 方法（局部卷积/置信度调度） |
| **xgrammar** | 结构化输出的语法编译后端 |
| **MRv2** | Model Runner V2（0.25+ 默认，0.34 唯一） |
| **vLLM IR** | 0.20 起引入的中间表示，用于自定义算子组合 |

### B. 完整版本时间线

共 102 个发布记录：0.0.1 → 0.34.0，其中 0.2.1 因 TP bug 被 yank（实际可用 101 个）。

| 版本 | 发布日期 | 阶段 | 关键事件 |
|---|---|---|---|
| `0.0.1` | 2023-06-19 | ① 起源 | 首个 PyPI 包（论文配套），PagedAttention + continuous batching |
| `0.1.0` | 2023-06-20 | ① 起源 |  |
| `0.1.1` | 2023-06-22 | ① 起源 |  |
| `0.1.2` | 2023-07-05 | ① 起源 |  |
| `0.1.3` | 2023-08-02 | ① 起源 |  |
| `0.1.4` | 2023-08-25 | ① 起源 | 首个 Linux wheel（cp38–cp311），告别源码编译 |
| `0.1.5` | 2023-09-08 | ① 起源 |  |
| `0.1.6` | 2023-09-08 | ① 起源 |  |
| `0.1.7` | 2023-09-11 | ① 起源 |  |
| `0.2.0` | 2023-09-28 | ① 起源 | 多卡 TP、AWQ/GPTQ 量化、PagedAttention V2 kernel |
| `0.2.1` | 2023-10-16 | ① 起源 |  |
| `0.2.1.post1` | 2023-10-17 | ① 起源 |  |
| `0.2.2` | 2023-11-19 | ① 起源 |  |
| `0.2.3` | 2023-12-03 | ① 起源 | wheel 体积大幅下降（依赖拆分） |
| `0.2.4` | 2023-12-11 | ① 起源 |  |
| `0.2.5` | 2023-12-14 | ① 起源 |  |
| `0.2.6` | 2023-12-17 | ① 起源 |  |
| `0.2.7` | 2024-01-04 | ① 起源 |  |
| `0.3.0` | 2024-01-31 | ② 生产化 | 多模态/模型覆盖扩展，投机解码起步 |
| `0.3.1` | 2024-02-17 | ② 生产化 |  |
| `0.3.2` | 2024-02-21 | ② 生产化 |  |
| `0.3.3` | 2024-03-01 | ② 生产化 |  |
| `0.4.0` | 2024-03-31 | ② 生产化 | 自动前缀缓存、chunked prefill 推进、LLaVA、自定义 AllReduce |
| `0.4.0.post1` | 2024-04-03 | ② 生产化 |  |
| `0.4.1` | 2024-04-24 | ② 生产化 |  |
| `0.4.2` | 2024-05-05 | ② 生产化 |  |
| `0.4.3` | 2024-06-01 | ② 生产化 | 生产稳定性修复 |
| `0.5.0` | 2024-06-11 | ③ 吞吐 | 吞吐与 kernel 优化，VLM 支持扩展 |
| `0.5.0.post1` | 2024-06-14 | ③ 吞吐 |  |
| `0.5.1` | 2024-07-06 | ③ 吞吐 |  |
| `0.5.2` | 2024-07-15 | ③ 吞吐 |  |
| `0.5.3` | 2024-07-23 | ③ 吞吐 | 0.6.0 的对比基线 |
| `0.5.3.post1` | 2024-07-23 | ③ 吞吐 |  |
| `0.5.4` | 2024-08-05 | ③ 吞吐 |  |
| `0.5.5` | 2024-08-23 | ③ 吞吐 |  |
| `0.6.0` | 2024-09-05 | ③ 吞吐 | 官方口径 2× 吞吐：多步调度 + 异步输出处理 + FlashInfer FP8 KV |
| `0.6.1` | 2024-09-11 | ③ 吞吐 |  |
| `0.6.1.post1` | 2024-09-13 | ③ 吞吐 |  |
| `0.6.1.post2` | 2024-09-13 | ③ 吞吐 |  |
| `0.6.2` | 2024-09-25 | ③ 吞吐 |  |
| `0.6.3` | 2024-10-14 | ③ 吞吐 |  |
| `0.6.3.post1` | 2024-10-17 | ③ 吞吐 |  |
| `0.6.4` | 2024-11-15 | ③ 吞吐 | 多模态与模型覆盖继续扩张 |
| `0.6.4.post1` | 2024-11-15 | ③ 吞吐 |  |
| `0.6.5` | 2024-12-18 | ③ 吞吐 |  |
| `0.6.6` | 2024-12-27 | ③ 吞吐 |  |
| `0.6.6.post1` | 2024-12-27 | ③ 吞吐 |  |
| `0.7.0` | 2025-01-27 | ④ V1 | V1 Alpha（VLLM_USE_V1=1）、torch.compile 默认、LLM.sleep/wake_up |
| `0.7.1` | 2025-02-01 | ④ V1 |  |
| `0.7.2` | 2025-02-06 | ④ V1 |  |
| `0.7.3` | 2025-02-20 | ④ V1 |  |
| `0.8.0` | 2025-03-18 | ④ V1 | V1 成为默认引擎；结构化输出/LoRA/PP 增强；可插拔调度器 |
| `0.8.1` | 2025-03-19 | ④ V1 |  |
| `0.8.2` | 2025-03-25 | ④ V1 |  |
| `0.8.3` | 2025-04-06 | ④ V1 |  |
| `0.8.4` | 2025-04-15 | ④ V1 |  |
| `0.8.5` | 2025-04-28 | ④ V1 | 稳定性与模型修复 |
| `0.8.5.post1` | 2025-05-02 | ④ V1 |  |
| `0.9.0` | 2025-05-28 | ④ V1 | PyTorch 2.7 + CUDA 12.8（破坏性）；Blackwell 优化；DP/EP/PD 起步 |
| `0.9.0.1` | 2025-05-30 | ④ V1 |  |
| `0.9.1` | 2025-06-10 | ④ V1 |  |
| `0.9.2` | 2025-07-08 | ④ V1 |  |
| `0.10.0` | 2025-07-25 | ⑤ 异步 | V0 清理开始；实验性异步调度；MLA FlashInfer ragged prefill |
| `0.10.1` | 2025-08-19 | ⑤ 异步 |  |
| `0.10.1.1` | 2025-08-20 | ⑤ 异步 |  |
| `0.10.2` | 2025-09-13 | ⑤ 异步 |  |
| `0.11.0` | 2025-10-04 | ⑤ 异步 | V0 引擎彻底移除；CUDA Graph 默认 FULL_AND_PIECEWISE |
| `0.11.1` | 2025-11-19 | ⑤ 异步 |  |
| `0.11.2` | 2025-11-20 | ⑤ 异步 | DeepSeek-V3.2、Qwen3-VL/Next 等支持 |
| `0.12.0` | 2025-12-03 | ⑤ 异步 | PyTorch 2.9/CUDA 12.9；EAGLE 多步 CUDA Graph；ROCm 扩展 |
| `0.13.0` | 2025-12-19 | ⑤ 异步 | 稳定性与模型覆盖 |
| `0.14.0` | 2026-01-20 | ⑥ 换代 | 异步调度默认；gRPC 入口；--max-model-len auto |
| `0.14.1` | 2026-01-24 | ⑥ 换代 |  |
| `0.15.0` | 2026-01-29 | ⑥ 换代 | 持续迭代（模型/性能） |
| `0.15.1` | 2026-02-05 | ⑥ 换代 |  |
| `0.16.0` | 2026-02-26 | ⑥ 换代 | 异步调度 + PP（+30.8% 吞吐）；Realtime API；NCCL 权重同步 |
| `0.17.0` | 2026-03-07 | ⑥ 换代 |  |
| `0.17.1` | 2026-03-11 | ⑥ 换代 |  |
| `0.18.0` | 2026-03-20 | ⑥ 换代 | gRPC 正式；GPU-less Render；KV offload 改进；弹性 EP |
| `0.18.1` | 2026-03-31 | ⑥ 换代 |  |
| `0.19.0` | 2026-04-03 | ⑥ 换代 |  |
| `0.19.1` | 2026-04-18 | ⑥ 换代 |  |
| `0.20.0` | 2026-04-27 | ⑥ 换代 | DeepSeek V4；CUDA 13 默认；FA4 MLA prefill 默认；TurboQuant 2-bit KV；在线量化 |
| `0.20.1` | 2026-05-03 | ⑥ 换代 |  |
| `0.20.2` | 2026-05-10 | ⑥ 换代 |  |
| `0.21.0` | 2026-05-15 | ⑥ 换代 | DeepSeek V4 周边与稳定性 |
| `0.22.0` | 2026-05-29 | ⑦ Runner V2 | MRv2 对 Qwen3 dense 默认；Rust 前端；多级 KV offload；batch-invariant 优化 |
| `0.22.1` | 2026-06-05 | ⑦ Runner V2 |  |
| `0.23.0` | 2026-06-13 | ⑦ Runner V2 |  |
| `0.24.0` | 2026-06-30 | ⑦ Runner V2 | MRv2 支持量化模型；Streaming Parser Engine；DeepEP v2；device_ids 取代 CUDA_VISIBLE_DEVICES |
| `0.25.0` | 2026-07-11 | ⑦ Runner V2 | MRv2 默认（全部 dense）；PagedAttention 原始实现移除；TLI/DSpark/DFlash |
| `0.25.1` | 2026-07-14 | ⑦ Runner V2 | ECU / 国产卡常见基线版本（本手册源码路径以此为准） |
| `0.26.0` | 2026-07-25 | ⑦ Runner V2 | 按 KV group 选 attention backend；sliding window 成显式能力；fp32 lm_head |
| `0.27.0` | 2026-08-10 | ⑦ Runner V2 |  |
| `0.27.1` | 2026-08-11 | ⑦ Runner V2 |  |
| `0.28.0` | 2026-08-26 | ⑦ Runner V2 | DCP；max_num_batched_tokens 默认 16384；bitsandbytes 转插件 |
| `0.29.0` | 2026-09-09 | ⑦ Runner V2 |  |
| `0.30.0` | 2026-09-22 | ⑦ Runner V2 | Fast Start（ipc_cache）；HiSparse；Watermarking；PCP+DCP |
| `0.31.0` | 2026-10-05 | ⑦ Runner V2 |  |
| `0.32.0` | 2026-10-18 | ⑦ Runner V2 | NVFP4 KV 全平台；GLM-5.x 默认 FP8 KV；vllm download-kernels；api-server-count |
| `0.33.0` | 2026-11-02 | ⑦ Runner V2 | 持续迭代 |
| `0.34.0` | 2026-11-16 | ⑦ Runner V2 | 可扩展 KV 默认；MRv1 删除；block_size 默认 64；1061 commits/412 贡献者 |


### C. CLI 速查

```bash
# 服务
vllm serve <model> --tensor-parallel-size 8 --max-model-len 32768 \
  --gpu-memory-utilization 0.9 --enable-prefix-caching --async-scheduling

# 离线批量
python -c "from vllm import LLM; llm=LLM(model='<m>'); print(llm.generate(prompts, SamplingParams(max_tokens=256)))"

# 压测（开环）
vllm bench serve --model <m> --request-rate 50 --num-prompts 2000 \
  --input-len 1024 --output-len 256 --percentile-metrics ttft,tpot,itl

# 压测（闭循环）
vllm bench serve --model <m> --max-concurrency 128 --num-prompts 2000

# 环境信息
vllm collect-env

# 内核预编译 / 预热（0.32+）
vllm download-kernels
vllm preload --preload-autotune
```

### D. 验收与压测清单

```text
[ ] 负载模型：真实长度分布（p50/p95/p99）+ 真实到达过程（Poisson）
[ ] 指标：TTFT/TPOT/ITL 的 p50/p95/p99 + 吞吐 + goodput
[ ] 资源：GPU 利用率、显存、KV 使用率、PCIe/NVLink 利用
[ ] 稳定性：4h 长跑、抢占注入、故障注入（KV 传输失败/网卡抖动）
[ ] 正确性：金标准 prompt 逐 token、任务级指标（代码/JSON/数学）
[ ] 回归基线：与上一版本同 workload 对比，记录 diff
```

### E. 参考资料

- 论文：*Efficient Memory Management for Large Language Model Serving with PagedAttention*（SOSP 2023）
- 官方文档：<https://docs.vllm.ai>
- 官方博客（V1 设计、P/D、量化等）：<https://blog.vllm.ai>
- 代码与 Release Notes：<https://github.com/vllm-project/vllm/releases>
- 版本记录（本文日期来源）：<https://pypi.org/project/vllm/#history>

---

> **文档结束**。建议每季度对照最新 Release Notes 更新：§1.6（时代五）、§3.4（后端矩阵）、§4.3（参数默认值）、§7（面试题）。版本会变，**五条主线的权衡不会变**。

---

> 上一篇：[07-面试题库](07-面试题库.md) ｜ [返回总览](00-总览.md)
