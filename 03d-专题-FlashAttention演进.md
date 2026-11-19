# 第 3 章 专题 A：FlashAttention 演进与 Attention 内核

> [← 总览](00-总览.md) ｜ 下一章：[03e-专题-KV池化与Mooncake](03e-专题-KV池化与Mooncake.md)

### 3d.1 先说结论：为什么 attention 内核值得单独一章

- **它是唯一同时被"形状、精度、硬件、KV 布局"四重约束的算子**：prefill 是长 query 的因果矩阵乘，decode 是 q_len=1 的"读 KV"操作，两者最优 kernel 完全不同。
- **它决定了 decode 的下限**：decode 阶段 80% 的时间在搬 KV，attention kernel 的访存效率直接换算成 TPOT。
- **它是最容易"选错还不报错"的地方**：同一份权重，backend 选错可能慢 2×，也可能精度悄悄退化。

一句话：**PagedAttention 决定"KV 存在哪、怎么找"，FlashAttention 决定"找到之后怎么算得快"**，两者正交，必须分开理解。

### 3d.2 问题的本质：标准 attention 是 IO-bound

朴素的 `softmax(QKᵀ)V` 在 GPU 上会经历：

```
① 算 S = QKᵀ        → 写 N×N 矩阵到 HBM（N=序列长度）
② 读 S，做 softmax  → 再写回 N×N
③ 读 S，乘 V        → 再读一次 N×N
```

**三次 N² 级别的 HBM 读写**。当 N=8K、head_dim=128、BF16 时，单个 head 的 S 矩阵就是 128 MB；GPU 的 HBM 带宽（~3 TB/s）看起来很高，但 N² 的增长会让 attention 变成**纯带宽问题**——这就是 flash attention 系列解决的核心矛盾。

**FlashAttention 的三板斧**（四代都没变）：

1. **Tiling（分块）**：把 Q/K/V 切成能放进 SRAM 的小块，块内完成计算。
2. **Online softmax（在线归一化）**：不等整行算完，边算边维护 `(max, sum)`，用 rescale 修正历史结果。
3. **Recompute（重计算）**：反向传播不存 N×N 的中间结果，用 O(N) 的统计量重算，显存从 O(N²) 降到 O(N)。

### 3d.3 四代演进：差异到底在哪

| 代际 | 时间 | 目标硬件 | 核心改进 | 相对收益（官方/论文口径） | 关键限制 |
|---|---|---|---|---|---|
| **FA1** | 2022 | Ampere（A100） | 分块 + online softmax + 重计算，彻底不落地 N×N | 比 PyTorch 2–4×，显存 O(N) | 仅前向；并行度不足；非 matmul 开销占比高 |
| **FA2** | 2023 | Ampere / Ada | ① 减少非 matmul FLOPs（延迟 rescale）；② 沿**序列维**并行；③ causal mask 跳过整块；④ 支持 head_dim 256 | 约 2× FA1，可达 A100 峰值的 50–73% | 仍是"每 head 独立"的并行；Hopper 新指令没用上 |
| **FA3** | 2024 | **Hopper（H100/H200）** | ① TMA（异步张量搬运）；② WGMMA（warpgroup 级矩阵乘）；③ **warp specialization**（生产者/消费者分工）；④ FP8；⑤ 更好的 varlen | 约 1.5–2× FA2 | 强绑定 Hopper；移植性差；早期 paged KV 支持不完整 |
| **FA4** | 2025–2026 | **Blackwell（SM100）** | 面向新架构的指令与流水；**paged KV** 与 **head_dim 512（MLA）**；FP8/低精度路径 | 官方口径在 Blackwell 上显著加速（随 shape/精度变化，以 release note 为准） | 新硬件专属；生态仍在补齐 |

> **vLLM 侧的落地事实**（比论文更重要）：0.20 起 **FA4 成为 MLA prefill 默认后端**，支持 head_dim 512 与 paged KV（SM90+）。选型时不要背"FA4 最强"，要看**你的卡、你的 head_dim、你的 KV 布局**是否在支持矩阵里。

#### FA2 相比 FA1 到底改了什么（面试最常问）

1. **不再对每个 block 做一次 rescale**：FA1 在每个 K/V block 后都更新输出；FA2 把 rescale 推迟到"需要时"，减少非 matmul 的逐元素运算——这是 2× 加速的主要来源。
2. **并行维度从 (batch × heads) 扩展到 (batch × heads × 序列块)**：A100 有 108 个 SM，长序列下 FA1 的并行度不够。
3. **causal mask 的块级跳过**：对角线以上的块直接不算，理论上省一半 FLOPs。
4. **支持 head_dim 到 256**：适配更多模型结构。

#### FA3 为什么要"warp specialization"

Hopper 上矩阵乘单元（WGMMA）和搬运单元（TMA）是异步的。如果所有 warp 都既搬运又计算，就会出现"等数据"的流水气泡。FA3 把 warp 拆成两类：

```
producer warp：发 TMA 请求，把 K/V 块搬进 shared memory
consumer warp：发 WGMMA，算 QKᵀ 与 PV，做 softmax
两者通过 barrier 同步 → 形成软件流水线（software pipelining）
```

这是"用架构特性换 1.5–2×"的典型案例，也解释了**为什么 FA3 不能简单移植到别的卡**。

### 3d.4 与 PagedAttention 如何结合

FlashAttention 原生假设 KV 是**连续布局**，而 vLLM 的 KV 是**分页**的。结合点在 kernel 内部：

```
输入：
  q            [num_tokens, num_heads, head_dim]
  k_cache      [num_blocks, block_size, num_kv_heads, head_dim]
  block_table  [num_reqs, max_blocks]          ← 逻辑块 → 物理块
  seq_lens     [num_reqs]
  slot_mapping [num_tokens]                    ← 写入位置

kernel 内：
  for each kv_block in block_table[req]:
      k = gather(k_cache, physical_block_id)   ← 间接寻址
      累加 QKᵀ / online softmax / PV
```

由此带来四个工程约束（也是高频 bug 来源）：

| 约束 | 后果 |
|---|---|
| `block_table` 必须是**每请求独立**的 | 共享前缀时多条请求指向同一物理块，写时要注意 COW/只读 |
| GQA 下 KV head 被多个 Q head 共享 | kernel 要按 `num_queries_per_kv` 做 head 广播，否则算错 |
| varlen（变长 batch） | `query_start_loc` 必须严格单调，chunked prefill 下容易错位 |
| 投机解码的 tree mask | 多 query 的因果 mask 不是简单三角，kernel 需专门支持 |

**为什么 decode 比 prefill 更依赖这些**：prefill 还能靠大矩阵乘摊薄开销，decode 每步只有 1 个 query，**几乎全部时间在按 block_table 随机读 KV**，所以 paged kernel 的 gather 效率、block 大小（0.34 默认 64）、KV 精度（FP8/NVFP4）直接决定 TPOT。

### 3d.5 vLLM 里的内核分工（选型速查）

| 内核 | 强项 | 什么时候选它 | 什么时候别用 |
|---|---|---|---|
| **FlashAttention 2/3/4** | 通用 prefill + decode，生态最广 | 默认首选；MLA prefill 用 FA4（SM90+） | 极低延迟 decode（FlashInfer 常更快） |
| **FlashInfer** | decode 专用优化、JIT 生成、paged/ragged、FP8、Blackwell | 低延迟 decode、Blackwell、需要 FP8 KV | 首次启动要 autotune（可用 `vllm preload --preload-autotune` 预热） |
| **FlashMLA** | DeepSeek 系 MLA decode，SM90/SM100，fp8_ds_mla | DeepSeek/Qwen-MoE 等 MLA 模型 decode | 非 MLA 模型 |
| **Triton MLA / Cutlass MLA** | 可移植、可改 | 新卡适配、需要魔改 kernel | 追极致性能 |
| **Triton Attention** | 能跑、能改 | 没有 FA 的硬件、研究 | 生产（性能一般） |
| **FlexAttention** | 自定义 mask/稀疏 | 实验性 attention 变体 | 生产默认路径 |
| **TurboQuant** | 2-bit KV、4× 容量 | 显存极度紧张、可接受精度损失 | 精度敏感任务 |

**排障第一步永远是确认实际后端**：启动日志会打印 `Using ... attention backend`；不要假设配置生效。

### 3d.6 兼容性坑清单

1. **head_dim 不匹配**：FA2 支持到 256，MLA 的 512 需要 FA4/FlashMLA；报错通常是"unsupported head size"。
2. **FP8 KV 与 kernel**：不是所有 backend 都支持 FP8 KV；混用会 fallback 到慢路径或直接报错。
3. **chunked prefill**：部分 backend 的 prefix 处理不完整，表现为"长 prompt 结果错乱"，而不是崩溃。
4. **投机解码**：tree attention 需要 kernel 支持自定义 mask；否则 draft token 的注意力会串。
5. **滑动窗口/混合模型**：SWA 需要 backend 显式支持（0.26 起成为显式能力），否则窗口外的 KV 会被错误读到。

### 3d.7 性能验证方法

```bash
# 1) 确认后端与图形
VLLM_LOGGING_LEVEL=DEBUG vllm serve <m> 2>&1 | grep -i "attention backend\|cudagraph"

# 2) 单 kernel 级对比（隔离调度与框架开销）
#    固定 batch/seq_len，比较 --enforce-eager 下的 decode 时间
# 3) 数值对齐：与参考实现比较（BF16 容差 1e-2，FP8 5e-2）
# 4) 用 nsys 看 attention kernel 占比与 HBM 吞吐
nsys profile -t cuda,nvtx -o attn vllm serve <m> --enforce-eager
```

**判断标准**：decode 阶段 attention kernel 若占 step 时间 > 50%，说明是 KV 访存瓶颈——优先查 block 大小、KV 精度、是否命中了正确的 paged kernel，而不是换卡。

### 3d.8 面试追问（速答）

- **FA2 比 FA1 快在哪？** 减少 rescale 次数、序列维并行、causal 块跳过；本质是把"非 matmul 开销"压下去。
- **FA3 为什么绑定 Hopper？** 依赖 TMA/WGMMA/warp specialization 这些 Hopper 专属能力，其他架构没有等价指令。
- **PagedAttention 和 FlashAttention 冲突吗？** 不冲突。前者是 KV 的内存布局，后者是计算方式；paged kernel 在内部做 block_table gather。
- **decode 为什么不用普通 FA？** decode 的 q_len=1，瓶颈是随机读 KV，专用 paged-decode kernel（FlashInfer/FlashMLA）更合适。
- **MLA 的 attention 特殊在哪？** KV 是压缩 latent，需要先"解压"再算；head_dim 大（512），对 kernel 的 SRAM 与寄存器压力高。

---

> 上一篇：[03c-核心机制-量化与服务化](03c-核心机制-量化与服务化.md) ｜ [返回总览](00-总览.md) ｜ 下一篇：[03e-专题-KV池化与Mooncake](03e-专题-KV池化与Mooncake.md)
