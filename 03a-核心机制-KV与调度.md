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
