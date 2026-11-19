# 第 3 章 专题 B：KV 池化、Mooncake 与 KV 压缩

> [← 总览](00-总览.md) ｜ 下一章：[03f-专题-框架横向对比](03f-专题-框架横向对比.md)

### 3e.1 为什么 KV 要"池化"

KV Cache 是推理系统里**最贵、最可复用、也最难管理**的资源。单机视角下它只是显存；集群视角下它有三个结构性浪费：

| 浪费 | 现象 | 量化（经验值） |
|---|---|---|
| **跨实例重复** | 同一份 system prompt / 文档在 8 个副本上各算一遍 | 命中率每提升 10pt，prefill 算力省 ~10% |
| **跨请求重复** | 多轮对话每轮重算历史 | 第 10 轮时 90% 的 token 是重复的 |
| **时间维度浪费** | 昨天算过的长文档，今天冷启动又要重算 | 长文档 RAG 的 TTFT 主要花在这 |

把 KV 从"某个 GPU 的私有内存"提升为**集群级可寻址的缓存**（池化），就是 Mooncake / LMCache / NIXL 这类系统的共同目标。它解决的不是"算得快"，而是"**不用重算**"。

### 3e.2 Mooncake 架构

Mooncake 是月之暗面（Moonshot AI）提出的 **KVCache 为中心的分离式架构**，论文标题即 *A KVCache-centric Disaggregated Architecture for LLM Serving*。核心思想：**把 KV Cache 当作一等存储资源来调度**，而不是推理的副产品。

#### 组件

| 组件 | 职责 | 关键设计 |
|---|---|---|
| **Conductor** | 全局请求调度 | 感知 KV 缓存位置与负载，决定"发到哪个 prefill / decode 实例" |
| **KVCache Pool** | 分层缓存（GPU / CPU DRAM / SSD） | 按块管理，支持跨实例共享与淘汰 |
| **Transfer Engine** | KV 高速传输 | 多协议（RDMA / NVLink / TCP）、多网卡聚合、零拷贝、批量传输 |
| **Master** | 元数据与路由 | 记录"哪个块在哪个节点"，提供全局视图 |

#### 两种工作模式

```
① P2P（点对点）：Prefill 实例算完 → 直接 RDMA 推给指定 Decode 实例
   优点：路径最短、延迟最低；缺点：需要调度器精确配对

② Store（共享池）：Prefill 写入 KVCache Pool → Decode 按需拉取
   优点：解耦、可复用、支持多消费者；缺点：多一跳、池容量与带宽成为瓶颈
```

#### 请求流（Store 模式）

```
Client → Conductor
   ├─ 查 Master：这段前缀的 KV 在池里吗？
   │     ├─ 命中 → 直接把请求发给持有该 KV 的 Decode 实例（跳过 prefill）
   │     └─ 未命中 → 发到 Prefill 实例算 KV → 写入 Pool → Decode 拉取
   └─ 返回流式结果
```

#### 收益口径（论文/官方，随负载变化）

- 在 Kimi 的真实负载上，相比基线系统**吞吐最高提升约 5×**（论文口径 525%）；
- 在满足 SLO 的前提下**可多服务约 75% 的请求**；
- 核心来源：**长上下文请求的 prefill 被大量跳过**，以及 P/D 资源按瓶颈独立扩缩。

> 注意：这些数字强依赖"前缀重复度"和"网络带宽"。前缀重复度低（开放域聊天）时收益会大幅缩水——这也是评估 PD/池化方案时最容易踩的坑。

### 3e.3 在 vLLM 里怎么接

vLLM 通过 **KV Connector** 抽象接入外部 KV 系统（接口见 §6.7），Mooncake 相关有三条路径：

| 路径 | 用途 | 配置形态（示意，以你的版本为准） |
|---|---|---|
| **MooncakeConnector（P2P）** | P/D 分离的点对点传输 | `--kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer/kv_consumer"}'` |
| **MooncakeStore** | 写入共享 KVCache 池，跨实例复用 | connector 指向 Store 地址，配合 prefix-aware 路由 |
| **Mooncake 磁盘卸载** | 把冷 KV 下沉到 SSD（0.22+） | 作为多级 offload 的二级存储实现 |

**最小可用拓扑**（单机验证）：

```
P 实例（--kv-role producer） ──RDMA──> D 实例（--kv-role consumer）
        │                                   ▲
        └────── Mooncake Store（可选）──────┘
```

**验收要点**：

1. KV 传输失败能**回退到本地重算**（0.25+ 有 KV Load Failure Recovery 测试用例可参考）；
2. 逐层流水传输，而不是等整个 prefill 结束再传；
3. 全局前缀命中的 token 数与路由决策一致（否则会出现"路由到 A、KV 在 B"的抖动）。

### 3e.4 同类方案对比

| 方案 | 定位 | 传输 | 缓存层 | 与 vLLM 关系 | 适合 |
|---|---|---|---|---|---|
| **Mooncake** | KVCache 中心化 + PD 分离 | 自研 Transfer Engine（RDMA/NVLink） | GPU/DRAM/SSD 分层池 | Connector + Store + 磁盘卸载 | 自建大规模集群、长上下文 |
| **LMCache** | 独立 KV 缓存层 | 多种后端（本地/Redis/S3/Mooncake） | CPU/磁盘/远端 | 官方 connector，集成度高 | 想快速加"KV 缓存服务" |
| **NIXL** | 通用传输库（NVIDIA） | RDMA/NVLink/GDS | 不管缓存，只负责搬 | Dynamo/llm-d 的底座 | 自研编排层 |
| **FlexKV** | 灵活 KV 连接（0.25 示例） | 可插拔 | 可插拔 | 示例级 | 学习与定制 |
| **vLLM 原生 KV Offload** | 单机分层 | PCIe/NVMe | GPU→CPU→磁盘 | 内建（0.18/0.22+） | 单机长会话、不想引入外部组件 |

**选型直觉**：单机先试原生 offload；多实例共享前缀上 LMCache（接入最快）；要做 P/D + 全局池化且有 RDMA 集群，再上 Mooncake。

### 3e.5 全局前缀缓存与路由

池化之后，**"把请求发到谁"比"算得多快"更重要**。前缀感知路由的核心：

```
对每个请求：
  1. 取 prompt 的块级哈希序列（与 §3.3 一致）
  2. 在路由表中查"哪些实例/池持有这些块"
  3. 选命中块数最多的目标（tie-break 用负载）
  4. 预测命中长度 → 决定是否值得跨机传输（传输时间 vs 重算时间）
```

**命中率的三层影响因素**：

| 层 | 因素 | 优化 |
|---|---|---|
| 前缀本身 | system prompt 是否稳定、模板是否带随机量 | 模板工程（§4.4 场景 A） |
| 缓存容量 | 池大小、淘汰策略、块粒度 | 分层存储、partial block caching |
| 路由质量 | 是否前缀感知、是否抖动 | 一致性哈希 + 负载感知，避免"路由漂移" |

**关键判断式**（决定"传还是算"）：

```
传输收益 = 跳过的 prefill 时间 − (KV 字节 / 有效带宽)
若为负 → 不要传，本地重算更便宜
```

### 3e.6 容量与收益模型

```
单请求 KV 字节 = prompt_tokens × per_token_bytes          （§3.1.2）
池命中可省算力 ≈ 命中的 prompt_tokens × 2 × 参数量 FLOPs
池所需容量 ≈ 活跃前缀数 × 平均前缀长度 × per_token_bytes × 副本系数

例：32B 模型、4 卡 TP、平均前缀 8K、活跃前缀 2000 条
  per_token ≈ 40 KB（见 §3.1.2 例）→ 单份 8K ≈ 320 MB
  2000 × 320 MB ≈ 640 GB  → 必须分层（DRAM/SSD），不能全放显存
```

**成本账**：

- 池化的存储成本（DRAM/SSD）远低于 GPU 显存；
- 但**传输带宽**是硬约束：100 Gbps 下 2.5 GB 的 KV 要 0.2 s（§3.11.3），所以池化只对"大前缀 + 高复用"划算。

### 3e.7 KV 压缩与稀疏：另一条省显存的路

池化解决"放不下"，压缩解决"装更多"。三类手段要分清：

| 类别 | 代表方法 | 保留什么 | 收益 | 代价 | vLLM 主线对应 |
|---|---|---|---|---|---|
| **驱逐/稀疏选择** | StreamingLLM、H2O、SnapKV、Quest | 一部分 token 的 KV | 显存 ↓、decode 带宽 ↓ | 长程召回损失，需评估 | 主线未内置（DSA 除外），多用于研究/定制 |
| **稀疏注意力（模型侧）** | DeepSeek **DSA**（V3.2-Exp 起） | indexer 选 top-k token | 长上下文算力大幅下降 | 依赖模型训练，kernel 复杂 | 稀疏 MLA 后端已支持 |
| **量化** | FP8 KV、NVFP4 KV、**TurboQuant 2-bit** | 全部 KV，但更低精度 | 容量 ×2 / ×4 | 精度衰减（长文最敏感） | 内建，`--kv-cache-dtype` |
| **分层卸载** | KV Offload、**HiSparse** | 全部 KV，但不在显存 | 等效容量 ↑ | 延迟 ↑、带宽依赖 | 内建（0.18–0.30） |

**各方法的要点**：

- **StreamingLLM**：保留开头几个"attention sink" + 最近窗口，显存恒定；适合无限流式对话，但会忘掉中间内容。
- **H2O**：按累计注意力分数保留"heavy hitters"，约 20% 预算可维持多数任务质量。
- **SnapKV**：在 prompt 末尾开"观察窗口"，按 head 选出重要 KV 再压缩；偏 prefill 侧压缩。
- **Quest**：query-aware 的**页级** top-k 选择（与 vLLM 的 block 天然契合），是"稀疏 + 分页"的代表。
- **DSA**：模型自带稀疏注意力（lightning indexer + top-k），推理时只需算被选中的 token；这是 2026 年长上下文的主流路线之一。
- **TurboQuant**：不是驱逐而是量化到 2-bit，容量 ×4，适合"显存极度紧张 + 可接受精度损失"。

> **面试提醒**：不要把这些混为一谈。**驱逐会丢信息（不可逆），量化只是加噪声（可评估），卸载只是换介质（延迟换容量）**。

### 3e.8 落地 checklist 与坑

```text
[ ] 先量化收益：前缀重复度、平均长度、跨实例重合率（没有数据不要上池化）
[ ] 传输带宽实测：RDMA/NVLink 的 all-to-all 与 P2P 基线（§3.9.2）
[ ] 失败回退：KV 拉取失败能否本地重算（否则一次抖动 = 一批请求失败）
[ ] 一致性：块哈希要含 tokenizer/模板/模型版本（§3.3.1），跨实例必须同构
[ ] 隔离：多租户必须 cache_salt，池化会放大侧信道风险
[ ] 淘汰策略：池是全局 LRU 还是按租户配额？防止一个大租户冲掉所有人
[ ] 观测：命中率、传输字节/耗时、回退率、P/D 各自利用率
```

**高频坑**：

1. **拓扑不一致**：P 用 TP=8、D 用 TP=4，KV 分片方式不同，直接算错——必须同构或显式重分片。
2. **版本漂移**：P 与 D 的 vLLM 版本不同，KV 布局/量化格式变化 → 脏读。
3. **路由抖动**：请求在实例间反复漂移，命中率反而低于不池化。
4. **传输打满网卡**：KV 传输与 NCCL 抢带宽，导致 TPOT 抖动（要给 RDMA 做 QoS/分平面）。

### 3e.9 面试速答

- **Mooncake 的核心创新？** 把 KVCache 当作集群级存储来调度（Conductor + Pool + Transfer Engine），让长上下文请求跳过 prefill。
- **池化一定更快吗？** 不一定，取决于前缀重复度与传输带宽；短 prompt 或低重复场景不划算。
- **KV 压缩和 KV 量化区别？** 压缩（驱逐/稀疏）丢 token 信息，量化只降精度；前者不可逆，后者可评估。
- **为什么 DSA 类稀疏注意力这两年重要？** 长上下文下 attention 是 O(N²)，稀疏化把计算降到"选中 token"级别，是模型与 kernel 协同设计的方向。

---

> 上一篇：[03d-专题-FlashAttention演进](03d-专题-FlashAttention演进.md) ｜ [返回总览](00-总览.md) ｜ 下一篇：[03f-专题-框架横向对比](03f-专题-框架横向对比.md)
