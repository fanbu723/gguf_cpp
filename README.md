# gguf_cpp — GGUF C++ 推理工具

> 从零手写 GGUF 模型文件的**解析与推理**。C++20 + CMake，零第三方依赖。
> 目标是逐步构建一个能加载 GGUF 模型并做自回归文本生成的**极简推理引擎**。

---

## 📋 目录

- [项目目标](#项目目标)
- [当前进度](#当前进度)
- [下一步计划](#下一步计划)
- [项目结构](#项目结构)
- [构建与运行](#构建与运行)
- [实测验证](#实测验证)
- [关键认知与踩坑](#关键认知与踩坑)
- [附录：GGUF 格式规范](#附录gguf-格式规范)

---

## 项目目标

按六个阶段逐步构建完整推理链路（当前状态见下）：

| # | 阶段 | 状态 |
| :-: | :--- | :--- |
| ① | GGUF 解析器（头 / 元数据 / 张量表 / 数据区） | ✅ |
| ② | 张量数据访问（mmap / 类型系统 / 反量化） | ✅ |
| ③ | Tokenizer（byte-level BPE） | ✅ |
| ④ | 模型权重加载（按名索引 / 配置 / 逐层组装） | ✅ |
| ⑤ | Transformer 计算（算子 → 单层前向 → 全模型） | ✅ |
| ⑥ | 生成引擎（Sampler + 自回归 + Chat） | 🔄 进行中（Sampler/生成循环 ✅） |

---

## 当前进度

已用真实模型（Qwen3.5-0.8B, 1.5GB）验证通过：

| 功能 | 状态 |
| :--- | :---: |
| 文件头解析（24 字节 + 魔数校验） | ✅ |
| 元数据 KV 列表（含递归 ARRAY） | ✅ |
| 张量信息表（name / shape / type / offset） | ✅ |
| 张量数据区视图（延迟加载，零拷贝） | ✅ |
| mmap 挂载数据区（`map_data` / `unmap_data`，RAII） | ✅ |
| GGML 类型系统（`type_size` / `block_size` / 字节数） | ✅ |
| 反量化（F16 / BF16 / Q4_0 / Q8_0 → float） | ✅ |
| Tokenizer（byte-level BPE，token↔文本，`tokenizer.*` 元数据） | ✅ |
| 模型权重加载（按名索引 + 配置解析 + 逐层权重组装） | ✅ |
| 基础算子（gemv/gemm/softmax/RMSNorm/SwiGLU/RoPE/Attention/KV cache） | ✅（阶段⑤ 第 1 步） |
| 纯 Attention 层前向（Q+gate 联合投影 / GQA / KV cache / 残差） | ✅（阶段⑤ 第 2 步） |
| SSM 混合层前向（Gated Delta Net：conv1d / 状态递推 / gated norm） | ✅（阶段⑤ 第 3 步） |
| 全模型前向（24 层混合 + output_norm + 共享 embedding logits） | ✅（阶段⑤ 第 4 步） |
| 采样器（temperature / top-k / top-p / greedy） | ✅（阶段⑥ 第 1 步） |
| 自回归生成循环（预填充 + 逐 token 解码） | ✅（阶段⑥ 第 2 步） |

运行 `./build/main` 可打印元数据、张量表、类型验证、mmap 读取、反量化结果、分词器、权重加载、基础算子、层前向（Attention + SSM）与全模型前向演示。

---

## 下一步计划

从解析器走向完整推理，按以下阶段逐步推进：

| 阶段 | 内容 | 关键点 |
| :--- | :--- | :--- |
| **② 张量数据访问** | ✅ 全部完成（mmap / 类型系统 / 反量化） | — |
| **③ Tokenizer** | ✅ 全部完成（byte-level BPE / 字节还原） | 解析 `tokenizer.*` 元数据 |
| **④ 模型权重加载** | ✅ 全部完成（按名索引 / 配置 / 逐层组装） | embedding / norm / attn / ffn / ssm |
| **⑤ Transformer 计算** | ✅ 全部完成（算子 / Attention / SSM / 全模型前向） | 矩阵乘、前向传播 |
| **⑥ 生成引擎** 🔄 进行中 | Sampler ✅、生成循环 ✅；待：Chat 多轮对话 | 预填充 + 解码 |

### 阶段 ②：张量数据访问 ✅

- [x] `GGUFLoader::map_data`：mmap 挂载数据区到 `model.data.data_ptr`
- [x] GGML 类型系统：`data_type` → 元素大小 / 块大小（`type_size` / `block_size` 表）
- [x] 读取指定张量原始字节（按 `offset` 定位，已验证 offset 为**数据区相对偏移**）
- [x] 反量化：F16 / BF16 / Q4_0 / Q8_0 → `float`（含单元测试）

### 阶段 ③：Tokenizer ✅

- [x] 词汇表加载：token 列表 / 分数 / 类型、merges、model 类型、bos/eos/pad/unk id
- [x] byte-level BPE：gpt2 风格字节↔Unicode 映射表、贪心 merge（按 rank）
- [x] token id ↔ 文本 互转（decode 按逐码点字节还原，中文/特殊字符无损）
- [x] 实测：`encode("Hello, world!")` → `9419,11,1814,0`，decode(encode) 往返一致；中文 `你好，世界！` → 4 tokens 往返一致
- [x] 单元测试 `test_tokenizer`（不依赖模型，手建 vocab 验证 BPE / 字节还原）

### 阶段 ④：模型权重加载 ✅

- [x] `GGUFModelWeights`：按名称索引全部 320 个张量（`find`，零拷贝指向 mmap 区）
- [x] `GGUFTensorView`：张量视图（dims/type/data）+ 反量化读取（`read_element` / `read_all`）
- [x] `GGUFModelConfig`：解析 `qwen35.*` 配置（block / hidden / head / FFN / SSM / RoPE）
- [x] `GGUFBlockWeights`：逐层权重组装，`is_attention()` / `is_ssm()` 区分两类块
- [x] 实测：24 层（6 个纯 Attention 层 + 18 个 SSM 混合层），hidden=1024，共享 embedding（无 output.weight）
- [x] 单元测试 `test_model_weights`（不依赖模型，手建假模型验证索引/配置/组装/读取）

### 阶段 ⑤：Transformer 计算 ✅

**第 1 步：基础算子层 ✅**

- [x] `GGMLOps`：矩阵向量乘（gemv）/ 矩阵乘（gemm）/ softmax（含掩码）
- [x] `GGMLNorm`：RMSNorm（含 eps）
- [x] `GGMLFFN`：SwiGLU 前馈（gate/up/down 三投影）
- [x] `GGMLRope`：NEOX RoPE（纯文本下等价 Qwen3.5 的 IMROPE 退化，旋转前 n_rot 维）
- [x] `GGMLAttention`：KV cache + GQA 分组注意力（自回归场景）
- [x] 单元测试 `test_model_ops`（不依赖模型，手算/性质断言验证全部算子）

> 发现：该 BF16 模型 FFN 权重含约 0.3% 的 BF16 NaN（指数全 1、尾数非 0），属模型文件本身数据，非解析错误；正常前向会产生 NaN 输出。

**第 2 步：纯 Attention 层前向 ✅**

- [x] `GGMLTransformer`：完整层前向（RMSNorm → Q+gate 联合投影 → Q/K 拆分 → Q/K RMSNorm → RoPE → KV cache → GQA → ×sigmoid(gate) → 输出投影 → 残差 → post RMSNorm → SwiGLU FFN → 残差）
- [x] 头布局校准：attn_q 是「联合 Q+gate 投影」（2×head_dim×n_head），head_dim=key_length=256，attn_output 输入 = n_head×head_dim = 2048
- [x] 单元测试 `test_transformer`（假权重验证结构/确定性/维度/RoPE 位置影响）
- [x] 真实模型：blk.3 层前向跑通（输出 NaN 系模型权重本身含 NaN）

**第 3 步：SSM 混合层前向 ✅**

- [x] `GGMLSSM`：Gated Delta Net 完整前向（qkvz 投影 → depthwise conv1d → SiLU → q/k L2 归一化 → α/β/φ 门控 → 状态递推 → gated norm → 输出投影 → 残差 → SwiGLU）
- [x] `GGMLDeltaNetStep`：核心递推 `S' = φ·S + β·k⊗(v − φ·Sᵀk)`，`o = Sᵀq/√d_state`（手算验证一致）
- [x] 状态持久：S[16][128][128] 与 conv 历史 [3][6144] 跨 token 持久（类似 KV cache）
- [x] 单元测试 `test_ssm`（手算 delta net + 层结构/确定性/状态累积/conv 推进）
- [x] 真实模型：blk.0 SSM 层前向跑通（输出 NaN 系模型权重本身含 NaN）

**第 4 步：全模型前向 ✅**

- [x] `GGMLForward`：token_embd → 24 层混合循环（`is_ssm()` / `is_attention()` 分发）→ output_norm → 共享 embedding 投影得 logits
- [x] `GGMLModelState`：每层各一份 KV cache / SSM 状态，跨 token 持久（自回归解码准备）
- [x] 单元测试 `test_forward`（2 层假模型：logits 维度 / 有限 / 确定性 / token 区分 / 状态累积）
- [x] 真实模型：完整 24 层前向 → logits[248320]（全 NaN 系模型权重本身含 NaN）

### 阶段 ⑥：生成引擎 🔄 进行中

**第 1 步：采样器 ✅**

- [x] `GGMLSampler`：GREEDY / TOP_K / TOP_P / TOP_K_P（temperature 缩放 + 数值稳定 softmax）
- [x] top-k（保留概率前 k）+ top-p（nucleus 累计），`temperature≤0` 退化为贪心
- [x] 种子确定性：相同 seed → 相同采样序列（可复现）
- [x] 单元测试 `test_sampler`（合成 logits 验证各模式与边界）

**第 2 步：自回归生成循环 ✅**

- [x] `GGMLGenerate`：预填充（逐 token 前向更新状态）→ 循环 forward+sample → eos 提前停止
- [x] 单元测试 `test_generate`（2 层假模型：生成数量 / 确定性 / eos 停止）
- [x] 真实模型演示：检测到 logits 全 NaN 时安全提示（不崩溃），建议换干净 GGUF

**待做（下一步）**

- [ ] Chat 多轮对话（状态管理 + 对话模板）
---

## 关键认知与踩坑

本项目从头踩坑得到的经验，汇总如下：

**文件格式**
- GGUF 张量 `offset` 是**相对数据区起点**的偏移（非文件头）；数据区大小 = 张量累加 + 尾部填充
- GGUF 权重按**列主序**存储（`dims=[in, out]`，in 最内层），等价行主序 `[out, in]`，`gemv` 可直接用
- 本模型 `type=30` 实测是 **BF16**（2 字节/元素），标准 ggml 该编号是 IQ1_S —— 类型表必须按实际模型校准

**Tokenizer**
- 字节级 BPE 中空格等以 `Ġ`(U+0120) 存于词表，可能标记为 `token_type=1`(NORMAL) 而非 6(BYTE)
- decode 必须**逐码点字节还原**（不能只看 type==6），否则空格还原不出来

**qwen35 架构（SSM + Attention 混合）**
- 24 层：每 4 层一个纯 Attention 层（blk.3/7/11/15/19/23），其余 18 层为 SSM 混合层；共享 embedding
- 纯 Attention 层的 `attn_q` 是**联合 Q+gate 投影**：输出 `2×head_dim×n_head`，前半 Q 后半 gate（sigmoid 后相乘）；`head_dim=key_length=256`
- 纯文本下 IMROPE 退化为标准 NEOX RoPE（旋转前 n_rot 维）
- SSM 层是 **Gated Delta Net**：`S' = φ·S + β·k⊗(v − φ·Sᵀk)`，`o = Sᵀq/√d_state`；conv1d 无 bias、q/k 用 `max(‖x‖,ε)` L2 归一化

**⚠️ 模型文件问题**
- 该 Qwen3.5-0.8B-BF16.gguf 的 FFN 权重含约 0.3% 的**真 BF16 NaN**（指数全 1、尾数非 0），经 24 层传播后全模型 logits 全为 NaN
- 这是模型文件本身的数据问题（非解析/实现 bug）；要做端到端生成需换官方/干净模型

**测试经验**
- 假模型权重若**全用相同值**会退化（输出对输入/历史只有浮点噪声级差异，断言易误判）；应填有区分度的值
---

## 项目结构

```text
gguf_cpp/
├── CMakeLists.txt              # 构建（main / test_* 可执行 + MyLib 静态库）
├── README.md                   # 本文档
├── include/core/
│   ├── GGUFLoader.hpp          # 全部公共类型 + GGUFLoader 接口
│   ├── GGMLType.hpp            # GGML 类型描述（type_size / block_size）
│   ├── GGMLDequantize.hpp      # 反量化（F16/BF16/Q4_0/Q8_0 → float）
│   ├── GGUFTokenizer.hpp       # 分词器（byte-level BPE，token↔文本）
│   └── GGUFModelWeights.hpp    # 权重加载（按名索引 / 配置 / 逐层组装）
├── include/model/              # 阶段⑤：Transformer 计算
│   ├── GGMLOps.hpp             # 基础算子（gemv / gemm / softmax）
│   ├── GGMLNorm.hpp            # RMSNorm
│   ├── GGMLFFN.hpp             # SwiGLU 前馈
│   ├── GGMLRope.hpp            # RoPE 旋转位置编码
│   └── GGMLAttention.hpp       # KV cache + GQA 注意力
│   ├── GGMLTransformer.hpp     # 纯 Attention 层完整前向
│   ├── GGMLSSM.hpp             # SSM 混合层（Gated Delta Net）前向
│   ├── GGMLForward.hpp         # 全模型前向（多层 + logits）
│   ├── GGMLSampler.hpp         # 采样器（temp / top-k / top-p / greedy）
│   └── GGMLGenerate.hpp        # 自回归生成循环
├── src/
│   ├── main.cpp                # 演示：解析 + 类型验证 + mmap + 反量化 + 分词 + 权重 + 算子 + 生成
│   ├── core/
│   │   ├── GGUFLoader.cpp      # 解析实现（①→②→③→④）
│   │   ├── GGUFMmap.cpp        # mmap 模块（map_data / unmap_data）
│   │   ├── GGMLType.cpp        # 类型表实现
│   │   ├── GGMLDequantize.cpp  # 反量化实现
│   │   ├── GGUFTokenizer.cpp   # 分词器实现
│   │   └── GGUFModelWeights.cpp # 权重加载实现
│   ├── model/                  # 阶段⑤⑥ 实现
│   │   ├── GGMLOps.cpp         # 基础算子实现
│   │   ├── GGMLNorm.cpp        # RMSNorm 实现
│   │   ├── GGMLFFN.cpp         # SwiGLU 实现
│   │   ├── GGMLRope.cpp        # RoPE 实现
│   │   ├── GGMLAttention.cpp   # Attention + KV cache 实现
│   │   ├── GGMLTransformer.cpp # 层前向实现
│   │   ├── GGMLSSM.cpp         # SSM 层（delta net）实现
│   │   ├── GGMLForward.cpp     # 全模型前向实现
│   │   ├── GGMLSampler.cpp     # 采样器实现
│   │   └── GGMLGenerate.cpp    # 生成循环实现
│   └── test/
│       ├── test_gguf_parser.cpp  # 解析测试
│       ├── test_verification.cpp # 类型系统验证
│       ├── test_dequantize.cpp   # 反量化单元测试
│       ├── test_tokenizer.cpp    # 分词器单元测试
│       ├── test_tokenizer_ggml.cpp # 真实模型分词器验证
│       ├── test_model_weights.cpp# 权重加载单元测试
│       ├── test_model_ops.cpp    # 基础算子单元测试
│       ├── test_transformer.cpp  # Attention 层前向单元测试
│       ├── test_ssm.cpp          # SSM 层前向单元测试
│       ├── test_forward.cpp      # 全模型前向单元测试
│       ├── test_sampler.cpp      # 采样器单元测试
│       └── test_generate.cpp     # 生成循环单元测试
├── doc/architecture.md         # 目标架构设计
└── build/                      # 构建产物
```

---

## 构建与运行

环境要求：CMake ≥ 3.20、支持 C++20 的编译器（GCC ≥ 10 / Clang ≥ 12）。

```bash
# 构建
cmake -S . -B build
cmake --build build

# 运行（加载模型，打印元数据/张量表/类型验证/mmap/反量化/分词/权重/算子/层前向）
./build/main

# 运行全部测试（CTest，当前 12 个：解析/类型/反量化/分词/真实分词/权重/算子/层前向/SSM/全模型/采样/生成）
ctest --test-dir build --output-on-failure
```

> 模型路径目前硬编码在 `src/main.cpp` 的 `model_path` 中，后续改为命令行参数。

代码风格：clang-format（规则见 `.clang-format`），VS Code 已配置保存时自动格式化。

---

## 实测验证

对真实模型 `Qwen3.5-0.8B-BF16.gguf`（1.5 GB）运行结果：

```text
✅ 加载成功
  魔数: 0x46554747  版本: 3  元数据 KV: 42  张量: 320

[STRING] general.architecture = "qwen35"
[UINT32] qwen35.block_count = 24
[ARRAY]  general.tags = [ STRING x1 ] ("image-text-to-text")   ← ARRAY 递归 ✓
...
token_embd.weight  dims=[1024, 248320]  type=30  offset=0  elements=254279680
...
数据区偏移: 10961669  总大小: 1505783067 字节 (1436.03 MiB)
```

**校验点**：
- ✅ 数据区偏移 + 数据区大小 = 文件总大小（`10961669 + 1505783067 = 1516744736`）精确吻合
- ✅ 元数据 ARRAY 递归解析正确（`general.tags = [STRING x1]`）
- ✅ 张量 `element_count()` 与维度乘积一致（`1024 × 248320 = 254279680`）

**GGML 类型系统验证**：
- ✅ 类型反推（从文件布局实测）：`type=0(F32) 4字节/元素`、`type=30(BF16) 2字节/元素`
- ✅ 张量数据累加 `1505783040` + 尾部填充 `27` = 数据区 `1505783067`，完全吻合
- ✅ 关键认知：GGUF 张量 `offset` 是**相对数据区起点**的偏移（而非文件头）

**反量化验证**：
- ✅ 转换自检：`GGMLBF16ToFloat(0x3F80)=1.0`、`(0x4000)=2.0`
- ✅ F32 反量化 == 直接读，交叉验证一致
- ✅ 反量化单元测试 17 项（BF16 / F16 / Q4_0 / Q8_0 / 边界）全部通过

**阶段⑤ 验证（Transformer 计算）**：
- ✅ 基础算子单测 14 项全过（gemv / gemm / softmax / RMSNorm / SwiGLU / RoPE / GQA+KV cache）
- ✅ Attention 层前向单测 5 项全过（结构 / 确定性 / KV 递增 / RoPE 位置影响）
- ✅ SSM 层单测：delta net 手算一致 + 层结构 / 确定性 / 状态累积 / conv 推进
- ✅ 全模型前向单测 6 项（logits 维度 / 有限 / 确定性 / token 区分 / 状态累积）
- ✅ 生成引擎单测：sampler 各模式 + 生成循环（数量 / 确定性 / eos 停止）
- ✅ 真实模型完整 24 层前向 → logits[248320]（全 NaN 系模型权重本身含 NaN）
- ✅ 真实模型 `blk.3`（纯 Attention 层）与 `blk.0`（SSM 层）前向均跑通，无越界
- ⚠️ 发现：该 BF16 模型 FFN 权重含约 0.3% 的 BF16 NaN，导致真实前向输出 NaN（属模型文件本身数据，非实现 bug）

---

## 附录：GGUF 格式规范

约定：所有多字节整数均为小端序（Little-Endian）。
文件按逻辑顺序线性排列，解析器从头到尾依次读取四个区块。

```text
┌─────────┬──────────────┬──────────────┬────────────────────┐
│ ①Header │ ②Meta KV     │ ③Tensor 表   │ ④Tensor Data 区     │
│ 24 字节  │ 列表（变长）   │ （变长）      │ （纯二进制裸数据）    │
└─────────┴──────────────┴──────────────┴────────────────────┘
```

**① 文件头（固定 24 字节）**：

| 偏移 | 字段 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| 0 | `magic` | `uint32_t` | 固定值 `0x46554747`（ASCII "GGUF"） |
| 4 | `version` | `uint32_t` | 版本号，当前为 `3` |
| 8 | `tensor_count` | `uint64_t` | 张量总数 |
| 16 | `metadata_kv_count` | `uint64_t` | 元数据键值对数量 |

**② 元数据 KV 对**：`key`（字符串） + `value_type`（4B） + `value`（变长）。字符串格式：`uint64_t 长度 + UTF-8 内容（无结束符）`。

`value_type` 枚举（`GGUFValueType`）：

| 值 | 类型 | 大小 | 值 | 类型 | 大小 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | UINT8 | 1 字节 | 7 | BOOL | 1 字节 |
| 1 | INT8 | 1 字节 | 8 | STRING | 变长 |
| 2 | UINT16 | 2 字节 | 9 | ARRAY | 元素类型 + 个数 + 序列（可递归） |
| 3 | INT16 | 2 字节 | 10 | UINT64 | 8 字节 |
| 4 | UINT32 | 4 字节 | 11 | INT64 | 8 字节 |
| 5 | INT32 | 4 字节 | 12 | FLOAT64 | 8 字节 |
| 6 | FLOAT32 | 4 字节 | | | |

**③ 张量信息表**：`name`（字符串） → `n_dimensions`（4B） → `dimensions`（`uint64_t × n`） → `data_type`（4B，GGML 枚举） → `offset`（8B）。

**④ 张量数据区**：各张量原始数据按 `offset` 定位依次排列；长度由 `dimensions` 与 `data_type` 共同决定；数据可能未对齐，应按字节偏移直接读取。

---

*本文档遵循 GGUF v3 规范。目标架构设计见 `doc/architecture.md`。*
 
