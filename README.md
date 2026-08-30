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
- [附录：GGUF 格式规范](#附录gguf-格式规范)

---

## 项目目标

1. **解析层**：完整读取 GGUF v3 文件 —— ✅ 已完成
2. **数据层**：访问张量数据（mmap 挂载 + 反量化）
3. **模型层**：加载权重并执行 Transformer 前向计算
4. **生成层**：自回归文本生成（Sampler + 生成循环）

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

运行 `./build/main` 可打印元数据、张量表、类型验证、mmap 读取、反量化结果、分词器演示、权重加载与基础算子演示。

---

## 下一步计划

从解析器走向完整推理，按以下阶段逐步推进：

| 阶段 | 内容 | 关键点 |
| :--- | :--- | :--- |
| **② 张量数据访问** | ✅ 全部完成（mmap / 类型系统 / 反量化） | — |
| **③ Tokenizer** | ✅ 全部完成（byte-level BPE / 字节还原） | 解析 `tokenizer.*` 元数据 |
| **④ 模型权重加载** | ✅ 全部完成（按名索引 / 配置 / 逐层组装） | embedding / norm / attn / ffn / ssm |
| **⑤ Transformer 计算** 🔄 进行中 | 算子层 ✅；待：Attention 层前向、SSM 混合层、全模型 | 矩阵乘、前向传播 |
| **⑥ 生成引擎** | Sampler(top-k/p/temp)、自回归循环、Chat | 预填充 + 解码 |

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

### 阶段 ⑤：Transformer 计算 🔄 进行中

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

**待做（下一步）**

- [ ] SSM 混合层前向（qwen35 用 gated delta net：qkvz 投影 + conv1d + 状态空间递推 + gated norm，Mamba 风格）
- [ ] 多层前向 → 全模型 logits（含 output_norm + 共享 embedding）

### 阶段 ⑥：生成引擎

- [ ] Sampler（temperature / top-k / top-p）
- [ ] 自回归生成循环（预填充 + 逐 token 解码）
- [ ] Chat 多轮对话

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
│   └── GGMLTransformer.hpp     # 纯 Attention 层完整前向
├── src/
│   ├── main.cpp                # 演示：解析 + 类型验证 + mmap + 反量化 + 分词 + 权重 + 算子
│   ├── core/
│   │   ├── GGUFLoader.cpp      # 解析实现（①→②→③→④）
│   │   ├── GGUFMmap.cpp        # mmap 模块（map_data / unmap_data）
│   │   ├── GGMLType.cpp        # 类型表实现
│   │   ├── GGMLDequantize.cpp  # 反量化实现
│   │   ├── GGUFTokenizer.cpp   # 分词器实现
│   │   └── GGUFModelWeights.cpp # 权重加载实现
│   ├── model/                  # 阶段⑤ 实现
│   │   ├── GGMLOps.cpp         # 基础算子实现
│   │   ├── GGMLNorm.cpp        # RMSNorm 实现
│   │   ├── GGMLFFN.cpp         # SwiGLU 实现
│   │   ├── GGMLRope.cpp        # RoPE 实现
│   │   ├── GGMLAttention.cpp   # Attention + KV cache 实现
│   │   └── GGMLTransformer.cpp # 层前向实现
│   └── test/
│       ├── test_gguf_parser.cpp  # 解析测试
│       ├── test_verification.cpp # 类型系统验证
│       ├── test_dequantize.cpp   # 反量化单元测试
│       ├── test_tokenizer.cpp    # 分词器单元测试
│       ├── test_tokenizer_ggml.cpp # 真实模型分词器验证
│       ├── test_model_weights.cpp# 权重加载单元测试
│       ├── test_model_ops.cpp    # 基础算子单元测试
│       └── test_transformer.cpp  # Attention 层前向单元测试
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

# 运行（加载模型并打印元数据 / 张量表 / 类型验证 / mmap 读取）
./build/main

# 运行测试（CTest）
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
 
