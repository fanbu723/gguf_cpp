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

**GGUF 解析器已完成**，已用真实模型（Qwen3.5-0.8B, 1.5GB）验证：

| 功能 | 状态 |
| :--- | :---: |
| 文件头解析（24 字节 + 魔数校验） | ✅ |
| 元数据 KV 列表（含递归 ARRAY） | ✅ |
| 张量信息表（name / shape / type / offset） | ✅ |
| 张量数据区视图（延迟加载，零拷贝） | ✅ |

运行 `./build/gguf_parser` 可打印 42 条元数据、320 个张量、数据区信息。

---

## 下一步计划

从解析器走向完整推理，按以下阶段逐步推进：

| 阶段 | 内容 | 关键点 |
| :--- | :--- | :--- |
| **② 张量数据访问** ⬅ **下一步** | mmap 挂载、GGML 类型系统、反量化 | 让 `data_ptr` 真正可用 |
| **③ Tokenizer** | vocab 加载、byte-level BPE、token↔文本 | 解析 `tokenizer.*` 元数据 |
| **④ 模型权重加载** | Model 容器、按名查张量、组装各层权重 | embedding / norm / attn / ffn |
| **⑤ Transformer 计算** | RMSNorm / RoPE / Attention(+KV cache) / SwiGLU | 矩阵乘、前向传播 |
| **⑥ 生成引擎** | Sampler(top-k/p/temp)、自回归循环、Chat | 预填充 + 解码 |

### 阶段 ②：张量数据访问（当前待做）

- [ ] `GGUFLoader::map_data`：mmap 挂载数据区到 `model.data.data_ptr`
- [ ] GGML 类型系统：`data_type` → 元素大小 / 块大小（`type_size` / `block_size` 表）
- [ ] 反量化：F16 / BF16 / Q4_0 / Q8_0 → `float`
- [ ] 读取指定张量原始字节（按 `offset` 定位）

### 阶段 ③：Tokenizer

- [ ] 词汇表加载（token 列表 / 分数 / 类型，来自 `tokenizer.ggml.*` 元数据）
- [ ] byte-level BPE 分词
- [ ] token id ↔ 文本 互转

### 阶段 ④：模型权重加载

- [ ] Model 容器：按名称查找张量（embedding、各层权重）
- [ ] Transformer 层权重组装（RMSNorm / RoPE / Attention / FFN）

### 阶段 ⑤：Transformer 计算

- [ ] 基础算子：矩阵乘、RMSNorm、RoPE、Attention、SwiGLU
- [ ] KV cache 管理
- [ ] 前向传播（单层 → 多层 → 全模型）

### 阶段 ⑥：生成引擎

- [ ] Sampler（temperature / top-k / top-p）
- [ ] 自回归生成循环（预填充 + 逐 token 解码）
- [ ] Chat 多轮对话

---

## 项目结构

```text
gguf_cpp/
├── CMakeLists.txt              # 构建（gguf_parser 可执行 + MyLib 静态库）
├── README.md                   # 本文档
├── include/gguf/
│   └── GGUFLoader.hpp          # 全部公共类型 + GGUFLoader 接口
├── src/
│   ├── main.cpp                # 演示：加载并打印元数据 / 张量表 / 数据区
│   └── gguf/
│       └── GGUFLoader.cpp      # 解析实现（①→②→③→④）
├── doc/                        # 设计文档
└── build/                      # 构建产物
```

---

## 构建与运行

环境要求：CMake ≥ 3.20、支持 C++20 的编译器（GCC ≥ 10 / Clang ≥ 12）。

```bash
# 构建
cmake -S . -B build
cmake --build build

# 运行（加载模型并打印元数据 / 张量表 / 数据区）
./build/gguf_parser
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

*本文档遵循 GGUF v3 规范。详细架构设计见 `doc/GGUF文件格式处理系统.md`。*
 
