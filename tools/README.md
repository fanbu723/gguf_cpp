# tools — 调试 / 交叉验证工具

> 这些工具在排查「模型 NaN 误报」、定位 SSM 发散与 Q/gate 布局 bug 时发挥了关键作用。
> 它们不作为主构建目标，按需单独编译 / 运行。

## 工具清单

| 文件 | 作用 | 依赖 |
| :--- | :--- | :--- |
| `check_nan.cpp` | 用本项目解析器扫描 GGUF 全张量 NaN/Inf（验证解析/反量化正确性） | 本项目 `gguf_model` 库 |
| `scan_nan_ggufpy.py` | 用 llama.cpp 的 gguf-py 独立读取并按 BF16/F16/F32 正确解释后统计 NaN/Inf（交叉验证模型文件干净） | llama.cpp 的 gguf-py |
| `ref_full.py` | 独立 numpy 实现 Qwen3.5 全 25 层前向，逐层 dump 输出 + 最终 logits top-10（定位分歧层/布局 bug） | llama.cpp 的 gguf-py + numpy |
| `gguf_logits.cpp` | 本项目前向对显式 token id 序列 dump top-k logits | 本项目 `gguf_model` 库 |
| `llama_logits.cpp` | llama.cpp 前向对同一 token id 序列 dump top-k logits（A/B 对照） | llama.cpp 头文件 + libllama |

## 构建 / 运行

> 依赖本项目时，先按 README 用 preset 构建出静态库：
> `cmake --preset release && cmake --build build-release -j`

```bash
MODEL=~/llm/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf
# 项目内路径
INC="-I include/core -I include/model"
LIBS="build-release/src/model/libgguf_model.a build-release/src/core/libgguf_core.a"

# 1) 项目侧 NaN 扫描
g++ -std=c++20 -O2 $INC tools/check_nan.cpp -o /tmp/check_nan $LIBS
/tmp/check_nan "$MODEL"

# 2) 独立 NaN 扫描（llama.cpp gguf-py）
python3 tools/scan_nan_ggufpy.py "$MODEL"

# 3) numpy 全模型参考（模型路径、token id 可作参数）
python3 tools/ref_full.py "$MODEL" 9419

# 4) 项目侧 logits dump
g++ -std=c++20 -O2 $INC tools/gguf_logits.cpp -o /tmp/gguf_logits $LIBS
/tmp/gguf_logits "$MODEL" 248045,846,198,9419,11,1814,0,248046,198,248045,74455,198

# 5) llama.cpp 侧 logits dump（A/B 对照）
LLAMA=~/Projects/llama.cpp
g++ -std=c++11 -O2 -I $LLAMA/include -I $LLAMA/ggml/include tools/llama_logits.cpp \
    -o /tmp/llama_logits -L $LLAMA/build/bin -lllama -Wl,-rpath,$LLAMA/build/bin
/tmp/llama_logits "$MODEL" 248045,846,198,9419,11,1814,0,248046,198,248045,74455,198
```

## 依赖说明

- `scan_nan_ggufpy.py` / `ref_full.py` 需要 llama.cpp 仓库里的 `gguf-py`（Python 库），
  默认路径 `~/Projects/llama.cpp/gguf-py`，可用环境变量 `LLAMA_CPP_GGUF_PY` 覆盖；另需 numpy。
- `llama_logits.cpp` 需要 llama.cpp 头文件（`llama.h`、`ggml.h`）与其构建出的 `libllama.so`。
- `check_nan.cpp` / `gguf_logits.cpp` 只依赖本项目自身库。

## 使用要点

- **BF16 解释**：gguf-py 对 BF16 张量只返回原始 uint8 字节，必须手动按 `bits << 16` 转 float32
  再判 NaN（否则 uint8 数组永远不会是 NaN，会误报「模型干净」）。F32 张量勿当 BF16 读。
- **logits A/B 对照**：喂**相同的 token id 序列**给 `gguf_logits` 与 `llama_logits`，
  误差应 ≤0.05（浮点精度）。单 token（pos=0，RoPE 恒等）可分离非 RoPE 类 bug。
