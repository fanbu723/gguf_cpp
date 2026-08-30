#!/usr/bin/env python3
"""用 llama.cpp 自带 gguf-py 读取张量原始字节，手动按 BF16/F16/F32 解释后统计 NaN/Inf。
（gguf-py 对 BF16 只返回 uint8 字节，需自行解释）
用法: python scan_nan_ggufpy.py <model.gguf>
依赖: llama.cpp 的 gguf-py（路径可用环境变量 LLAMA_CPP_GGUF_PY 覆盖，默认 ~/Projects/llama.cpp/gguf-py）
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.environ.get("LLAMA_CPP_GGUF_PY", "/home/dongfan/Projects/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/dongfan/llm/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf"

reader = GGUFReader(path)
print(f"模型: {path}\n张量总数: {len(reader.tensors)}")

BF16 = GGMLQuantizationType.BF16
F16 = GGMLQuantizationType.F16
F32 = GGMLQuantizationType.F32


def bf16_to_float32(bits):
    # bits: uint16 array (little-endian already) -> float32 via <<16
    u = bits.astype(np.uint32) << 16
    return u.view(np.float32)


total_nan = 0
total_inf = 0
bad = []
for t in reader.tensors:
    raw = t.data
    if raw is None:
        continue
    # 展平并转成 float32 数组
    if t.tensor_type == F32:
        vals = raw.astype(np.float32)
    elif t.tensor_type == F16:
        vals = raw.astype(np.float16).astype(np.float32)
    elif t.tensor_type == BF16:
        b = raw.reshape(-1, 2)  # 每元素 2 字节
        le = (b[:, 0].astype(np.uint32) | (b[:, 1].astype(np.uint32) << 8))
        vals = bf16_to_float32(le)
    else:
        continue  # 量化类型跳过
    nan = int(np.isnan(vals).sum())
    inf = int(np.isinf(vals).sum())
    total_nan += nan
    total_inf += inf
    if nan or inf:
        bad.append((t.name, t.shape, str(t.tensor_type), nan, inf))

n_float = len([t for t in reader.tensors if t.tensor_type in (BF16, F16, F32)])
print(f"检查 BF16/F16/F32 张量: {n_float}")
print(f"含 NaN/Inf 张量: {len(bad)}")
print(f"NaN 总数: {total_nan}  Inf 总数: {total_inf}")
for name, shape, dtype, nan, inf in bad:
    print(f"  ⚠️ {name}  shape={shape}  type={dtype}  NaN={nan}  Inf={inf}")
if not bad:
    print("  ✅ 无 NaN/Inf")
