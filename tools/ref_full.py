#!/usr/bin/env python3
"""独立 numpy 实现 Qwen3.5 全部 24 层前向（SSM/Attention 混合），逐层 dump 输出 norm + first8，
并计算最终 logits top-10。用于与项目逐层对比定位分歧层（曾据此定位 Q/gate 交错 bug）。
用法: python ref_full.py <model.gguf> [token_id]
依赖: llama.cpp 的 gguf-py（路径可用环境变量 LLAMA_CPP_GGUF_PY 覆盖，默认 ~/Projects/llama.cpp/gguf-py）
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.environ.get("LLAMA_CPP_GGUF_PY", "/home/dongfan/Projects/llama.cpp/gguf-py"))
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType as QT

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/dongfan/llm/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf"
token = int(sys.argv[2]) if len(sys.argv) > 2 else 9419
r = GGUFReader(path)
by = {t.name: t for t in r.tensors}

def load(name, shape):
    t = by[name]
    if t.tensor_type == QT.F32:
        v = t.data.astype(np.float32)
    elif t.tensor_type == QT.BF16:
        b = t.data.reshape(-1, 2)
        le = (b[:, 0].astype(np.uint32) | (b[:, 1].astype(np.uint32) << 8)).astype(np.uint32)
        v = (le << 16).view(np.float32)
    elif t.tensor_type == QT.F16:
        v = t.data.astype(np.float16).astype(np.float32)
    else:
        raise ValueError(t.tensor_type)
    return v.reshape(shape)

def rmsnorm(x, gamma, eps):
    rms = np.sqrt((x * x).mean() + eps)
    return x / rms * gamma

def silu(x):
    return x / (1.0 + np.exp(-x))

def softplus(x):
    return np.log1p(np.exp(x))

hidden, d_inner, d_state, n_group = 1024, 2048, 128, 16
key_dim = d_state * n_group
conv_dim = 2 * key_dim + d_inner
conv_kernel, ffn_hidden, eps = 4, 3584, 1e-6
head_dim, n_head, n_kv = 256, 8, 2

def ffn_and_resid(x, l):
    pn = load(f"blk.{l}.post_attention_norm.weight", (hidden,))
    fg = load(f"blk.{l}.ffn_gate.weight", (ffn_hidden, hidden))
    fu = load(f"blk.{l}.ffn_up.weight", (ffn_hidden, hidden))
    fd = load(f"blk.{l}.ffn_down.weight", (hidden, ffn_hidden))
    return x + fd @ (silu(fg @ rmsnorm(x, pn, eps)) * (fu @ rmsnorm(x, pn, eps)))

def ssm_layer(l, x):
    a_norm_w = load(f"blk.{l}.attn_norm.weight", (hidden,))
    qkv_w = load(f"blk.{l}.attn_qkv.weight", (conv_dim, hidden))
    gate_w = load(f"blk.{l}.attn_gate.weight", (d_inner, hidden))
    alpha_w = load(f"blk.{l}.ssm_alpha.weight", (n_group, hidden))
    beta_w = load(f"blk.{l}.ssm_beta.weight", (n_group, hidden))
    out_w = load(f"blk.{l}.ssm_out.weight", (hidden, d_inner))
    a = load(f"blk.{l}.ssm_a", (n_group,))
    dt = load(f"blk.{l}.ssm_dt.bias", (n_group,))
    cw = load(f"blk.{l}.ssm_conv1d.weight", (conv_dim, conv_kernel))
    sn = load(f"blk.{l}.ssm_norm.weight", (d_state,))

    xn = rmsnorm(x, a_norm_w, eps)
    qkv = qkv_w @ xn
    z = gate_w @ xn
    alpha = alpha_w @ xn
    beta = 1.0 / (1.0 + np.exp(-(beta_w @ xn)))
    qkv_conv = np.zeros(conv_dim)
    for c in range(conv_dim):
        acc = 0.0
        for j in range(conv_kernel):
            u = qkv[c] if j == conv_kernel - 1 else 0.0
            acc += u * cw[c, j]
        qkv_conv[c] = silu(acc)
    q = qkv_conv[:key_dim].reshape(n_group, d_state)
    k = qkv_conv[key_dim:2*key_dim].reshape(n_group, d_state)
    v = qkv_conv[2*key_dim:].reshape(n_group, d_state)
    q_n = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), eps)
    k_n = k / np.maximum(np.linalg.norm(k, axis=1, keepdims=True), eps)
    o = np.zeros((n_group, d_state))
    for h in range(n_group):
        kq = np.sum(k_n[h] * q_n[h])
        o[h] = beta[h] * v[h] * kq / np.sqrt(d_state)
    out = np.zeros(d_inner)
    for h in range(n_group):
        o_h = o[h]
        rms = np.sqrt((o_h**2).mean() + eps)
        normed = o_h / rms * sn
        out[h*d_state:(h+1)*d_state] = normed * silu(z[h*d_state:(h+1)*d_state])
    h1 = x + out_w @ out
    return ffn_and_resid(h1, l)

def attn_layer(l, x):
    a_norm_w = load(f"blk.{l}.attn_norm.weight", (hidden,))
    q_w = load(f"blk.{l}.attn_q.weight", (2*head_dim*n_head, hidden))
    k_w = load(f"blk.{l}.attn_k.weight", (head_dim*n_kv, hidden))
    v_w = load(f"blk.{l}.attn_v.weight", (head_dim*n_kv, hidden))
    o_w = load(f"blk.{l}.attn_output.weight", (hidden, head_dim*n_head))
    q_norm_w = load(f"blk.{l}.attn_q_norm.weight", (head_dim,))
    k_norm_w = load(f"blk.{l}.attn_k_norm.weight", (head_dim,))
    xn = rmsnorm(x, a_norm_w, eps)
    qg = q_w @ xn
    # 交错布局（llama.cpp）：qg[h*2*head_dim : h*2*head_dim+head_dim] = Q(h)，后半 = gate(h)
    q = np.stack([qg[h*2*head_dim:(h+1)*2*head_dim-head_dim] for h in range(n_head)])
    gate = np.stack([qg[h*2*head_dim+head_dim:(h+1)*2*head_dim] for h in range(n_head)])
    k = (k_w @ xn).reshape(n_kv, head_dim)
    v = (v_w @ xn).reshape(n_kv, head_dim)
    q_n = np.stack([rmsnorm(q[h], q_norm_w, eps) for h in range(n_head)])
    k_n = np.stack([rmsnorm(k[kv], k_norm_w, eps) for kv in range(n_kv)])
    attn = np.zeros((n_head, head_dim))
    group = n_head // n_kv
    for h in range(n_head):
        attn[h] = v[h // group]
    attn = attn * (1.0 / (1.0 + np.exp(-gate)))
    y = x + o_w @ attn.reshape(-1)
    return ffn_and_resid(y, l)

x = load("token_embd.weight", (248320, hidden))[token].astype(np.float64)
attn_layers = {3, 7, 11, 15, 19, 23}
for l in range(24):
    if l in attn_layers:
        x = attn_layer(l, x)
    else:
        x = ssm_layer(l, x)
    n = np.sqrt((x**2).sum())
    print(f"layer {l:2d} ({'ATTN' if l in attn_layers else 'SSM '})  norm={n:.6f}  first8={np.round(x[:8],6)}")

# output norm + logits
on = load("output_norm.weight", (hidden,))
hn = rmsnorm(x, on, eps)
embd = load("token_embd.weight", (248320, hidden))
logits = embd @ hn
idx = np.argsort(-logits)[:10]
print("numpy Top-10 logits:")
for i in idx:
    print(f"  [{i}] {logits[i]:.4f}")
