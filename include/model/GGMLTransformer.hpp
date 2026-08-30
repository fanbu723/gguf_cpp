/*
 * GGMLTransformer.hpp — 单层前向（阶段 ⑤ 第 2 步）
 *
 * 实现"纯 Attention 层"（blk.3/7/11/15/19/23）的完整前向：
 *
 *   x ──▶ RMSNorm ──▶ Q+gate 联合投影 ──▶ Q/K 拆分
 *              │            │                 ├─ Q：RMSNorm → RoPE → GQA 注意力
 *              │            │                 ├─ K：RMSNorm → RoPE → 存入 KV cache
 *              │            │                 └─ V：存入 KV cache
 *              │            ▼
 *              │     attn = 注意力输出 × sigmoid(gate)
 *              │            ▼
 *              │     输出投影（attn_output）→ 残差相加 → y1
 *              │
 *              └──▶ post RMSNorm → SwiGLU FFN → 残差相加 → y
 *
 * 说明（对照 llama.cpp qwen35.cpp）：
 *  - attn_q 输出 = 2 × head_dim × n_head：前半 Q、后半 gate（联合 QG 投影）
 *  - Q/K 各自再做 RMSNorm（attn_q_norm / attn_k_norm），再应用 RoPE
 *  - GQA：n_head 个 Q 头共享 n_head_kv 个 KV 头
 *  - 纯文本下 IMROPE 退化为标准 NEOX RoPE（旋转前 rope_dimension_count 维）
 *
 * 权重从 GGUFBlockWeights 读取，每次前向会重新反量化（BF16→float），
 * 学习阶段可接受；后续可加权重缓存优化。
 */

#pragma once

#include "GGMLAttention.hpp"
#include "GGUFModelWeights.hpp"

/**
 * @brief 纯 Attention 层完整前向（含 KV cache 更新与残差）
 * @param w 该层的权重（GGUFBlockWeights，data 需已 map_data）
 * @param cfg 模型配置（hidden / head / key_length / eps / rope 等）
 * @param cache KV cache（需已 init，长度应能容纳 pos）
 * @param pos 当前 token 位置（从 0 开始）
 * @param x 输入向量（hidden 个元素）
 * @param y 输出向量（hidden 个元素）
 */
void GGMLTransformerAttentionBlock(const GGUFBlockWeights &w, const GGUFModelConfig &cfg,
                                   GGMLKVCache &cache, int pos, const float *x, float *y);
