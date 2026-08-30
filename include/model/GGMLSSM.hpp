/*
 * GGMLSSM.hpp — SSM 混合层前向（阶段 ⑤ 第 3 步）
 *
 * Qwen3.5（qwen35）的 SSM 层是 Mamba 风格的 **Gated Delta Net**，完整前向：
 *
 *   x_norm = RMSNorm(x, attn_norm)
 *   qkv = W_qkv·x_norm（6144）┐
 *   z   = W_gate·x_norm(2048) │  qkv 与最近 3 个历史做 depthwise conv1d + SiLU
 *   α   = W_α·x_norm, β = σ(W_β·x_norm)
 *   q,k = L2Norm(每 head 128 维), v 不变
 *   φ   = exp( A_h · softplus(α_h + dt_h) )
 *   S   = φ·S + β·k ⊗ (v − φ·Sᵀk)          ← Gated Delta Net 状态递推
 *   o   = Sᵀq / √128
 *   out = RMSNorm(o, γ_ssm_norm) ⊙ SiLU(z)  ← gated norm
 *   y   = W_out·out（d_inner → hidden）
 *   h1  = y + x（残差，加 attn_norm 之前的输入）
 *   h2  = RMSNorm(h1, post_attention_norm)；h_out = h1 + SwiGLU(h2)
 *
 * 参考 llama.cpp src/models/qwen35.cpp + llm_build_delta_net_base。
 * 本模块实现 decode 单 token 场景（T=1）：状态 S 与 conv 历史跨 token 持久。
 */

#pragma once

#include <cstddef>
#include <vector>

#include "GGUFModelWeights.hpp"

/**
 * @brief Gated Delta Net 状态（每层一份，跨 token 持久）
 *
 * - S：状态矩阵 [num_v_heads][d_state][d_state]（16×128×128），初始全 0
 * - conv_hist：最近 3 个 qkv 历史 [3][conv_dim]（6144），初始全 0
 */
struct GGMLSSMState {
    std::vector<float> S;         // [n_v_heads][d_state][d_state]
    std::vector<float> conv_hist; // [3][conv_dim]
    int n_v_heads = 0;
    int d_state = 0;
    int conv_dim = 0;

    /**
     * @brief 初始化状态（全 0）
     * @param n_v_heads v-head 数（= ssm_group_count）
     * @param d_state 状态维度（= ssm_state_size）
     * @param conv_dim conv 通道数（= 2×key_dim + value_dim）
     */
    void init(int n_v_heads, int d_state, int conv_dim);

    /**
     * @brief 重置为全 0（新序列）
     */
    void reset();
};

/**
 * @brief 单个 v-head 的一步 Gated Delta Net 递推（解码单个 token）
 * @param q 归一化后的 Q 向量（Sv 个元素）
 * @param k 归一化后的 K 向量（Sv 个元素）
 * @param v V 向量（Sv 个元素）
 * @param beta 门控标量（已 sigmoid）
 * @param phi 衰减标量（= exp(gate)）
 * @param S 状态矩阵 [Sv][Sv]，原地更新（S[i*Sv+j]，i 为 key/行维、j 为 value/列维）
 * @param Sv 状态维度
 * @param o 输出向量（Sv 个元素）
 *
 * 递推：kv = Sᵀk；S = φ·S + β·k⊗(v − φ·kv)；o = Sᵀq / √Sv
 */
void GGMLDeltaNetStep(const float *q, const float *k, const float *v, float beta, float phi,
                      float *S, int Sv, float *o);

/**
 * @brief SSM 混合层完整前向（解码单个 token）
 * @param w 该层权重（GGUFBlockWeights，data 需已 map_data）
 * @param cfg 模型配置
 * @param state SSM 状态（每层一份，跨 token 持久）
 * @param x 输入向量（hidden 个元素，是上一块输出）
 * @param y 输出向量（hidden 个元素）
 */
void GGMLSSMLayer(const GGUFBlockWeights &w, const GGUFModelConfig &cfg, GGMLSSMState &state,
                  const float *x, float *y);
