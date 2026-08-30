/*
 * GGMLForward.hpp — 全模型前向（阶段 ⑤ 第 4 步）
 *
 * 把 24 层（6 个纯 Attention 层 + 18 个 SSM 混合层）串成完整前向：
 *
 *   h = token_embd[token]                       ← token 的 embedding
 *   for il in 0..block_count-1:
 *       if blocks[il].is_attention(): h = AttentionBlock(h)   ← 更新该层 KV cache
 *       else:                       h = SSMLayer(h)          ← 更新该层 SSM 状态
 *   h = RMSNorm(h, output_norm)
 *   logits = h · token_embdᵀ                     ← 共享 embedding（无 output.weight）
 *
 * 自回归解码：每步喂一个 token id + 位置 pos，状态（KV cache / SSM 状态）
 * 在 GGMLModelState 中跨 token 持久。
 */

#pragma once

#include <vector>

#include "GGMLAttention.hpp"
#include "GGMLSSM.hpp"
#include "GGUFModelWeights.hpp"

/**
 * @brief 模型运行时状态：每层各一份 KV cache（Attention 层）或 SSM 状态（SSM 层）
 */
struct GGMLModelState {
    std::vector<GGMLKVCache> kv;   // 与 blocks 对齐，Attention 层使用
    std::vector<GGMLSSMState> ssm; // 与 blocks 对齐，SSM 层使用

    /**
     * @brief 按模型配置初始化各层状态
     * @param w 模型权重（需已 build）
     * @param max_len KV cache 最大长度（自回归解码时用）
     */
    void init(const GGUFModelWeights &w, int max_len = 256);
};

/**
 * @brief 全模型前向：token id → logits 向量
 * @param w 模型权重（需已 build 且 data 已 map_data）
 * @param state 运行时状态（跨 token 持久，需已 init）
 * @param token 当前 token id
 * @param pos 当前位置（从 0 开始）
 * @param logits 输出 logits（大小 = 词汇表大小）
 */
void GGMLForward(const GGUFModelWeights &w, GGMLModelState &state, int token, int pos,
                 std::vector<float> &logits);
