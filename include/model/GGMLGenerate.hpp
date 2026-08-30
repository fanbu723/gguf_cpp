/*
 * GGMLGenerate.hpp — 自回归生成循环（阶段 ⑥ 第 2 步）
 *
 * 给定 prompt token ids，配合 GGMLForward + GGMLModelState 逐 token 自回归：
 *   1) 预填充：对 prompt 每个 token 前向（pos 递增），更新 KV/SSM 状态
 *   2) 采样：从最后一个 token 的 logits 采下一个 token（GGMLSampler）
 *   3) 循环：把新 token 喂回模型继续前向，直到生成 n_tokens 个或遇到 eos
 *
 * 注意：本项目 GGMLForward 是逐 token（decode 风格），预填充也逐 token 跑，
 * 正确但比并行 prefill 慢；学习阶段可接受。
 */

#pragma once

#include <vector>

#include "GGMLForward.hpp"
#include "GGMLSampler.hpp"

/**
 * @brief 自回归生成
 * @param w 模型权重（需已 build 且 data 已 map_data）
 * @param state 运行时状态（跨 token 持久，需已 init）
 * @param sampler 采样器（内部随机数会推进）
 * @param prompt 输入 prompt 的 token ids
 * @param n_tokens 最多生成的新 token 数
 * @param eos_id 结束 token id（≥0 时遇到即停；<0 不检查）
 * @return 生成的新 token ids（不含 prompt）
 */
std::vector<int> GGMLGenerate(const GGUFModelWeights &w, GGMLModelState &state,
                              GGMLSampler &sampler, const std::vector<int> &prompt, int n_tokens,
                              int eos_id = -1);
