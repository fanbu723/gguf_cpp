/*
 * GGMLChat.hpp — 多轮对话（阶段 ⑥ 第 3 步）
 *
 * 在生成引擎之上封装"多轮对话"：内部维护对话历史文本，
 * 每轮按 Qwen 风格模板拼接后编码 → 自回归生成 → 追加回复。
 *
 * 模板（简化 Qwen3 chat template）：
 *   <|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{回复}<|im_end|>\n
 *
 * 说明：本项目 GGMLForward 是逐 token（decode 风格），每轮会重新预填充完整
 * 历史，正确但比 KV-cache 增量方案慢；学习阶段可接受。
 */

#pragma once

#include <cstdint>
#include <string>

#include "GGMLForward.hpp"
#include "GGMLGenerate.hpp"
#include "GGMLSampler.hpp"
#include "GGUFTokenizer.hpp"

/**
 * @brief 多轮对话封装
 */
class GGMLChat {
  public:
    /**
     * @brief 初始化
     * @param w 模型权重（需已 build 且 data 已 map_data）
     * @param tok 分词器
     * @param mode 采样模式
     * @param seed 随机种子
     * @param max_len 每轮 KV cache 最大长度
     * @return 成功返回 true
     */
    bool init(const GGUFModelWeights &w, GGUFTokenizer &tok,
              GGMLSampleMode mode = GGMLSampleMode::TOP_K_P, std::uint64_t seed = 42,
              int max_len = 512);

    /**
     * @brief 一轮对话：输入 user 文本，返回 assistant 回复文本
     * @param user 用户输入
     * @param max_new_tokens 最多生成的新 token 数
     * @return assistant 回复（可能为空）
     */
    std::string chat(const std::string &user, int max_new_tokens = 128);

    /**
     * @brief 清空对话历史，开始新会话
     */
    void clear() {
        history_text_.clear();
    }

  private:
    const GGUFModelWeights *w_ = nullptr;
    GGUFTokenizer *tok_ = nullptr;
    GGMLSampler sampler_;
    int max_len_ = 512;
    std::string history_text_; // 对话历史（纯文本，含模板）
};
