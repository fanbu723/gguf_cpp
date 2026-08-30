/*
 * GGUFModelWeights.hpp — 模型权重加载（阶段 ④）
 *
 * 在 GGUF 解析（①）+ mmap（②）的基础上，把"张量信息表 + 数据区"组织成
 * 便于阶段⑤ Transformer 计算使用的结构：
 *  - 按名称索引所有张量（find）
 *  - 从 qwen35.* 元数据解析模型配置（层数 / hidden / head / SSM 参数...）
 *  - 把每层权重组装成 GGUFBlockWeights（Attention 层 与 SSM 混合层 的并集，
 *    缺失的权重为 nullptr，用 is_attention() / is_ssm() 区分）
 *
 * 零拷贝：GGUFTensorView::data 直接指向 mmap 映射区，不复制权重；
 * 反量化按需进行（read_element / read_all 调用 GGMLDequantize）。
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "GGUFLoader.hpp"

// ============================================================================
// 模型配置（从 qwen35.* 元数据解析）
// ============================================================================

/**
 * @brief 模型超参数配置（从 GGUF 元数据的 qwen35.* / 通用字段解析）
 */
struct GGUFModelConfig {
    std::string arch;                      // 架构名（"qwen35"）
    std::uint32_t block_count = 0;         // Transformer 块总数
    std::uint32_t context_length = 0;      // 上下文长度
    std::uint32_t embedding_length = 0;    // 隐藏层维度（hidden）
    std::uint32_t feed_forward_length = 0; // FFN 中间维度
    std::uint32_t head_count = 0;          // Q 头数
    std::uint32_t head_count_kv = 0;       // KV 头数（GQA）
    std::uint32_t key_length = 0;          // 每个 KV 头的 key 维度
    std::uint32_t value_length = 0;        // 每个 KV 头的 value 维度
    float rms_eps = 1e-6f;                 // RMSNorm epsilon
    // SSM（Mamba 风格）参数
    std::uint32_t ssm_conv_kernel = 0;         // 卷积核大小
    std::uint32_t ssm_state_size = 0;          // SSM 状态维度
    std::uint32_t ssm_group_count = 0;         // SSM 分组数
    std::uint32_t ssm_time_step_rank = 0;      // 时间步投影维度
    std::uint32_t ssm_inner_size = 0;          // SSM 内部/输出维度
    std::uint32_t full_attention_interval = 0; // 每 N 层插入一个纯 Attention 层
    // RoPE
    std::uint32_t rope_dimension_count = 0; // 参与旋转的维度数
    float rope_freq_base = 10000.0f;        // 旋转位置编码基频
};

// ============================================================================
// 张量视图
// ============================================================================

/**
 * @brief 单个张量的数据视图（零拷贝，指向 mmap 映射区）
 */
struct GGUFTensorView {
    std::string name;                   // 张量名称（如 "token_embd.weight"）
    std::vector<std::uint64_t> dims;    // 各维度大小（GGUF 列主序，最后一维在最内层）
    std::uint32_t type = 0;             // GGML 数据类型
    const std::uint8_t *data = nullptr; // 指向数据区内该张量的原始字节

    /**
     * @brief 计算元素总数（各维度乘积）
     * @return 元素总数
     */
    std::uint64_t element_count() const;

    /**
     * @brief 反量化读取第 idx 个元素
     * @param idx 元素下标（从 0 开始）
     * @param out 输出参数，接收反量化后的 float
     * @return 成功返回 true；类型暂不支持返回 false
     */
    bool read_element(std::uint64_t idx, float &out) const;

    /**
     * @brief 反量化整张张量为 float 数组
     * @param out 输出数组，调用前会 resize 到 element_count
     * @return 成功返回 true；类型暂不支持返回 false
     */
    bool read_all(std::vector<float> &out) const;
};

// ============================================================================
// 单层权重（Attention 层 与 SSM 混合层 的并集）
// ============================================================================

/**
 * @brief 一个 Transformer 块的权重引用集合
 *
 * 该模型有两类块：
 *  - 纯 Attention 层（blk.3/7/11/15/19/23）：attn_q/k/v/output + attn_q/k_norm
 *  - SSM 混合层（其余）：ssm_* + attn_qkv/attn_gate
 * 公共部分（attn_norm / post_attention_norm / ffn_*）两类都有。
 * 某类层不存在的权重指针为 nullptr，用 is_attention() / is_ssm() 区分。
 */
struct GGUFBlockWeights {
    // ---- 公共：RMSNorm + SwiGLU FFN ----
    const GGUFTensorView *attn_norm = nullptr;           // 输入 RMSNorm
    const GGUFTensorView *post_attention_norm = nullptr; // 残差后 RMSNorm
    const GGUFTensorView *ffn_down = nullptr;            // FFN 下投影
    const GGUFTensorView *ffn_gate = nullptr;            // FFN 门控
    const GGUFTensorView *ffn_up = nullptr;              // FFN 上投影
    // ---- 纯 Attention 层 ----
    const GGUFTensorView *attn_q = nullptr;
    const GGUFTensorView *attn_k = nullptr;
    const GGUFTensorView *attn_v = nullptr;
    const GGUFTensorView *attn_output = nullptr;
    const GGUFTensorView *attn_q_norm = nullptr;
    const GGUFTensorView *attn_k_norm = nullptr;
    // ---- SSM 混合层 ----
    const GGUFTensorView *ssm_a = nullptr;       // SSM 状态矩阵
    const GGUFTensorView *ssm_conv1d = nullptr;  // 输入卷积
    const GGUFTensorView *ssm_dt_bias = nullptr; // 时间步偏置
    const GGUFTensorView *ssm_alpha = nullptr;   // 时间步 alpha
    const GGUFTensorView *ssm_beta = nullptr;    // 时间步 beta
    const GGUFTensorView *ssm_norm = nullptr;    // SSM 输出 RMSNorm
    const GGUFTensorView *ssm_out = nullptr;     // SSM 输出投影
    const GGUFTensorView *attn_qkv = nullptr;    // 混合层 QKV 合并投影
    const GGUFTensorView *attn_gate = nullptr;   // 混合层注意力门控

    /**
     * @brief 判断是否为纯 Attention 层
     * @return 是 Attention 层返回 true
     */
    bool is_attention() const {
        return attn_q != nullptr;
    }

    /**
     * @brief 判断是否为 SSM 混合层
     * @return 是 SSM 混合层返回 true
     */
    bool is_ssm() const {
        return ssm_a != nullptr;
    }
};

// ============================================================================
// 模型权重容器
// ============================================================================

/**
 * @brief 模型权重容器：按名索引全部张量 + 配置 + 逐层权重组装
 */
class GGUFModelWeights {
  public:
    /**
     * @brief 从解析并 mmap 后的 GGUFModel 构建权重索引
     * @param model 已 load 且 data 已 map_data 的模型
     * @return 成功返回 true；数据区未映射或无张量返回 false
     */
    bool build(const GGUFModel &model);

    /**
     * @brief 按名称查找张量
     * @param name 张量名称（如 "token_embd.weight"）
     * @return 找到返回对应视图指针；未找到返回 nullptr
     */
    const GGUFTensorView *find(const std::string &name) const;

    /**
     * @brief 获取模型配置
     * @return 解析出的配置（只读）
     */
    const GGUFModelConfig &config() const {
        return config_;
    }

    /**
     * @brief 获取逐层权重
     * @return 每层的权重引用集合（数量 = block_count）
     */
    const std::vector<GGUFBlockWeights> &blocks() const {
        return blocks_;
    }

    /**
     * @brief 获取词嵌入权重（token_embd.weight）
     * @return 词嵌入张量；缺失返回 nullptr
     */
    const GGUFTensorView *token_embd() const {
        return token_embd_;
    }

    /**
     * @brief 获取输出归一化权重（output_norm.weight）
     * @return 输出 RMSNorm 权重；缺失返回 nullptr
     */
    const GGUFTensorView *output_norm() const {
        return output_norm_;
    }

    /**
     * @brief 获取已索引的张量总数
     * @return 张量视图数量
     */
    std::size_t count() const {
        return views_.size();
    }

  private:
    GGUFModelConfig config_;                                // 模型配置
    std::unordered_map<std::string, GGUFTensorView> views_; // 名称 → 张量视图
    std::vector<GGUFBlockWeights> blocks_;                  // 逐层权重
    const GGUFTensorView *token_embd_ = nullptr;            // 词嵌入
    const GGUFTensorView *output_norm_ = nullptr;           // 输出归一化
};
