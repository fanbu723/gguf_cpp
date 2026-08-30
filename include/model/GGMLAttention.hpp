/*
 * GGMLAttention.hpp — KV cache + GQA 注意力（阶段 ⑤）
 *
 * GQA（Grouped-Query Attention）：
 *   有 n_head 个 Q 头、n_head_kv 个 KV 头（n_head_kv ≤ n_head），
 *   每 (n_head / n_head_kv) 个 Q 头共享同一个 KV 头。
 *   相比 MHA 减少 KV cache 内存与访存，是 Qwen 等主流模型的标配。
 *
 * KV cache：把已生成 token 的 K/V 按位置缓存，自回归时只需计算当前 token
 *   的注意力，无需重算历史（自回归关键优化）。
 *
 * 本模块是"纯数学"实现，不依赖权重文件，便于单元测试。
 */

#pragma once

#include <vector>

/**
 * @brief KV cache：缓存各 KV 头在各位置上的 K / V 向量
 */
struct GGMLKVCache {
    int n_kv_heads = 0;   // KV 头数
    int head_dim_k = 0;   // 每个 KV 头的 key 维度
    int head_dim_v = 0;   // 每个 KV 头的 value 维度
    int max_len = 0;      // 最大缓存长度
    int len = 0;          // 当前已缓存的位置数
    std::vector<float> k; // [kv_head][pos][dim_k]
    std::vector<float> v; // [kv_head][pos][dim_v]

    /**
     * @brief 初始化缓存（分配内存）
     * @param n_kv_heads KV 头数
     * @param head_dim_k key 维度
     * @param head_dim_v value 维度
     * @param max_len 最大长度
     */
    void init(int n_kv_heads, int head_dim_k, int head_dim_v, int max_len);

    /**
     * @brief 存入某个 KV 头在 pos 位置的 K / V
     * @param kv_head KV 头下标
     * @param pos 位置（通常等于当前 len）
     * @param k_vec 该头的 K 向量（head_dim_k 个元素）
     * @param v_vec 该头的 V 向量（head_dim_v 个元素）
     */
    void store(int kv_head, int pos, const float *k_vec, const float *v_vec);

    /**
     * @brief 取某个 KV 头在 pos 位置的 K 向量
     * @param kv_head KV 头下标
     * @param pos 位置
     * @return K 向量指针（head_dim_k 个元素）
     */
    const float *k_at(int kv_head, int pos) const;

    /**
     * @brief 取某个 KV 头在 pos 位置的 V 向量
     * @param kv_head KV 头下标
     * @param pos 位置
     * @return V 向量指针（head_dim_v 个元素）
     */
    const float *v_at(int kv_head, int pos) const;
};

/**
 * @brief GQA 注意力（自回归：计算新 token 相对已缓存序列的注意力输出）
 * @param q 当前 token 的 Q，[n_head × head_dim_q]（已应用 RoPE）
 * @param cache KV cache（已含历史所有位置的 K/V）
 * @param n_head Q 头数
 * @param n_head_kv KV 头数
 * @param scale 缩放系数（通常 1/sqrt(head_dim_q)）
 * @param out 输出，[n_head × head_dim_v]（各头结果按顺序拼接）
 *
 * 对每个 Q 头 h：kv_head = h / (n_head / n_head_kv)，
 * 计算 q·k 得分 → softmax → 加权求和 V。
 */
void GGMLAttentionGQA(const float *q, const GGMLKVCache &cache, int n_head, int n_head_kv,
                      float scale, float *out);
