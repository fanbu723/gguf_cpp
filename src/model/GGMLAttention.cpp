#include "GGMLAttention.hpp"

#include "GGMLOps.hpp"

void GGMLKVCache::init(int n_kv_heads, int head_dim_k, int head_dim_v, int max_len) {
    this->n_kv_heads = n_kv_heads;
    this->head_dim_k = head_dim_k;
    this->head_dim_v = head_dim_v;
    this->max_len = max_len;
    this->len = 0;
    k.assign(static_cast<std::size_t>(n_kv_heads) * max_len * head_dim_k, 0.0f);
    v.assign(static_cast<std::size_t>(n_kv_heads) * max_len * head_dim_v, 0.0f);
}

void GGMLKVCache::store(int kv_head, int pos, const float *k_vec, const float *v_vec) {
    float *kdst = k.data() + (static_cast<std::size_t>(kv_head) * max_len + pos) * head_dim_k;
    float *vdst = v.data() + (static_cast<std::size_t>(kv_head) * max_len + pos) * head_dim_v;
    for (int d = 0; d < head_dim_k; ++d)
        kdst[d] = k_vec[d];
    for (int d = 0; d < head_dim_v; ++d)
        vdst[d] = v_vec[d];
    if (pos + 1 > len)
        len = pos + 1;
}

const float *GGMLKVCache::k_at(int kv_head, int pos) const {
    return k.data() + (static_cast<std::size_t>(kv_head) * max_len + pos) * head_dim_k;
}

const float *GGMLKVCache::v_at(int kv_head, int pos) const {
    return v.data() + (static_cast<std::size_t>(kv_head) * max_len + pos) * head_dim_v;
}

void GGMLAttentionGQA(const float *q, const GGMLKVCache &cache, int n_head, int n_head_kv,
                      float scale, float *out) {
    const int group = n_head / n_head_kv; // 每组 Q 头数（共享一个 KV 头）
    const int head_dim_q = cache.head_dim_k;
    const int head_dim_v = cache.head_dim_v;

    std::vector<float> scores(static_cast<std::size_t>(cache.len));

    for (int h = 0; h < n_head; ++h) {
        const int kv_head = h / group;
        const float *qh = q + static_cast<std::size_t>(h) * head_dim_q;

        // 1. 得分：scores[j] = q·k[j] * scale
        for (int j = 0; j < cache.len; ++j) {
            const float *kj = cache.k_at(kv_head, j);
            float dot = 0.0f;
            for (int d = 0; d < head_dim_q; ++d)
                dot += qh[d] * kj[d];
            scores[static_cast<std::size_t>(j)] = dot * scale;
        }

        // 2. softmax
        GGMLSoftmax(scores.data(), cache.len);

        // 3. 加权求和 V
        float *oh = out + static_cast<std::size_t>(h) * head_dim_v;
        for (int d = 0; d < head_dim_v; ++d)
            oh[d] = 0.0f;
        for (int j = 0; j < cache.len; ++j) {
            const float *vj = cache.v_at(kv_head, j);
            const float w = scores[static_cast<std::size_t>(j)];
            for (int d = 0; d < head_dim_v; ++d)
                oh[d] += w * vj[d];
        }
    }
}
