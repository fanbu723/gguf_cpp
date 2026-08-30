#include "GGMLTransformer.hpp"

#include <cmath>
#include <vector>

#include "GGMLFFN.hpp"
#include "GGMLNorm.hpp"
#include "GGMLOps.hpp"
#include "GGMLRope.hpp"

namespace {

// 从 GGUFTensorView 反量化整张张量到 vector；失败返回空 vector
std::vector<float> load_tensor(const GGUFTensorView *v) {
    std::vector<float> buf;
    if (v)
        v->read_all(buf);
    return buf;
}

} // namespace

void GGMLTransformerAttentionBlock(const GGUFBlockWeights &w, const GGUFModelConfig &cfg,
                                   GGMLKVCache &cache, int pos, const float *x, float *y) {
    const int hidden = static_cast<int>(cfg.embedding_length);
    const int n_head = static_cast<int>(cfg.head_count);
    const int n_kv = static_cast<int>(cfg.head_count_kv);
    const int head_dim = static_cast<int>(cfg.key_length); // 256
    const int n_rot = static_cast<int>(cfg.rope_dimension_count);
    const int ffn_hidden = static_cast<int>(cfg.feed_forward_length);

    // ---- 反量化本层权重（每次调用重新反量化，学习阶段可接受）----
    const std::vector<float> q_w = load_tensor(w.attn_q);        // [2*head_dim*n_head, hidden]
    const std::vector<float> k_w = load_tensor(w.attn_k);        // [head_dim*n_kv, hidden]
    const std::vector<float> v_w = load_tensor(w.attn_v);        // [head_dim*n_kv, hidden]
    const std::vector<float> o_w = load_tensor(w.attn_output);   // [hidden, head_dim*n_head]
    const std::vector<float> q_norm = load_tensor(w.attn_q_norm); // [head_dim]
    const std::vector<float> k_norm = load_tensor(w.attn_k_norm); // [head_dim]
    const std::vector<float> a_norm = load_tensor(w.attn_norm);  // [hidden]
    const std::vector<float> p_norm = load_tensor(w.post_attention_norm); // [hidden]
    const std::vector<float> f_gate = load_tensor(w.ffn_gate);   // [ffn_hidden, hidden]
    const std::vector<float> f_up = load_tensor(w.ffn_up);       // [ffn_hidden, hidden]
    const std::vector<float> f_down = load_tensor(w.ffn_down);   // [hidden, ffn_hidden]

    const std::size_t hid = static_cast<std::size_t>(hidden);
    // qg 是联合 Q+gate 投影的输出：2 × n_head × head_dim（Q 与 gate 各 n_head×head_dim）
    std::vector<float> xn(hid), qg(static_cast<std::size_t>(2 * n_head * head_dim)),
        qbuf(static_cast<std::size_t>(n_head) * head_dim),
        gate(static_cast<std::size_t>(n_head) * head_dim);
    std::vector<float> kbuf(static_cast<std::size_t>(n_kv) * head_dim),
        vbuf(static_cast<std::size_t>(n_kv) * head_dim);
    std::vector<float> attn_out(static_cast<std::size_t>(n_head) * head_dim), tmp(hid);

    // 1. 输入 RMSNorm
    GGMLRmsNorm(x, a_norm.data(), xn.data(), hidden, cfg.rms_eps);

    // 2. 联合 Q+gate 投影：qg = x_norm · attn_qᵀ（[2*head_dim*n_head]）
    GGMLGemmVec(q_w.data(), xn.data(), qg.data(), 2 * head_dim * n_head, hidden);

    // 3. 拆分 Q（前一半）与 gate（后一半），Q 按头 reshape
    //    qg 布局：[n_head*head_dim 个 Q] + [n_head*head_dim 个 gate]
    for (int h = 0; h < n_head; ++h)
        for (int d = 0; d < head_dim; ++d) {
            qbuf[static_cast<std::size_t>(h) * head_dim + d] =
                qg[static_cast<std::size_t>(h) * head_dim + d];
            gate[static_cast<std::size_t>(h) * head_dim + d] =
                qg[static_cast<std::size_t>(n_head * head_dim + h * head_dim + d)];
        }

    // 4. Q 归一化 + RoPE
    for (int h = 0; h < n_head; ++h) {
        const float *qh = qbuf.data() + static_cast<std::size_t>(h) * head_dim;
        std::vector<float> qn(static_cast<std::size_t>(head_dim));
        GGMLRmsNorm(qh, q_norm.data(), qn.data(), head_dim, cfg.rms_eps);
        GGMLRopeNeox(qn.data(), qbuf.data() + static_cast<std::size_t>(h) * head_dim, head_dim,
                     n_rot, pos, cfg.rope_freq_base);
    }

    // 5. K/V 投影 + K 归一化 + RoPE，然后存入 KV cache
    GGMLGemmVec(k_w.data(), xn.data(), kbuf.data(), n_kv * head_dim, hidden);
    GGMLGemmVec(v_w.data(), xn.data(), vbuf.data(), n_kv * head_dim, hidden);
    for (int kv = 0; kv < n_kv; ++kv) {
        const float *kh = kbuf.data() + static_cast<std::size_t>(kv) * head_dim;
        const float *vh = vbuf.data() + static_cast<std::size_t>(kv) * head_dim;
        std::vector<float> kn(static_cast<std::size_t>(head_dim));
        GGMLRmsNorm(kh, k_norm.data(), kn.data(), head_dim, cfg.rms_eps);
        std::vector<float> krot(static_cast<std::size_t>(head_dim));
        GGMLRopeNeox(kn.data(), krot.data(), head_dim, n_rot, pos, cfg.rope_freq_base);
        cache.store(kv, pos, krot.data(), vh);
    }

    // 6. GQA 注意力（当前 token 相对全部缓存位置）
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    GGMLAttentionGQA(qbuf.data(), cache, n_head, n_kv, scale, attn_out.data());

    // 7. 门控：attn_out × sigmoid(gate)
    for (std::size_t i = 0; i < attn_out.size(); ++i)
        attn_out[i] *= 1.0f / (1.0f + std::exp(-gate[i]));

    // 8. 输出投影 + 残差：y = x + attn_out · attn_outputᵀ
    GGMLGemmVec(o_w.data(), attn_out.data(), tmp.data(), hidden, n_head * head_dim);
    for (int i = 0; i < hidden; ++i)
        y[i] = x[i] + tmp[static_cast<std::size_t>(i)];

    // 9. post RMSNorm + SwiGLU FFN + 残差
    GGMLRmsNorm(y, p_norm.data(), xn.data(), hidden, cfg.rms_eps);
    GGMLSwiGLU(f_gate.data(), f_up.data(), f_down.data(), xn.data(), tmp.data(), hidden, ffn_hidden);
    for (int i = 0; i < hidden; ++i)
        y[i] += tmp[static_cast<std::size_t>(i)];
}
