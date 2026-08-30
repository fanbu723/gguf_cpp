#include "GGMLForward.hpp"

#include <vector>

#include "GGMLDequantize.hpp"
#include "GGMLNorm.hpp"
#include "GGMLOps.hpp"
#include "GGMLTransformer.hpp"

void GGMLModelState::init(const GGUFModelWeights &w, int max_len) {
    const auto &cfg = w.config();
    const auto &blocks = w.blocks();
    kv.clear();
    ssm.clear();
    kv.resize(blocks.size());
    ssm.resize(blocks.size());
    const int n_kv = static_cast<int>(cfg.head_count_kv);
    const int key_len = static_cast<int>(cfg.key_length);
    const int val_len = static_cast<int>(cfg.value_length);
    const int n_group = static_cast<int>(cfg.ssm_group_count);
    const int d_state = static_cast<int>(cfg.ssm_state_size);
    const int key_dim = d_state * n_group;
    const int conv_dim = 2 * key_dim + static_cast<int>(cfg.ssm_inner_size);
    for (std::size_t il = 0; il < blocks.size(); ++il) {
        if (blocks[il].is_attention())
            kv[il].init(n_kv, key_len, val_len, max_len);
        else
            ssm[il].init(n_group, d_state, conv_dim);
    }
}

void GGMLForward(const GGUFModelWeights &w, GGMLModelState &state, int token, int pos,
                 std::vector<float> &logits) {
    const auto &cfg = w.config();
    const int hidden = static_cast<int>(cfg.embedding_length);
    const auto *embd = w.token_embd();
    if (!embd)
        return;
    const int vocab = static_cast<int>(embd->dims[1]);

    // 1. token embedding：token_embd 列主序 [hidden, vocab]，token 向量 = 第 token 列
    std::vector<float> h(static_cast<std::size_t>(hidden));
    for (int i = 0; i < hidden; ++i)
        embd->read_element(static_cast<std::uint64_t>(token) * hidden + i,
                           h[static_cast<std::size_t>(i)]);

    // 2. 逐层前向（in-place：层函数保证写 y 前不修改 x，x==y 安全）
    const auto &blocks = w.blocks();
    for (std::size_t il = 0; il < blocks.size(); ++il) {
        if (blocks[il].is_attention())
            GGMLTransformerAttentionBlock(blocks[il], cfg, state.kv[il], pos, h.data(), h.data());
        else
            GGMLSSMLayer(blocks[il], cfg, state.ssm[il], h.data(), h.data());
    }

    // 3. 输出归一化（output_norm）
    std::vector<float> hn(static_cast<std::size_t>(hidden));
    if (const auto *on = w.output_norm()) {
        std::vector<float> gamma;
        on->read_all(gamma);
        GGMLRmsNorm(h.data(), gamma.data(), hn.data(), hidden, cfg.rms_eps);
    } else {
        hn = h;
    }

    // 4. logits = hn · token_embdᵀ（共享 embedding 作输出投影）
    //    embedding 列主序 [hidden, vocab] → 等价行主序 [vocab, hidden]
    logits.assign(static_cast<std::size_t>(vocab), 0.0f);
    for (int v = 0; v < vocab; ++v) {
        float sum = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            float e = 0.0f;
            GGMLDequantizeOne(embd->type, embd->data, static_cast<std::uint64_t>(v) * hidden + i,
                              e);
            sum += e * hn[static_cast<std::size_t>(i)];
        }
        logits[static_cast<std::size_t>(v)] = sum;
    }
}
