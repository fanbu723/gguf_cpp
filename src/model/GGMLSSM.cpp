#include "GGMLSSM.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "GGMLFFN.hpp"
#include "GGMLNorm.hpp"
#include "GGMLOps.hpp"

namespace {

// 从 GGUFTensorView 反量化整张张量到 vector；失败返回空 vector
std::vector<float> load_tensor(const GGUFTensorView *v) {
    std::vector<float> buf;
    if (v)
        v->read_all(buf);
    return buf;
}

// L2 归一化：scale = 1/max(||x||, eps)，逐 head 128 维（llama.cpp ggml_l2_norm）
void l2_norm(const float *x, float *y, int n, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i)
        sum_sq += x[i] * x[i];
    const float norm = std::sqrt(sum_sq);
    const float scale = 1.0f / std::max(norm, eps);
    for (int i = 0; i < n; ++i)
        y[i] = x[i] * scale;
}

// softplus
inline float softplus(float u) {
    return std::log1p(std::exp(u));
}

} // namespace

// ---------------------------------------------------------------------------
// GGMLSSMState
// ---------------------------------------------------------------------------

void GGMLSSMState::init(int n_v_heads, int d_state, int conv_dim) {
    this->n_v_heads = n_v_heads;
    this->d_state = d_state;
    this->conv_dim = conv_dim;
    reset();
}

void GGMLSSMState::reset() {
    S.assign(static_cast<std::size_t>(n_v_heads) * d_state * d_state, 0.0f);
    conv_hist.assign(static_cast<std::size_t>(3) * conv_dim, 0.0f);
}

// ---------------------------------------------------------------------------
// Gated Delta Net 单步递推
// ---------------------------------------------------------------------------

void GGMLDeltaNetStep(const float *q, const float *k, const float *v, float beta, float phi,
                      float *S, int Sv, float *o) {
    // 1. kv[j] = Σ_i S[i][j]·k[i]（Sᵀk）
    std::vector<float> kv(static_cast<std::size_t>(Sv));
    for (int j = 0; j < Sv; ++j) {
        float s = 0.0f;
        for (int i = 0; i < Sv; ++i)
            s += S[static_cast<std::size_t>(i) * Sv + j] * k[i];
        kv[static_cast<std::size_t>(j)] = s;
    }
    // 2. S = φ·S + β·k ⊗ (v − φ·kv)；o = Sᵀq / √Sv
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(Sv));
    for (int j = 0; j < Sv; ++j) {
        const float delta = beta * (v[j] - phi * kv[static_cast<std::size_t>(j)]);
        float s = 0.0f;
        for (int i = 0; i < Sv; ++i) {
            float &Sij = S[static_cast<std::size_t>(i) * Sv + j];
            Sij = phi * Sij + k[i] * delta; // 原地更新
            s += Sij * q[i];
        }
        o[j] = s * inv_sqrt;
    }
}

// ---------------------------------------------------------------------------
// SSM 层完整前向（解码单 token）
// ---------------------------------------------------------------------------

void GGMLSSMLayer(const GGUFBlockWeights &w, const GGUFModelConfig &cfg, GGMLSSMState &state,
                  const float *x, float *y) {
    const int hidden = static_cast<int>(cfg.embedding_length);
    const int d_inner = static_cast<int>(cfg.ssm_inner_size);      // 2048
    const int d_state = static_cast<int>(cfg.ssm_state_size);      // 128
    const int n_group = static_cast<int>(cfg.ssm_group_count);     // 16（k/v head）
    const int key_dim = d_state * n_group;                         // 2048
    const int value_dim = d_inner;                                 // 2048
    const int conv_dim = 2 * key_dim + value_dim;                  // 6144
    const int conv_kernel = static_cast<int>(cfg.ssm_conv_kernel); // 4
    const int ffn_hidden = static_cast<int>(cfg.feed_forward_length);

    // ---- 反量化本层权重 ----
    const std::vector<float> qkv_w = load_tensor(w.attn_qkv);      // [conv_dim, hidden]
    const std::vector<float> gate_w = load_tensor(w.attn_gate);    // [d_inner, hidden]
    const std::vector<float> alpha_w = load_tensor(w.ssm_alpha);   // [n_group, hidden]
    const std::vector<float> beta_w = load_tensor(w.ssm_beta);     // [n_group, hidden]
    const std::vector<float> out_w = load_tensor(w.ssm_out);       // [hidden, d_inner]
    const std::vector<float> a = load_tensor(w.ssm_a);             // [n_group]
    const std::vector<float> dt_bias = load_tensor(w.ssm_dt_bias); // [n_group]
    const std::vector<float> conv1d_w =
        load_tensor(w.ssm_conv1d); // [conv_dim, conv_kernel] 实际 [conv_kernel, conv_dim] 列主序
    const std::vector<float> ssm_norm = load_tensor(w.ssm_norm); // [d_inner] 即 [n_group*d_state]
    const std::vector<float> a_norm = load_tensor(w.attn_norm);  // [hidden]
    const std::vector<float> p_norm = load_tensor(w.post_attention_norm); // [hidden]
    const std::vector<float> f_gate = load_tensor(w.ffn_gate);            // [ffn_hidden, hidden]
    const std::vector<float> f_up = load_tensor(w.ffn_up);
    const std::vector<float> f_down = load_tensor(w.ffn_down); // [hidden, ffn_hidden]

    const std::size_t hid = static_cast<std::size_t>(hidden);
    const std::size_t cd = static_cast<std::size_t>(conv_dim);
    const std::size_t ds = static_cast<std::size_t>(d_state);
    const std::size_t ng = static_cast<std::size_t>(n_group);

    std::vector<float> xn(hid);
    std::vector<float> qkv(cd), qkv_conv(cd), z(static_cast<std::size_t>(d_inner));
    std::vector<float> alpha(ng), beta(ng), qbuf(ng * ds), kbuf(ng * ds), vbuf(ng * ds);
    std::vector<float> o(ng * ds), out(static_cast<std::size_t>(d_inner));
    std::vector<float> tmp(hid);

    // 1. 输入 RMSNorm
    GGMLRmsNorm(x, a_norm.data(), xn.data(), hidden, cfg.rms_eps);

    // 2. qkv / z / alpha / beta 投影（并行投影，注意都用 x_norm）
    GGMLGemmVec(qkv_w.data(), xn.data(), qkv.data(), conv_dim, hidden);
    GGMLGemmVec(gate_w.data(), xn.data(), z.data(), d_inner, hidden);
    GGMLGemmVec(alpha_w.data(), xn.data(), alpha.data(), n_group, hidden);
    GGMLGemmVec(beta_w.data(), xn.data(), beta.data(), n_group, hidden);

    // 3. 因果 depthwise conv1d + SiLU（当前 token 用最近 conv_kernel-1 个历史 + 自身）
    //    conv1d_w 列主序 dims=[conv_kernel, conv_dim] → 每通道 conv_kernel 个 tap 连续
    for (int c = 0; c < conv_dim; ++c) {
        float acc = 0.0f;
        for (int j = 0; j < conv_kernel; ++j) {
            const float u = (j == conv_kernel - 1)
                                ? qkv[static_cast<std::size_t>(c)]
                                : state.conv_hist[static_cast<std::size_t>(j) * cd + c];
            acc += u * conv1d_w[static_cast<std::size_t>(c) * conv_kernel + j];
        }
        qkv_conv[static_cast<std::size_t>(c)] = GGMLSiLU(acc);
    }
    // 更新 conv 历史（推进一 token）：历史槽有 conv_kernel-1 个，
    // 整体左移 conv_kernel-2 次（丢最旧），最后写入当前原始 qkv。
    for (int j = 0; j < conv_kernel - 2; ++j)
        for (int c = 0; c < conv_dim; ++c)
            state.conv_hist[static_cast<std::size_t>(j) * cd + c] =
                state.conv_hist[static_cast<std::size_t>(j + 1) * cd + c];
    for (int c = 0; c < conv_dim; ++c)
        state.conv_hist[static_cast<std::size_t>(conv_kernel - 2) * cd + c] =
            qkv[static_cast<std::size_t>(c)];

    // 4. 拆 q/k/v（通道顺序 q 前、k 中、v 后），q/k 每 head 做 L2 归一化
    //    conv 不混合通道，故切法与投影输出一致
    for (int h = 0; h < n_group; ++h) {
        const float *qh = qkv_conv.data() + static_cast<std::size_t>(h) * ds;
        const float *kh =
            qkv_conv.data() + static_cast<std::size_t>(key_dim) + static_cast<std::size_t>(h) * ds;
        const float *vh = qkv_conv.data() + static_cast<std::size_t>(key_dim + key_dim) +
                          static_cast<std::size_t>(h) * ds;
        l2_norm(qh, qbuf.data() + static_cast<std::size_t>(h) * ds, d_state, cfg.rms_eps);
        l2_norm(kh, kbuf.data() + static_cast<std::size_t>(h) * ds, d_state, cfg.rms_eps);
        std::copy(vh, vh + ds, vbuf.begin() + static_cast<std::ptrdiff_t>(h) * ds);
    }

    // 5. Gated Delta Net 递推（每 v-head h：β/φ 按 head 广播，状态 S[h] 跨 token 持久）
    for (int h = 0; h < n_group; ++h) {
        const float beta_h =
            1.0f / (1.0f + std::exp(-beta[static_cast<std::size_t>(h)])); // sigmoid
        const float gate =
            a[static_cast<std::size_t>(h)] *
            softplus(alpha[static_cast<std::size_t>(h)] + dt_bias[static_cast<std::size_t>(h)]);
        const float phi = std::exp(gate);
        float *S = state.S.data() + static_cast<std::size_t>(h) * ds * ds;
        const float *q = qbuf.data() + static_cast<std::size_t>(h) * ds;
        const float *k = kbuf.data() + static_cast<std::size_t>(h) * ds;
        const float *v = vbuf.data() + static_cast<std::size_t>(h) * ds;
        float *o_h = o.data() + static_cast<std::size_t>(h) * ds;
        GGMLDeltaNetStep(q, k, v, beta_h, phi, S, d_state, o_h);
    }

    // 6. gated norm：out = RMSNorm(o, γ_ssm_norm) ⊙ SiLU(z)，按 head 维对齐
    //    o 布局 [n_group][d_state]，z 布局 [n_group][d_state]（z 为 d_inner = n_group×d_state）
    for (int h = 0; h < n_group; ++h) {
        const float *o_h = o.data() + static_cast<std::size_t>(h) * ds;
        const float *z_h = z.data() + static_cast<std::size_t>(h) * ds;
        const float *g = ssm_norm.data() + static_cast<std::size_t>(h) * ds;
        std::vector<float> rms(ds);
        GGMLRmsNorm(o_h, g, rms.data(), d_state, cfg.rms_eps);
        for (int d = 0; d < d_state; ++d)
            out[static_cast<std::size_t>(h) * ds + d] =
                rms[static_cast<std::size_t>(d)] * GGMLSiLU(z_h[d]);
    }

    // 7. 输出投影 + 残差①：h1 = x + W_out·out
    GGMLGemmVec(out_w.data(), out.data(), tmp.data(), hidden, d_inner);
    for (int i = 0; i < hidden; ++i)
        y[i] = x[i] + tmp[static_cast<std::size_t>(i)];

    // 8. post RMSNorm + SwiGLU FFN + 残差②：h_out = h1 + FFN(RMSNorm(h1))
    GGMLRmsNorm(y, p_norm.data(), xn.data(), hidden, cfg.rms_eps);
    GGMLSwiGLU(f_gate.data(), f_up.data(), f_down.data(), xn.data(), tmp.data(), hidden,
               ffn_hidden);
    for (int i = 0; i < hidden; ++i)
        y[i] += tmp[static_cast<std::size_t>(i)];
}
