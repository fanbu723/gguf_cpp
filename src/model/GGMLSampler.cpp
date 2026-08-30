#include "GGMLSampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {

// 数值稳定的 softmax（返回概率分布）
std::vector<float> softmax(const std::vector<float> &logits, float temperature) {
    std::vector<float> p(logits.size());
    const float maxv = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        p[i] = std::exp((logits[i] - maxv) / temperature);
        sum += p[i];
    }
    for (auto &x : p)
        x /= sum;
    return p;
}

// top-k：只保留概率最大的前 k 个，其余置 0
void apply_top_k(std::vector<float> &p, int top_k) {
    if (top_k <= 0 || static_cast<int>(p.size()) <= top_k)
        return;
    // 找第 top_k 大的概率作为阈值
    std::vector<float> sorted = p;
    std::nth_element(sorted.begin(), sorted.begin() + (top_k - 1), sorted.end(),
                     std::greater<float>());
    const float thr = sorted[static_cast<std::size_t>(top_k - 1)];
    for (auto &x : p)
        if (x < thr)
            x = 0.0f;
}

// top-p（nucleus）：按概率降序累积，保留累计 ≤ p 的最小集合，其余置 0
void apply_top_p(std::vector<float> &p, float top_p) {
    if (top_p >= 1.0f)
        return;
    std::vector<std::size_t> idx(p.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return p[a] > p[b]; });
    float cum = 0.0f;
    for (std::size_t n = 0; n < idx.size(); ++n) {
        cum += p[idx[n]];
        if (cum > top_p) {
            for (std::size_t m = n + 1; m < idx.size(); ++m)
                p[idx[m]] = 0.0f;
            break;
        }
    }
}

} // namespace

int GGMLSampler::sample(const std::vector<float> &logits) {
    if (logits.empty())
        return 0;

    // 贪心：直接 argmax（temperature ≤ 0 也退化为贪心）
    if (mode == GGMLSampleMode::GREEDY || temperature <= 0.0f) {
        return static_cast<int>(
            std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    // temperature 缩放 + softmax
    std::vector<float> p = softmax(logits, temperature);

    // top-k / top-p（组合时先 k 后 p）
    if (mode == GGMLSampleMode::TOP_K || mode == GGMLSampleMode::TOP_K_P)
        apply_top_k(p, top_k);
    if (mode == GGMLSampleMode::TOP_P || mode == GGMLSampleMode::TOP_K_P)
        apply_top_p(p, top_p);

    // 从最终分布采样（discrete_distribution 内部会再归一化）
    std::discrete_distribution<int> dist(p.begin(), p.end());
    return dist(rng);
}
