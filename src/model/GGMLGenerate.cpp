#include "GGMLGenerate.hpp"

#include <vector>

std::vector<int> GGMLGenerate(const GGUFModelWeights &w, GGMLModelState &state,
                              GGMLSampler &sampler, const std::vector<int> &prompt, int n_tokens,
                              int eos_id) {
    std::vector<int> out;
    if (prompt.empty() || n_tokens <= 0)
        return out;

    // 1. 预填充：对 prompt 每个 token 前向（pos 递增），只更新状态，不采样
    int last = prompt[0];
    for (int i = 0; i < static_cast<int>(prompt.size()); ++i) {
        std::vector<float> logits;
        GGMLForward(w, state, prompt[static_cast<std::size_t>(i)], i, logits);
        last = prompt[static_cast<std::size_t>(i)];
    }

    // 2. 自回归生成
    for (int i = 0; i < n_tokens; ++i) {
        const int pos = static_cast<int>(prompt.size()) + i;
        std::vector<float> logits;
        GGMLForward(w, state, last, pos, logits);
        const int next = sampler.sample(logits);
        if (eos_id >= 0 && next == eos_id)
            break; // 遇到结束 token，提前停止
        out.push_back(next);
        last = next;
    }
    return out;
}
