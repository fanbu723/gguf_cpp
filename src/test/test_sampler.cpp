/*
 * test_sampler.cpp — 阶段⑥ 第 1 步：采样器单元测试（不依赖模型）
 *
 * 用合成 logits 验证：
 *  - greedy / temperature=0 → argmax
 *  - top_k=1 / top_p 极小 → argmax（只剩最高概率项）
 *  - 空 logits 安全
 *  - 相同 seed → 相同采样序列（可复现）
 */

#include <iostream>
#include <vector>

#include "GGMLSampler.hpp"

static int g_fail = 0;
static void check(bool ok, const char *msg) {
    if (ok)
        std::cout << "  ✅ " << msg << std::endl;
    else {
        std::cout << "  ❌ " << msg << std::endl;
        ++g_fail;
    }
}

int main() {
    std::cout << "=== 阶段⑥ 第1步：采样器单元测试 ===" << std::endl;

    GGMLSampler s;

    std::cout << "[1] 贪心 / 确定性采样" << std::endl;
    s.mode = GGMLSampleMode::GREEDY;
    check(s.sample({1, 3, 2}) == 1, "greedy: argmax=1");
    s.mode = GGMLSampleMode::TOP_K_P;
    s.temperature = 0.0f;
    check(s.sample({1, 3, 2}) == 1, "temperature=0 → 退化为 argmax");

    std::cout << "[2] top-k / top-p 截断" << std::endl;
    s.mode = GGMLSampleMode::TOP_K;
    s.temperature = 1.0f;
    s.top_k = 1; // 只保留最高概率 → 必返回 argmax
    check(s.sample({1, 3, 2}) == 1, "top_k=1 → 只剩 argmax");
    s.mode = GGMLSampleMode::TOP_P;
    s.top_k = 0;     // 禁用 top-k
    s.top_p = 0.01f; // 极小 → 只剩最高概率
    check(s.sample({1, 3, 2}) == 1, "top_p 极小 → 只剩 argmax");

    std::cout << "[3] 边界" << std::endl;
    check(s.sample({}) == 0, "空 logits → 返回 0（安全）");

    std::cout << "[4] 种子确定性（可复现）" << std::endl;
    GGMLSampler a, b;
    a.mode = b.mode = GGMLSampleMode::TOP_K_P;
    a.temperature = b.temperature = 1.0f;
    a.top_k = b.top_k = 5;
    a.top_p = b.top_p = 0.9f;
    const std::vector<float> lg = {0.5f, 1.5f, 2.0f, 0.1f, 3.0f, 1.0f};
    bool same = true;
    for (int i = 0; i < 50; ++i)
        if (a.sample(lg) != b.sample(lg))
            same = false;
    check(same, "相同 seed → 采样序列一致");

    std::cout << (g_fail == 0 ? "\n✅ 全部通过" : "\n❌ 有失败") << std::endl;
    return g_fail == 0 ? 0 : 1;
}
