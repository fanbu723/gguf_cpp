
#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "GGMLType.hpp"
#include "GGUFLoader.hpp"

std::string model_path = "/home/dongfan/llm/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf";

int main() {
    GGUFModel model;

    if (!GGUFLoader::load(model_path, model)) {
        std::cerr << "❌ 加载 GGUF 文件失败: " << model_path << std::endl;
        return 1;
    }

    std::cout << "\n=== GGML 类型系统验证 ===" << std::endl;

    // ① 按类型计算每个张量的字节数并累加
    std::uint64_t total_bytes = 0;
    for (const auto &t : model.tensors)
        total_bytes += GGMLBytes(t.data_type, t.element_count());

    // ② 最后一个张量之后可能还有文件尾部填充（不属于任何张量）
    const auto &last = model.tensors.back();
    const std::uint64_t last_end = last.offset + GGMLBytes(last.data_type, last.element_count());
    const std::uint64_t trailing =
        (last_end <= model.data.data_size) ? model.data.data_size - last_end : 0;

    const bool ok = (total_bytes + trailing) == model.data.data_size;
    std::cout << "  ① 张量数据累加: " << total_bytes << std::endl;
    std::cout << "  ② 尾部填充: " << trailing << std::endl;
    std::cout << "  数据区实际大小: " << model.data.data_size << std::endl;
    std::cout << "  ① + ② 与文件吻合: " << (ok ? "✅ 完全一致" : "❌ 不一致") << std::endl;

    // 诊断：从张量 offset 反推每种 data_type 的真实 字节/元素（校准类型表用）
    // 注意：tensor.offset 是相对数据区起点的偏移，范围 [0, data_size)
    std::cout << "\n=== 类型诊断：反推 字节/元素 ===" << std::endl;
    std::map<std::uint32_t, std::pair<std::uint64_t, std::uint64_t>> stat;
    for (std::size_t i = 0; i < model.tensors.size(); ++i) {
        const auto &t = model.tensors[i];
        const std::uint64_t end = (i + 1 < model.tensors.size())
                                      ? model.tensors[i + 1].offset
                                      : model.data.data_size; // 最后一个到数据区末尾
        stat[t.data_type].first += end - t.offset;
        stat[t.data_type].second += t.element_count();
    }
    for (const auto &[type, p] : stat) {
        std::cout << "  type=" << type << "(" << GGMLTypeName(type) << ")"
                  << "  字节/元素=" << static_cast<double>(p.first) / static_cast<double>(p.second)
                  << "  总字节=" << p.first << std::endl;
    }

    return ok ? 0 : 1; // 校验失败时返回非零，让 CTest 判定测试失败
}
