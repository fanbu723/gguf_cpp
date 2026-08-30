// 临时工具：扫描 GGUF 所有张量，反量化后统计 NaN/Inf（通用，支持已实现的全部类型）
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "GGUFLoader.hpp"
#include "GGUFModelWeights.hpp"

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "用法: check_nan <model.gguf>" << std::endl;
        return 1;
    }
    const std::string path = argv[1];
    GGUFModel model;
    if (!GGUFLoader::load(path, model)) {
        std::cerr << "加载失败" << std::endl;
        return 1;
    }
    GGUFLoader::map_data(path, model);
    GGUFModelWeights w;
    w.build(model);

    std::uint64_t total_nan = 0, total_inf = 0;
    int bad_tensors = 0, checked = 0, skipped = 0;
    std::cout << "=== 全张量 NaN/Inf 扫描 ===" << std::endl;
    for (const auto &t : model.tensors) {
        const auto *v = w.find(t.name);
        if (!v) {
            ++skipped;
            continue;
        }
        std::vector<float> buf;
        if (!v->read_all(buf)) {
            std::cout << "  !! read_all 失败 type=" << v->type << " : " << t.name << std::endl;
            ++skipped;
            continue;
        }
        ++checked;
        std::uint64_t nan = 0, inf = 0;
        for (float f : buf) {
            if (std::isnan(f))
                ++nan;
            else if (std::isinf(f))
                ++inf;
        }
        total_nan += nan;
        total_inf += inf;
        if (nan || inf) {
            ++bad_tensors;
            std::cout << "  ⚠️ " << t.name << "  type=" << v->type << "  NaN=" << nan
                      << "  Inf=" << inf << std::endl;
        }
    }
    std::cout << "=== 汇总 ===" << std::endl;
    std::cout << "检查 " << checked << " 张量（跳过 " << skipped
              << "）  含 NaN/Inf 张量: " << bad_tensors << std::endl;
    std::cout << "NaN 总数: " << total_nan << "  Inf 总数: " << total_inf << std::endl;
    return 0;
}
