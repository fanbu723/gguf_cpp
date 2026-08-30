
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

#include "GGUFLoader.hpp"

std::string model_path = "/home/dongfan/llm/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf";

int main() {
    GGUFModel model;

    if (!GGUFLoader::load(model_path, model)) {
        std::cerr << "❌ 加载 GGUF 文件失败: " << model_path << std::endl;
        return 1;
    }

    std::cout << "✅ 加载成功" << std::endl;
    std::cout << "  魔数: 0x" << std::hex << model.header.magic << std::dec << std::endl;
    std::cout << "  版本: " << model.header.version << std::endl;
    std::cout << "  元数据 KV 数量: " << model.header.metadata_kv_count << std::endl;
    std::cout << "  张量数量: " << model.header.tensor_count << std::endl;

    std::cout << "\n=== 元数据 KV 列表（前 20 条）===" << std::endl;
    const std::size_t meta_preview = std::min<std::size_t>(model.metadata.size(), 20);
    for (std::size_t i = 0; i < meta_preview; ++i) {
        const auto &kv = model.metadata[i];
        std::cout << "  [" << GGUFValueTypeName(kv.value_type) << "] " << kv.key << " = ";
        printMetadataValue(std::cout, kv.value);
        std::cout << std::endl;
    }
    if (model.metadata.size() > meta_preview) {
        std::cout << "  ... 共 " << model.metadata.size() << " 条" << std::endl;
    }

    std::cout << "\n=== 张量信息表（前 10 个）===" << std::endl;
    const std::size_t t_preview = std::min<std::size_t>(model.tensors.size(), 10);
    for (std::size_t i = 0; i < t_preview; ++i) {
        const auto &t = model.tensors[i];
        std::cout << "  " << t.name << "  dims=[";
        for (std::size_t d = 0; d < t.dimensions.size(); ++d) {
            if (d)
                std::cout << ", ";
            std::cout << t.dimensions[d];
        }
        std::cout << "]  type=" << t.data_type << "  offset=" << t.offset
                  << "  elements=" << t.element_count() << std::endl;
    }
    if (model.tensors.size() > t_preview) {
        std::cout << "  ... 共 " << model.tensors.size() << " 个张量" << std::endl;
    }

    std::cout << "\n=== 张量数据区 ===" << std::endl;
    std::cout << "  起始偏移: " << model.data.data_offset << std::endl;
    std::cout << "  总大小: " << model.data.data_size << " 字节 (" << std::fixed
              << std::setprecision(2)
              << static_cast<double>(model.data.data_size) / (1024.0 * 1024.0) << " MiB)"
              << std::endl;
    std::cout << "  数据指针: " << (model.data.data_ptr ? "已挂载" : "未挂载（延迟加载，待 mmap）")
              << std::endl;
    return 0;
}
