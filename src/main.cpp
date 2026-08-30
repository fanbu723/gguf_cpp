
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

    std::cout << "\n=== mmap 映射后读取张量数据 ===" << std::endl;
    if (GGUFLoader::map_data(model_path, model)) {
        std::cout << "  映射成功: data_ptr = " << static_cast<const void *>(model.data.data_ptr)
                  << "（映射 " << model.data.map_len << " 字节）" << std::endl;

        // 找一个 F32 张量（data_type == 0）演示读取
        const GGUFTensorInfo *target = nullptr;
        for (const auto &t : model.tensors) {
            if (t.data_type == 0) {
                target = &t;
                break;
            }
        }

        if (target) {
            // 张量数据在映射区中的位置 = data_ptr + (offset - data_offset)
            const std::uint8_t *raw =
                model.data.data_ptr + (target->offset - model.data.data_offset);
            const float *f = reinterpret_cast<const float *>(raw);

            std::cout << "  张量: " << target->name << "  elements=" << target->element_count()
                      << std::endl;
            std::cout << "  前 8 个值: ";
            const std::size_t n = std::min<std::size_t>(target->element_count(), 8);
            for (std::size_t i = 0; i < n; ++i)
                std::cout << f[i] << (i + 1 < n ? ", " : "");
            std::cout << std::endl;
        } else {
            std::cout << "  未找到 F32 张量" << std::endl;
        }

        GGUFLoader::unmap_data(model);
        std::cout << "  已 munmap 释放（data_ptr 已清空）" << std::endl;
    } else {
        std::cerr << "  ❌ map_data 失败" << std::endl;
    }
    return 0;
}
