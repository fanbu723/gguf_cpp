
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <vector>

#include "GGUFLoader.hpp"

std::string model_path = "/home/dongfan/llm/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf";

int main() {
    GGUFHeader header;

    // 方式1：只读头部
    if (GGUFLoader::load_header(model_path, header)) {
        std::cout << "✅ 成功读取 GGUF 文件头" << std::endl;
        std::cout << "  魔数: 0x" << std::hex << header.magic << std::dec << std::endl;
        std::cout << "  版本: " << header.version << std::endl;
        std::cout << "  张量数量: " << header.tensor_count << std::endl;
        std::cout << "  元数据键值对数量: " << header.metadata_kv_count << std::endl;
    }

    // 方式2：只检查是否为 GGUF 文件
    if (GGUFLoader::is_gguf_file(model_path)) {
        std::cout << "✅ 这是一个有效的 GGUF 文件" << std::endl;
    }

    return 0;
}
