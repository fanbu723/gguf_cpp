#include "GGUFLoader.hpp"
#include <fstream>
#include <iostream>

static constexpr uint32_t GGUF_MAGIC = 0x46554747;

bool GGUFLoader::load_header(const fs::path &filepath, GGUFHeader &header) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filepath << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char *>(&header), sizeof(GGUFHeader));

    if (file.gcount() != sizeof(GGUFHeader)) {
        std::cerr << "读取头部失败，期望 24 字节，实际读取 " << file.gcount() << " 字节"
                  << std::endl;
        return false;
    }

    if (!validate_magic(header.magic)) {
        std::cerr << "无效的 GGUF 文件：魔数错误 (0x" << std::hex << header.magic << std::dec << ")"
                  << std::endl;
        return false;
    }
    return true;
}

bool GGUFLoader::is_gguf_file(const fs::path &filepath) {
    GGUFHeader header;
    return load_header(filepath, header);
}

bool GGUFLoader::validate_magic(uint32_t magic) {
    return magic == GGUF_MAGIC;
}
