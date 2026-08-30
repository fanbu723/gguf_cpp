#include "GGMLType.hpp"

#include <cstddef>
#include <cstdint>

namespace {

// 类型信息表：按编号索引 {名称, block_size, type_size}
// 编号 4、5 已被规范移除，留空占位
struct GGMLTypeInfo {
    const char *name;
    std::uint32_t block_size;
    std::uint32_t type_size;
};

constexpr GGMLTypeInfo kTypes[] = {
    {"F32", 1, 4},          // 0
    {"F16", 1, 2},          // 1
    {"Q4_0", 32, 18},       // 2
    {"Q4_1", 32, 20},       // 3
    {nullptr, 0, 0},        // 4（已移除）
    {nullptr, 0, 0},        // 5（已移除）
    {"Q5_0", 32, 22},       // 6
    {"Q5_1", 32, 24},       // 7
    {"Q8_0", 32, 34},       // 8
    {"Q8_1", 32, 40},       // 9
    {"Q2_K", 256, 84},      // 10
    {"Q3_K", 256, 110},     // 11
    {"Q4_K", 256, 144},     // 12
    {"Q5_K", 256, 176},     // 13
    {"Q6_K", 256, 210},     // 14
    {"Q8_K", 256, 292},     // 15
    {"I8", 1, 1},           // 16
    {"I16", 1, 2},          // 17
    {"I32", 1, 4},          // 18
    {"I64", 1, 8},          // 19
    {"F64", 1, 8},          // 20
    {"BF16", 1, 2},         // 21
    {"Q4_0_4_4", 256, 144}, // 22
    {"Q4_0_4_8", 256, 144}, // 23
    {"Q4_0_8_8", 256, 144}, // 24
    {"TQ1_0", 256, 34},     // 25
    {"TQ2_0", 256, 68},     // 26
    {"IQ2_XXS", 256, 38},   // 27
    {"IQ2_XS", 256, 56},    // 28
    {"IQ3_XXS", 256, 44},   // 29
    {"BF16", 1, 2},         // 30（实测本模型：2 字节/元素；注：标准 ggml 该编号为 IQ1_S）
    {"IQ4_NL", 32, 18},     // 31
    {"IQ3_S", 256, 52},     // 32
    {"IQ2_S", 256, 48},     // 33
    {"IQ4_XS", 256, 88},    // 34
    {"I8_S", 256, 32},      // 35
    {"IQ3_NL", 32, 20},     // 36
    {"IQ4_XXS", 256, 64},   // 37
    {"IQ2_M", 256, 58},     // 38
};

constexpr std::size_t kTypeCount = sizeof(kTypes) / sizeof(kTypes[0]);

} // namespace

const char *GGMLTypeName(std::uint32_t type) {
    if (type < kTypeCount && kTypes[type].name != nullptr)
        return kTypes[type].name;
    return "?";
}

std::uint32_t GGMLBlockSize(std::uint32_t type) {
    if (type < kTypeCount && kTypes[type].block_size != 0)
        return kTypes[type].block_size;
    return 1;
}

std::uint32_t GGMLTypeSize(std::uint32_t type) {
    if (type < kTypeCount && kTypes[type].type_size != 0)
        return kTypes[type].type_size;
    return 0;
}

std::uint64_t GGMLBytes(std::uint32_t type, std::uint64_t nelements) {
    const std::uint32_t block = GGMLBlockSize(type);
    const std::uint32_t size = GGMLTypeSize(type);
    if (block == 0 || size == 0)
        return 0;
    // 向上取整到整 block：每个 block 装 block 个元素、占 size 字节
    return ((nelements + block - 1) / block) * size;
}
