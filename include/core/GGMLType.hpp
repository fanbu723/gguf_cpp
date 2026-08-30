/*
 * GGMLType.hpp — GGML 数据类型描述（阶段 ② 第 2 步）
 *
 * GGUF 文件里每个张量的 data_type 是一个 GGML 类型编号（uint32_t）。
 * 本模块提供：编号 → 名称 / 块大小 / 块字节数 / 张量总字节数 的映射，
 * 是"定位任意张量数据"和后续"反量化"的基础。
 *
 * 说明：量化类型按"块(block)"打包 —— 每个 block 固定装 block_size 个元素、
 *       占 type_size 字节；非量化类型（F32/F16/BF16...）block_size = 1。
 *       张量总字节数 = ceil(元素数 / block_size) × type_size。
 */

#pragma once

#include <cstdint>

// GGML 类型编号（与 ggml.h 一致）
enum class GGMLType : std::uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    F64 = 20,
    BF16 = 21,
    Q4_0_4_4 = 22,
    Q4_0_4_8 = 23,
    Q4_0_8_8 = 24,
    TQ1_0 = 25,
    TQ2_0 = 26,
    IQ2_XXS = 27,
    IQ2_XS = 28,
    IQ3_XXS = 29,
    IQ1_S = 30,
    IQ4_NL = 31,
    IQ3_S = 32,
    IQ2_S = 33,
    IQ4_XS = 34,
    I8_S = 35,
    IQ3_NL = 36,
    IQ4_XXS = 37,
    IQ2_M = 38,
};

/**
 * @brief GGML 类型编号 → 类型名称
 * @param type GGML 类型编号
 * @return 类型名称（"F32" / "Q4_0" / ...；未知返回 "?"）
 */
const char *GGMLTypeName(std::uint32_t type);

/**
 * @brief 查询每个 block 的元素数
 * @param type GGML 类型编号
 * @return 非量化类型为 1；未知返回 1
 */
std::uint32_t GGMLBlockSize(std::uint32_t type);

/**
 * @brief 查询每个 block 的字节数
 * @param type GGML 类型编号
 * @return F32=4、BF16=2、Q4_0=18...；未知返回 0
 */
std::uint32_t GGMLTypeSize(std::uint32_t type);

/**
 * @brief 计算张量总字节数（按整 block 向上取整）
 * @param type GGML 类型编号
 * @param nelements 元素总数
 * @return 张量占用的总字节数；未知类型返回 0
 */
std::uint64_t GGMLBytes(std::uint32_t type, std::uint64_t nelements);
