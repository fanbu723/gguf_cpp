/*
 * GGMLDequantize.hpp — GGML 张量反量化（阶段 ② 第 3 步）
 *
 * 把张量的原始字节按 data_type 反量化为 float：
 *  - 非量化类型（F32 / F16 / BF16）：直接按元素解释
 *  - 量化类型（Q4_0 / Q8_0 ...）：按 block 解包（scale + 量化值）
 *
 * 说明：BF16 是 float 的高 16 位（符号1 + 指数8 + 尾数7），
 *       反量化 = 把 16 位左移 16 位后按 float 解释。
 */

#pragma once

#include <cstdint>

/**
 * @brief F16（半精度）→ float
 * @param bits F16 的 16 位原始位模式
 * @return 转换后的 float 值
 */
float GGMLF16ToFloat(std::uint16_t bits);

/**
 * @brief BF16（brain float）→ float
 * @param bits BF16 的 16 位原始位模式
 * @return 转换后的 float 值
 */
float GGMLBF16ToFloat(std::uint16_t bits);

/**
 * @brief 反量化张量中的第 index 个元素
 * @param type GGML 数据类型
 * @param data 张量原始字节
 * @param index 元素下标（从 0 开始）
 * @param out 输出参数，接收反量化后的 float
 * @return 成功返回 true；该类型暂不支持返回 false（out 不变）
 */
bool GGMLDequantizeOne(std::uint32_t type, const std::uint8_t *data, std::uint64_t index,
                       float &out);

/**
 * @brief 反量化整张张量
 * @param type GGML 数据类型
 * @param data 张量原始字节
 * @param nelements 元素总数
 * @param out 输出数组，需预分配 nelements 个 float
 * @return 成功返回 true；失败（类型不支持）返回 false
 */
bool GGMLDequantize(std::uint32_t type, const std::uint8_t *data, std::uint64_t nelements,
                    float *out);
