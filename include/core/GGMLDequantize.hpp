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

// F16（半精度）→ float
float GGMLF16ToFloat(std::uint16_t bits);

// BF16（brain float）→ float
float GGMLBF16ToFloat(std::uint16_t bits);

// 反量化第 index 个元素（data 指向张量原始字节）。返回 false 表示该类型暂不支持
bool GGMLDequantizeOne(std::uint32_t type, const std::uint8_t *data, std::uint64_t index,
                       float &out);

// 反量化整张张量（nelements 个元素写入 out，out 需预分配 nelements 个 float）
bool GGMLDequantize(std::uint32_t type, const std::uint8_t *data, std::uint64_t nelements,
                    float *out);
