#include "GGMLDequantize.hpp"
#include "GGMLType.hpp"

#include <cstdint>
#include <cstring>

float GGMLBF16ToFloat(std::uint16_t bits) {
    // BF16 = float 的高 16 位 → 左移 16 位即得 float 位模式
    std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

float GGMLF16ToFloat(std::uint16_t bits) {
    const std::uint32_t sign = (bits >> 15) & 1u;
    const std::uint32_t exp = (bits >> 10) & 0x1Fu;
    const std::uint32_t frac = bits & 0x3FFu;
    std::uint32_t fbits;
    if (exp == 0) {
        if (frac == 0) {
            fbits = sign << 31; // ±0
        } else {
            // 次正规数：先归一化再转换
            std::uint32_t f = frac;
            std::int32_t e = -1;
            do {
                e++;
                f <<= 1;
            } while ((f & 0x400u) == 0);
            fbits = (sign << 31) | ((127 - 15 - e) << 23) | ((f & 0x3FFu) << 13);
        }
    } else if (exp == 0x1Fu) {
        fbits = (sign << 31) | 0x7F800000u | (frac << 13); // inf / nan
    } else {
        fbits = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    float f;
    std::memcpy(&f, &fbits, sizeof(f));
    return f;
}

bool GGMLDequantizeOne(std::uint32_t type, const std::uint8_t *data, std::uint64_t index,
                       float &out) {
    const std::uint32_t block = GGMLBlockSize(type);
    const std::uint32_t tsize = GGMLTypeSize(type);

    switch (type) {
    case 0: { // F32：直接按 4 字节 float 解释
        std::memcpy(&out, data + index * 4, 4);
        return true;
    }
    case 1: { // F16
        std::uint16_t h;
        std::memcpy(&h, data + index * 2, 2);
        out = GGMLF16ToFloat(h);
        return true;
    }
    case 21: // BF16
    case 30: // 本模型校准：30 也是 BF16（2 字节/元素）
    {
        std::uint16_t h;
        std::memcpy(&h, data + index * 2, 2);
        out = GGMLBF16ToFloat(h);
        return true;
    }
    case 2: { // Q4_0：block=32，每 block = 2字节 scale + 16字节 4bit 量化值
        const std::uint64_t blk = index / block;
        const std::uint32_t in = index % block;
        const std::uint8_t *b = data + blk * tsize;
        std::uint16_t d;
        std::memcpy(&d, b, 2); // fp16 scale
        const std::uint8_t nib = (in & 1) ? (b[2 + in / 2] >> 4) : (b[2 + in / 2] & 0x0F);
        const std::int8_t q = static_cast<std::int8_t>(nib) - 8; // 4bit 有符号偏移
        out = static_cast<float>(q) * GGMLF16ToFloat(d);
        return true;
    }
    case 8: { // Q8_0：block=32，每 block = 2字节 scale + 32字节 int8 量化值
        const std::uint64_t blk = index / block;
        const std::uint32_t in = index % block;
        const std::uint8_t *b = data + blk * tsize;
        std::uint16_t d;
        std::memcpy(&d, b, 2); // fp16 scale
        out = static_cast<float>(static_cast<std::int8_t>(b[2 + in])) * GGMLF16ToFloat(d);
        return true;
    }
    default:
        return false; // 暂不支持的量化类型
    }
}

bool GGMLDequantize(std::uint32_t type, const std::uint8_t *data, std::uint64_t nelements,
                    float *out) {
    for (std::uint64_t i = 0; i < nelements; ++i) {
        if (!GGMLDequantizeOne(type, data, i, out[i]))
            return false;
    }
    return true;
}
