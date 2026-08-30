/*
 * test_dequantize.cpp — 反量化单元测试（不依赖真实模型文件）
 *
 * 用"构造的已知数据"验证：
 *  - GGMLBF16ToFloat / GGMLF16ToFloat 转换正确性
 *  - F32 / BF16 整张量反量化
 *  - Q4_0 / Q8_0 按 block 解包
 *  - 未知/未实现类型返回 false
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "GGMLDequantize.hpp"

static int g_fail = 0;

static void check(bool ok, const char *msg) {
    if (ok)
        std::cout << "  ✅ " << msg << std::endl;
    else {
        std::cout << "  ❌ " << msg << std::endl;
        ++g_fail;
    }
}

static bool near(float a, float b) {
    return std::fabs(a - b) < 1e-3f;
}

int main() {
    std::cout << "=== 反量化单元测试 ===" << std::endl;

    std::cout << "[1] BF16 → float" << std::endl;
    check(near(GGMLBF16ToFloat(0x3F80), 1.0f), "BF16 0x3F80 = 1.0");
    check(near(GGMLBF16ToFloat(0x4000), 2.0f), "BF16 0x4000 = 2.0");
    check(near(GGMLBF16ToFloat(0xBF80), -1.0f), "BF16 0xBF80 = -1.0");
    check(near(GGMLBF16ToFloat(0xC000), -2.0f), "BF16 0xC000 = -2.0");
    check(near(GGMLBF16ToFloat(0x0000), 0.0f), "BF16 0x0000 = 0.0");

    std::cout << "[2] F16 → float" << std::endl;
    check(near(GGMLF16ToFloat(0x3C00), 1.0f), "F16 0x3C00 = 1.0");
    check(near(GGMLF16ToFloat(0x4000), 2.0f), "F16 0x4000 = 2.0");
    check(near(GGMLF16ToFloat(0xC000), -2.0f), "F16 0xC000 = -2.0");
    check(near(GGMLF16ToFloat(0x3D00), 1.25f), "F16 0x3D00 = 1.25");

    std::cout << "[3] F32 反量化" << std::endl;
    {
        const float src[4] = {1.0f, -2.0f, 3.5f, 0.25f};
        const std::uint8_t *raw = reinterpret_cast<const std::uint8_t *>(src);
        float v = 0;
        GGMLDequantizeOne(0, raw, 2, v);
        check(near(v, 3.5f), "F32 单元素 index=2 = 3.5");

        float out[4] = {};
        GGMLDequantize(0, raw, 4, out);
        const bool all =
            near(out[0], 1.0f) && near(out[1], -2.0f) && near(out[2], 3.5f) && near(out[3], 0.25f);
        check(all, "F32 整张量反量化一致");
    }

    std::cout << "[4] BF16 反量化（type=30 为本模型校准）" << std::endl;
    {
        const std::uint16_t bits[3] = {0x3F80, 0x4000, 0xBF80}; // 1, 2, -1
        const std::uint8_t *raw = reinterpret_cast<const std::uint8_t *>(bits);
        float out[3] = {};
        GGMLDequantize(30, raw, 3, out);
        check(near(out[0], 1.0f) && near(out[1], 2.0f) && near(out[2], -1.0f),
              "BF16 整张量 [1, 2, -1]");
    }

    std::cout << "[5] Q4_0 反量化（block=32, 2字节scale+16字节4bit）" << std::endl;
    {
        std::uint8_t block[18] = {};
        const std::uint16_t scale = 0x3C00; // fp16 = 1.0
        std::memcpy(block, &scale, 2);
        // 每个字节两个 nibble 都填 9 → 所有元素 q = 9-8 = 1
        std::memset(block + 2, 0x99, 16);
        float v = 0;
        GGMLDequantizeOne(2, block, 0, v);
        check(near(v, 1.0f), "Q4_0 元素0 (nibble=9) = 1.0");
        GGMLDequantizeOne(2, block, 1, v);
        check(near(v, 1.0f), "Q4_0 元素1 (nibble=9) = 1.0");
        // 元素2/3 在 byte3：元素2(低)=9→1，元素3(高)=F=15→7
        block[2 + 1] = 0xF9;
        GGMLDequantizeOne(2, block, 3, v);
        check(near(v, 7.0f), "Q4_0 元素3 (nibble=15) = 7.0");
    }

    std::cout << "[6] Q8_0 反量化（block=32, 2字节scale+32字节int8）" << std::endl;
    {
        std::uint8_t block[34] = {};
        const std::uint16_t scale = 0x3800; // fp16 = 0.5
        std::memcpy(block, &scale, 2);
        block[2 + 0] = 4;                             // q=4 → 2.0
        block[2 + 1] = static_cast<std::uint8_t>(-2); // q=-2 → -1.0
        float v = 0;
        GGMLDequantizeOne(8, block, 0, v);
        check(near(v, 2.0f), "Q8_0 元素0 (q=4, scale=0.5) = 2.0");
        GGMLDequantizeOne(8, block, 1, v);
        check(near(v, -1.0f), "Q8_0 元素1 (q=-2) = -1.0");
    }

    std::cout << "[7] 边界：未知/未实现类型" << std::endl;
    {
        std::uint8_t dummy[4] = {};
        float v = 0;
        check(!GGMLDequantizeOne(999, dummy, 0, v), "未知类型 999 返回 false");
        check(!GGMLDequantizeOne(10, dummy, 0, v), "Q2_K(未实现) 返回 false");
    }

    std::cout << (g_fail == 0 ? "\n✅ 全部通过" : "\n❌ 有失败") << std::endl;
    return g_fail == 0 ? 0 : 1;
}
