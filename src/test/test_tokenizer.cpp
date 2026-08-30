/*
 * test_tokenizer.cpp — 分词器单元测试（不依赖真实模型）
 *
 * 手建一个小词汇表 + BPE merges，验证：
 *  - encode：BPE 合并是否正确
 *  - decode：普通 token / 字节级 token（type==6）还原
 *  - 越界 id 安全返回空串
 */

#include <iostream>
#include <vector>

#include "GGUFTokenizer.hpp"

static int g_fail = 0;

static void check(bool ok, const char *msg) {
    if (ok)
        std::cout << "  ✅ " << msg << std::endl;
    else {
        std::cout << "  ❌ " << msg << std::endl;
        ++g_fail;
    }
}

int main() {
    std::cout << "=== 分词器单元测试 ===" << std::endl;

    std::cout << "[1] 手建 vocab + BPE 合并" << std::endl;
    {
        GGUFTokenizer t;
        t.tokens = {"a", "b", "ab", "aab"};
        t.token_types = {1, 1, 1, 1};
        t.merges = {{"a", "b"}}; // "a b" → rank 0
        t.rebuild_index();

        // "ab" → 单字符 [a, b] → 合并 "a b" → [ab]
        const auto ids = t.encode("ab");
        check(ids.size() == 1 && ids[0] == 2, "encode('ab') → [2] (token 'ab')");

        // "aab" → [a, a, b] → 合并末尾 "a b" → [a, ab]
        const auto ids2 = t.encode("aab");
        check(ids2.size() == 2 && ids2[0] == 0 && ids2[1] == 2, "encode('aab') → [0,2]");

        check(t.decode(0) == "a", "decode(0) = 'a'");
        check(t.decode(2) == "ab", "decode(2) = 'ab'");
    }

    std::cout << "[2] 字节级 token 解码（含 type==1 的字节编码 token）" << std::endl;
    {
        GGUFTokenizer t;
        // "Ġ" = 空格字节 32 的字节编码（U+0120）；真实模型里它可能标记为 NORMAL(1) 而非 BYTE(6)
        t.tokens = {"Ġ", "Ġ", "Hello", "<|endoftext|>"};
        t.token_types = {6, 1, 1, 3}; // BYTE / NORMAL / NORMAL / CONTROL
        t.rebuild_index();
        check(t.decode(0) == " ", "decode(type6 'Ġ') = 空格（字节 32 还原）");
        check(t.decode(1) == " ", "decode(type1 'Ġ') = 空格（模型实际情况）");
        check(t.decode(2) == "Hello", "decode(type1 'Hello') = 'Hello'（ASCII 字节还原不变）");
        check(t.decode(3) == "<|endoftext|>", "decode(type3 控制) 原样输出");
    }

    std::cout << "[3] 越界 id 安全" << std::endl;
    {
        GGUFTokenizer t;
        t.tokens = {"x"};
        t.token_types = {1};
        t.rebuild_index();
        check(t.decode(5) == "", "decode(越界 id) = 空串");
        check(t.decode(-1) == "", "decode(负 id) = 空串");
    }

    std::cout << (g_fail == 0 ? "\n✅ 全部通过" : "\n❌ 有失败") << std::endl;
    return g_fail == 0 ? 0 : 1;
}
