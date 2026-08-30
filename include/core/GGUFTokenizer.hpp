/*
 * GGUFTokenizer.hpp — 分词器（阶段 ③）
 *
 * 从 GGUF 的 tokenizer.ggml.* 元数据构建词汇表，并提供：
 *  - decode：token id → 文本（字节级 token 还原为原始字节）
 *  - encode：文本 → token ids（byte-level BPE，gpt2 风格）
 *
 * 说明：byte-level BPE 先把文本按 UTF-8 拆成字节，再把每个字节映射到一个
 *       unicode 字符（见 bytes_to_unicode），得到"字节编码串"；
 *       然后按 merge 规则把相邻字符贪心合并成 vocab 里的 token。
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "GGUFLoader.hpp"

struct GGUFTokenizer {
    // ---- 词汇表（来自 tokenizer.ggml.*）----
    std::vector<std::string> tokens; // id → token 文本（字节编码形式）
    std::vector<float> scores;       // id → 分数（可空）
    std::vector<std::int32_t> token_types; // id → 类型（1=正常, 3=控制, 6=字节级）
    std::vector<std::pair<std::string, std::string>> merges; // BPE merges（可空）
    std::string model_type;          // "gpt2" / "qwen2" ...
    std::int32_t bos_id = -1, eos_id = -1, pad_id = -1, unk_id = -1;

    // 从解析好的 GGUFModel 元数据构建词汇表
    bool build_from(const GGUFModel &model);

    // 由已填充的 tokens/merges 重建查找表（build_from 内部 / 单元测试用）
    void rebuild_index();

    // token id → 文本。
    // 特殊 token（控制/未知/用户定义）原样输出；普通与字节级 token 做字节还原。
    // 注意：字节级 BPE 中空格等是以 "Ġ"(U+0120) 形式存于词表（可能标记为 NORMAL 而非 BYTE），
    //       因此 decode 按"逐码点字节还原"处理，而不是只判断 token_type==6。
    std::string decode(std::int32_t id) const;

    // 文本 → token ids（byte-level BPE）
    std::vector<std::int32_t> encode(const std::string &text) const;

    std::size_t size() const { return tokens.size(); }

  private:
    std::unordered_map<std::string, std::int32_t> token_to_id_; // token 文本 → id
    std::unordered_map<std::string, int> ranks_;                // "a b" → merge 排名
};
