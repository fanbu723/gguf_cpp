#include "GGUFTokenizer.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace {

// ---------------------------------------------------------------------------
// gpt2 风格 byte ↔ unicode 字符 映射表
//  可打印字节（33-126、161-172、174-255）保持原码点；
//  其余字节（含空格 32、换行等）映射到 256+ 的码点（如 32 → 288 = "Ġ"）
// ---------------------------------------------------------------------------

const std::unordered_map<int, std::uint32_t> &byte_to_unicode() {
    static const auto table = [] {
        std::unordered_map<int, std::uint32_t> m;
        const auto is_kept = [](int b) {
            return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        };
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (is_kept(b))
                m[b] = static_cast<std::uint32_t>(b);
            else
                m[b] = static_cast<std::uint32_t>(256 + (n++));
        }
        return m;
    }();
    return table;
}

const std::unordered_map<std::uint32_t, int> &unicode_to_byte() {
    static const auto table = [] {
        std::unordered_map<std::uint32_t, int> m;
        for (const auto &[b, c] : byte_to_unicode())
            m[c] = b;
        return m;
    }();
    return table;
}

// 把 UTF-8 字符串按字节编码成 unicode 字符串（每个字节 → 一个 unicode 字符）
std::string bytes_to_unicode_str(const std::string &s) {
    std::string out;
    for (unsigned char c : s) {
        const std::uint32_t cp = byte_to_unicode().at(static_cast<int>(c));
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

// 把"字节级"字符串（每个 unicode 字符代表一个字节）还原成原始字节
std::string unicode_str_to_bytes(const std::string &s) {
    std::string out;
    for (std::size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        std::uint32_t cp = 0;
        std::size_t len = 1;
        if (c < 0x80) {
            cp = c;
        } else if ((c >> 5) == 0x6) {
            cp = ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
            len = 2;
        } else if ((c >> 4) == 0xE) {
            cp = ((c & 0xFu) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6) |
                 (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
            len = 3;
        } else {
            i += 1;
            continue;
        }
        const auto it = unicode_to_byte().find(cp);
        if (it != unicode_to_byte().end())
            out += static_cast<char>(it->second);
        i += len;
    }
    return out;
}

// 取一个 UTF-8 字符的子串
std::string utf8_char_at(const std::string &s, std::size_t i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    std::size_t len = 1;
    if (c >= 0x80) {
        if ((c >> 5) == 0x6)
            len = 2;
        else if ((c >> 4) == 0xE)
            len = 3;
        else if ((c >> 3) == 0x1F)
            len = 4;
    }
    return s.substr(i, len);
}

// 简单分词：byte-encoded 串 → 词列表（每个词 = 前导空格 + 非空格内容）
std::vector<std::string> split_words(const std::string &s) {
    static const std::string space = bytes_to_unicode_str(std::string(" ")); // "Ġ"
    std::vector<std::string> words;
    std::size_t i = 0;
    while (i < s.size()) {
        std::string cur;
        if (s.compare(i, space.size(), space) == 0) {
            while (i + space.size() <= s.size() && s.compare(i, space.size(), space) == 0) {
                cur += space;
                i += space.size();
            }
        }
        while (i < s.size() && s.compare(i, space.size(), space) != 0)
            cur += s[i++];
        if (!cur.empty())
            words.push_back(cur);
    }
    return words;
}

// 对单个词做 BPE 合并：初始为单字符序列，贪心合并 rank 最小的相邻对
std::vector<std::string> bpe_word(const std::string &word,
                                  const std::unordered_map<std::string, int> &ranks) {
    // 初始：把词拆成单字符序列
    std::vector<std::string> seq;
    for (std::size_t i = 0; i < word.size();) {
        const std::string ch = utf8_char_at(word, i);
        seq.push_back(ch);
        i += ch.size();
    }
    // 贪心合并：反复合并 rank 最小的相邻对，直到无可合并
    while (seq.size() > 1) {
        int best = 0x7FFFFFFF;
        std::size_t best_i = std::string::npos;
        for (std::size_t i = 0; i + 1 < seq.size(); ++i) {
            const auto it = ranks.find(seq[i] + " " + seq[i + 1]);
            if (it != ranks.end() && it->second < best) {
                best = it->second;
                best_i = i;
            }
        }
        if (best_i == std::string::npos)
            break;
        seq[best_i] += seq[best_i + 1];
        seq.erase(seq.begin() + static_cast<std::ptrdiff_t>(best_i + 1));
    }
    return seq;
}

// ---------------------------------------------------------------------------
// 元数据数组提取
// ---------------------------------------------------------------------------

// 从元数据 ARRAY 中提取指定类型的元素列表（模板：T = std::string / float / std::int32_t）
// 若 value 不是 ARRAY 返回 false；ARRAY 中类型不匹配的元素会被跳过
// @param v 元数据值（必须是 ARRAY）
// @param out 输出参数，接收提取出的元素列表
// @return 成功返回 true；v 不是 ARRAY 返回 false
//
// 说明：模板替代三个结构完全相同的重复函数（extract_string/float/int32_array），
//       只在元数据类型不同；本函数仅在本翻译单元内实例化，定义放 .cpp 即可。
template <typename T> bool extract_array(const MetadataValue &v, std::vector<T> &out) {
    const auto *arr = std::get_if<std::shared_ptr<ArrayValue>>(&v);
    if (!arr)
        return false;
    out.clear();
    out.reserve((*arr)->elements.size());
    for (const auto &e : (*arr)->elements)
        if (const auto *val = std::get_if<T>(&e))
            out.push_back(*val);
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// GGUFTokenizer 实现
// ---------------------------------------------------------------------------

bool GGUFTokenizer::build_from(const GGUFModel &model) {
    const auto find = [&](const std::string &key) -> const MetadataValue * {
        for (const auto &kv : model.metadata)
            if (kv.key == key)
                return &kv.value;
        return nullptr;
    };

    if (const auto *v = find("tokenizer.ggml.tokens")) {
        if (!extract_array<std::string>(*v, tokens))
            return false;
    }
    if (const auto *v = find("tokenizer.ggml.scores"))
        extract_array<float>(*v, scores);
    if (const auto *v = find("tokenizer.ggml.token_type"))
        extract_array<std::int32_t>(*v, token_types);
    if (const auto *v = find("tokenizer.ggml.merges")) {
        std::vector<std::string> raw;
        if (extract_array<std::string>(*v, raw)) {
            merges.clear();
            merges.reserve(raw.size());
            for (const auto &s : raw) {
                const auto pos = s.find(' ');
                if (pos != std::string::npos)
                    merges.emplace_back(s.substr(0, pos), s.substr(pos + 1));
            }
        }
    }
    if (const auto *v = find("tokenizer.ggml.model"))
        if (const auto *s = std::get_if<std::string>(v))
            model_type = *s;

    const auto get_id = [&](const std::string &key) -> std::int32_t {
        if (const auto *v = find(key)) {
            if (const auto *u = std::get_if<std::uint32_t>(v))
                return static_cast<std::int32_t>(*u);
            if (const auto *i = std::get_if<std::int32_t>(v))
                return *i;
        }
        return -1;
    };
    bos_id = get_id("tokenizer.ggml.bos_token_id");
    eos_id = get_id("tokenizer.ggml.eos_token_id");
    pad_id = get_id("tokenizer.ggml.pad_token_id");
    unk_id = get_id("tokenizer.ggml.unk_token_id");

    rebuild_index();
    return !tokens.empty();
}

void GGUFTokenizer::rebuild_index() {
    token_to_id_.clear();
    for (std::size_t i = 0; i < tokens.size(); ++i)
        token_to_id_[tokens[i]] = static_cast<std::int32_t>(i);

    ranks_.clear();
    for (std::size_t i = 0; i < merges.size(); ++i)
        ranks_[merges[i].first + " " + merges[i].second] = static_cast<int>(i);

    // 收集特殊 token（type 2=未知 / 3=控制 / 4=用户定义），按长度降序，
    // 使 encode 扫描时"先匹配到的最长"即为最优（<|im_start|> 优先于 <| 等子串）。
    special_tokens_.clear();
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const std::int32_t type = i < token_types.size() ? token_types[i] : 1;
        if (type == 2 || type == 3 || type == 4)
            special_tokens_.push_back(tokens[i]);
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const std::string &a, const std::string &b) { return a.size() > b.size(); });
}

std::string GGUFTokenizer::decode(std::int32_t id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= tokens.size())
        return {};
    const std::string &tok = tokens[static_cast<std::size_t>(id)];
    const std::int32_t type =
        id < static_cast<std::int32_t>(token_types.size()) ? token_types[id] : 1;

    // gpt2 风格字节级 BPE 的逆变换（参考 llama.cpp）：
    //   token_type：0 未定义 / 1 普通 / 2 未知 / 3 控制 / 4 用户定义 / 5 未用 / 6 字节
    switch (type) {
    case 2:         // 未知
    case 3:         // 控制（如 <|endoftext|>）
    case 4:         // 用户定义
        return tok; // 特殊 token 原样输出
    case 1:         // 普通
    case 6:         // 字节
        // 字节级 BPE 中词表里每个字符都是字节编码（如空格→"Ġ"），
        // 因此整串逐码点还原成原始字节（注意不能只看 type==6）
        return unicode_str_to_bytes(tok);
    default: // 未用等 → 不输出
        return {};
    }
}

// 对一段普通文本做字节级 BPE（编码 → 切词 → 贪心 merge）
void GGUFTokenizer::append_encode_chunk(std::vector<std::int32_t> &ids,
                                        const std::string &chunk) const {
    const std::string encoded = bytes_to_unicode_str(chunk); // 字节编码
    const auto words = split_words(encoded);                 // 分词
    for (const auto &w : words) {                            // 每词 BPE
        const auto pieces = bpe_word(w, ranks_);
        for (const auto &p : pieces) {
            const auto it = token_to_id_.find(p);
            if (it != token_to_id_.end())
                ids.push_back(it->second);
        }
    }
}

std::vector<std::int32_t> GGUFTokenizer::encode(const std::string &text) const {
    std::vector<std::int32_t> ids;

    // 1. 特殊 token（type 2/3/4，如 <|im_start|>）在字节编码/BPE 之前整体匹配，
    //    直接映射到它的单一 token id——否则会被字节级 BPE 拆成多个子 token。
    std::string chunk; // 累积的普通文本片段
    std::size_t i = 0;
    while (i < text.size()) {
        const std::string *best = nullptr;      // 匹配到的最长特殊 token
        for (const auto &s : special_tokens_) { // 已按长度降序
            if (text.compare(i, s.size(), s) == 0) {
                best = &s;
                break; // 第一个匹配即最长
            }
        }
        if (best) {
            if (!chunk.empty()) { // 先编码普通片段
                append_encode_chunk(ids, chunk);
                chunk.clear();
            }
            const auto it = token_to_id_.find(*best);
            if (it != token_to_id_.end())
                ids.push_back(it->second);
            i += best->size();
        } else {
            chunk.push_back(text[i]);
            ++i;
        }
    }
    if (!chunk.empty())
        append_encode_chunk(ids, chunk);
    return ids;
}
