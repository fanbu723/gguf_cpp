#include "GGMLChat.hpp"

#include <string>
#include <vector>

bool GGMLChat::init(const GGUFModelWeights &w, GGUFTokenizer &tok, GGMLSampleMode mode,
                    std::uint64_t seed, int max_len) {
    w_ = &w;
    tok_ = &tok;
    sampler_.mode = mode;
    sampler_.reseed(seed);
    max_len_ = max_len;
    history_text_.clear();
    return true;
}

std::string GGMLChat::chat(const std::string &user, int max_new_tokens) {
    if (!w_ || !tok_)
        return {};

    // 1. 拼接本轮 prompt（历史 + 新的 user 消息 + assistant 前缀）
    history_text_ += "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
    const auto prompt = tok_->encode(history_text_);
    if (prompt.empty())
        return {};

    // 2. 自回归生成回复（每轮重新预填充完整历史）
    GGMLModelState st;
    st.init(*w_, max_len_);
    const auto gen = GGMLGenerate(*w_, st, sampler_, prompt, max_new_tokens, tok_->eos_id);

    // 3. decode 回复并追加到历史
    std::string reply;
    for (int t : gen)
        reply += tok_->decode(t);
    history_text_ += reply + "<|im_end|>\n";
    return reply;
}
