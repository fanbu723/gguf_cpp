#include <cstring>
#include <iostream>
#include <ostream>
#include <string>

#include "GGUFLoader.hpp"
#include "GGUFTokenizer.hpp"

std::string model_path = "/home/dongfan/llm/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf";
int main() {
    GGUFModel model;

    if (!GGUFLoader::load(model_path, model)) {
        std::cerr << "❌ 加载 GGUF 文件失败: " << model_path << std::endl;
        return 1;
    }
    GGUFTokenizer tok;
    if (tok.build_from(model)) {
        std::cout << "  词汇表大小: " << tok.size()
                  << "  model=" << (tok.model_type.empty() ? "?" : tok.model_type)
                  << "  bos=" << tok.bos_id << "  eos=" << tok.eos_id << std::endl;
        std::cout << "  前 5 个 token: ";
        for (std::int32_t i = 0; i < 5; ++i)
            std::cout << "[" << i << "]'" << tok.decode(i) << "' ";
        std::cout << std::endl;

        const std::string text = "Hello, world!";
        const auto ids = tok.encode(text);
        std::cout << "  encode('" << text << "') → ";
        for (std::size_t i = 0; i < ids.size(); ++i)
            std::cout << ids[i] << (i + 1 < ids.size() ? "," : "");
        std::cout << std::endl;

        std::string round;
        for (std::int32_t id : ids)
            round += tok.decode(id);
        std::cout << "  decode(encode) = '" << round << "'"
                  << (round == text ? "  ✅ 往返一致" : "  （字节级，可能有空白差异）")
                  << std::endl;

        // 中文往返：字节级 BPE 对任意 UTF-8 都无损
        const std::string text2 = "你好，世界！";
        const auto ids2 = tok.encode(text2);
        std::string round2;
        for (std::int32_t id : ids2)
            round2 += tok.decode(id);
        std::cout << "  encode('" << text2 << "') → " << ids2.size() << " tokens"
                  << "  decode(encode) = '" << round2 << "'"
                  << (round2 == text2 ? "  ✅ 往返一致" : "  ❌ 不一致") << std::endl;
    } else {
        std::cerr << "  ❌ tokenizer 构建失败" << std::endl;
    }
    return 0;
}