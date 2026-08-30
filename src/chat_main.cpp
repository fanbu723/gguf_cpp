/*
 * chat_main.cpp — 交互式多轮对话程序（阶段⑥）
 *
 * 加载模型后进入命令行对话循环：
 *   User>     输入内容
 *   Assistant> 模型回复
 *   /clear    清空对话历史
 *   Ctrl+D / /quit  退出
 *
 * 用法：./build/chat [model_path] [max_new_tokens]
 *
 * 默认模型为本项目自转的干净 GGUF（Qwen3.5-0.8B-clean-BF16.gguf）。
 * 启动时探测 logits 是否有限（若模型含 NaN 会提示换干净 GGUF，避免采样崩溃）。
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "GGMLChat.hpp"
#include "GGUFLoader.hpp"
#include "GGUFModelWeights.hpp"
#include "GGUFTokenizer.hpp"

namespace {
std::string default_model = "/home/dongfan/llm/Qwen3.5-0.8B-clean-BF16.gguf";

// 探测模型 logits 是否可用（全 NaN 则不可对话）
bool model_has_nan(const GGUFModelWeights &w) {
    GGMLModelState st;
    st.init(w, 2048);
    std::vector<float> probe;
    GGMLForward(w, st, 0, 0, probe);
    for (float v : probe)
        if (!std::isfinite(v))
            return true;
    return false;
}
} // namespace

int main(int argc, char **argv) {
    std::string model_path = argc > 1 ? argv[1] : default_model;
    int max_new_tokens = argc > 2 ? std::atoi(argv[2]) : 128;
    if (max_new_tokens <= 0)
        max_new_tokens = 128;

    std::cout << "=== GGUF Chat ===" << std::endl;
    std::cout << "  模型: " << model_path << std::endl;

    GGUFModel model;
    if (!GGUFLoader::load(model_path, model)) {
        std::cerr << "❌ 加载 GGUF 失败: " << model_path << std::endl;
        return 1;
    }
    GGUFLoader::map_data(model_path, model);

    GGUFTokenizer tok;
    if (!tok.build_from(model)) {
        std::cerr << "❌ tokenizer 构建失败" << std::endl;
        return 1;
    }
    GGUFModelWeights weights;
    if (!weights.build(model)) {
        std::cerr << "❌ 权重索引构建失败" << std::endl;
        return 1;
    }

    // 模型可用性探测
    if (model_has_nan(weights)) {
        std::cerr << "⚠️ 模型 logits 含 NaN（FFN 权重含 NaN），无法对话。" << std::endl;
        std::cerr << "   请换干净的 Qwen3.5 GGUF 后再运行。" << std::endl;
        return 1;
    }

    GGMLChat chat;
    chat.init(weights, tok, GGMLSampleMode::TOP_K_P, /*seed=*/42, /*max_len=*/2048);

    std::cout << "  输入内容开始对话；/clear 清空；Ctrl+D 或 /quit 退出" << std::endl << std::endl;

    std::string line;
    while (true) {
        std::cout << "User> " << std::flush;
        if (!std::getline(std::cin, line))
            break; // Ctrl+D
        if (line == "/quit" || line == "/exit")
            break;
        if (line == "/clear") {
            chat.clear();
            std::cout << "  (对话已清空)" << std::endl;
            continue;
        }
        if (line.empty())
            continue;

        std::cout << "Assistant> " << std::flush;
        const std::string reply = chat.chat(line, max_new_tokens);
        std::cout << reply << std::endl << std::endl;
    }
    std::cout << "再见！" << std::endl;
    return 0;
}
