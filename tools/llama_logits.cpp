// 临时工具：用 llama.cpp C API 对给定 token ID 序列做前向，dump 最后一个 token 的 top-k logits
// 用法: llama_logits <model.gguf> <token_id1,id2,...> [top_k]
// 编译: g++ -std=c++11 -O2 -I ~/Projects/llama.cpp/include llama_logits.cpp \
//          -L ~/Projects/llama.cpp/build/bin -lllama -Wl,-rpath,... -o llama_logits
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "用法: llama_logits <model.gguf> <tokens...> [top_k]\n");
        return 1;
    }
    const char *model_path = argv[1];
    int top_k = 10;
    std::vector<llama_token> toks;
    for (int a = 2; a < argc; ++a) {
        if (strchr(argv[a], ',')) {
            // 逗号分隔
            char *p = argv[a], *tok = strtok(p, ",");
            while (tok) {
                toks.push_back((llama_token)atoi(tok));
                tok = strtok(nullptr, ",");
            }
        } else if (strcmp(argv[a], "-k") == 0 && a + 1 < argc) {
            top_k = atoi(argv[++a]);
        } else {
            toks.push_back((llama_token)atoi(argv[a]));
        }
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU
    llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "加载模型失败\n");
        return 1;
    }
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 4096;
    cparams.n_ubatch = 4096;
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "创建 context 失败\n");
        llama_model_free(model);
        return 1;
    }

    std::fprintf(stderr, "tokens(%zu):", toks.size());
    for (auto t : toks)
        std::fprintf(stderr, " %d", (int)t);
    std::fprintf(stderr, "\n");

    llama_batch batch = llama_batch_get_one(toks.data(), (int)toks.size());
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "decode 失败\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // 最后一个 token 的 logits
    const float *logits = llama_get_logits_ith(ctx, (int)toks.size() - 1);

    // top-k
    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; ++i)
        idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::printf("Top-%d logits (llama.cpp):\n", top_k);
    for (int i = 0; i < top_k; ++i) {
        int id = idx[i];
        std::string text = llama_vocab_get_text(vocab, (llama_token)id);
        std::printf("  [%d] %.4f  \"%s\"\n", id, logits[id], text.c_str());
    }
    // 固定 token 的 logits 数值
    int fixed[] = {0, 11, 13, 220, 332, 198, 9419, 248045, 248068};
    std::printf("fixed-logits:");
    for (int f : fixed)
        std::printf(" %d=%.4f", f, logits[f]);
    std::printf("\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
