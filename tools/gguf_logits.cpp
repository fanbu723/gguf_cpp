// 临时工具：项目前向对显式 token ID 序列 dump top-k logits（与 llama_logits 对照）
// 用法: gguf_logits <model.gguf> <id1,id2,...> [top_k]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "GGMLForward.hpp"
#include "GGUFLoader.hpp"
#include "GGUFModelWeights.hpp"

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "用法: gguf_logits <model.gguf> <id1,id2,...> [top_k]\n");
        return 1;
    }
    const std::string path = argv[1];
    int top_k = 10;
    std::vector<int> ids;
    for (int a = 2; a < argc; ++a) {
        if (strcmp(argv[a], "-k") == 0 && a + 1 < argc) {
            top_k = atoi(argv[++a]);
            continue;
        }
        // 逗号分隔或单个
        std::string s = argv[a];
        size_t p = 0;
        while (p < s.size()) {
            size_t q = s.find(',', p);
            ids.push_back(atoi(s.substr(p, q - p).c_str()));
            if (q == std::string::npos)
                break;
            p = q + 1;
        }
    }

    GGUFModel model;
    if (!GGUFLoader::load(path, model)) {
        std::fprintf(stderr, "加载失败\n");
        return 1;
    }
    GGUFLoader::map_data(path, model);
    GGUFModelWeights w;
    w.build(model);
    const auto *embd = w.token_embd();
    const int vocab = (int)embd->dims[1];

    std::fprintf(stderr, "tokens(%zu):", ids.size());
    for (int t : ids)
        std::fprintf(stderr, " %d", t);
    std::fprintf(stderr, "\n");

    GGMLModelState st;
    st.init(w, 4096);
    std::vector<float> logits;
    for (int pos = 0; pos < (int)ids.size(); ++pos) {
        logits.clear();
        GGMLForward(w, st, ids[pos], pos, logits);
    }

    std::vector<int> idx(vocab);
    for (int i = 0; i < vocab; ++i)
        idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::printf("Top-%d logits (gguf_cpp):\n", top_k);
    for (int i = 0; i < top_k; ++i) {
        int id = idx[i];
        std::printf("  [%d] %.4f\n", id, logits[id]);
    }
    int fixed[] = {0, 11, 13, 220, 332, 198, 9419, 248045, 248068};
    std::printf("fixed-logits:");
    for (int f : fixed)
        std::printf(" %d=%.4f", f, logits[f]);
    std::printf("\n");
    return 0;
}
