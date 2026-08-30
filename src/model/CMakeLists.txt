# gguf_model —— Transformer 计算 / 采样 / 生成 / 对话（阶段 ⑤⑥）
# 依赖 gguf_core（解析 + 权重）
add_library(gguf_model STATIC
    GGMLOps.cpp
    GGMLNorm.cpp
    GGMLFFN.cpp
    GGMLRope.cpp
    GGMLAttention.cpp
    GGMLTransformer.cpp
    GGMLSSM.cpp
    GGMLForward.cpp
    GGMLSampler.cpp
    GGMLGenerate.cpp
    GGMLChat.cpp)

target_include_directories(gguf_model PUBLIC ${PROJECT_SOURCE_DIR}/include/model)
target_link_libraries(gguf_model PUBLIC gguf_core)
