# gguf_core —— GGUF 文件解析 / 类型系统 / 反量化 / 分词器 / 权重加载（阶段 ①~④）
# 零第三方依赖，只依赖 include/core 头文件
add_library(gguf_core STATIC
    GGUFLoader.cpp
    GGUFMmap.cpp
    GGMLType.cpp
    GGMLDequantize.cpp
    GGUFTokenizer.cpp
    GGUFModelWeights.cpp)

target_include_directories(gguf_core PUBLIC ${PROJECT_SOURCE_DIR}/include/core)
