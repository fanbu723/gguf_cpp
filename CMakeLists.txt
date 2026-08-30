cmake_minimum_required(VERSION 4.4.3)

project(gguf_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 开启编译警告，尽早暴露潜在问题
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# CTest：启用测试，并定义 BUILD_TESTING 选项（默认 ON，可用 -DBUILD_TESTING=OFF 关闭）
include(CTest)

# ---- 子目录：核心库 / 模型库 / 性能测试 / 单元测试 ----
add_subdirectory(src/core)   # gguf_core 静态库（GGUF 解析 + 类型 + 反量化 + 分词 + 权重）
add_subdirectory(src/model)  # gguf_model 静态库（Transformer 计算 + 采样 + 生成 + 对话）
add_subdirectory(src/bench)  # bench 性能测试
add_subdirectory(src/test)   # 单元测试（由 BUILD_TESTING 控制）

# ---- 可执行程序 ----
# 演示：解析 + 类型验证 + mmap + 反量化 + 分词 + 权重 + 算子 + 层前向
add_executable(main
    src/main.cpp
    )
target_link_libraries(main PRIVATE gguf_model)

# 交互式多轮对话（./build/chat [model_path] [max_new_tokens]）
add_executable(chat
    src/chat_main.cpp
    )
target_link_libraries(chat PRIVATE gguf_model)


