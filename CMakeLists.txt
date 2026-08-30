cmake_minimum_required(VERSION 4.4.3)

project(gguf_parser VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 开启编译警告，尽早暴露潜在问题
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# CTest：启用测试，并定义 BUILD_TESTING 选项（默认 ON，可用 -DBUILD_TESTING=OFF 关闭）
include(CTest)

add_executable(main
    src/main.cpp
    include/core/GGUFModelWeights.hpp
    )

add_library(MyLib STATIC 
    src/core/GGUFLoader.cpp 
    src/core/GGUFMmap.cpp 
    src/core/GGMLType.cpp
    src/core/GGMLDequantize.cpp
    src/core/GGUFTokenizer.cpp
    src/core/GGUFModelWeights.cpp)

target_include_directories(MyLib PUBLIC 
    ${PROJECT_SOURCE_DIR}/include/core
)

target_link_libraries(main PRIVATE 
    MyLib
)

# ---- 测试（可被 -DBUILD_TESTING=OFF 关闭）----
if(BUILD_TESTING)
    add_executable(test_gguf_parser
        src/test/test_gguf_parser.cpp
        )
    target_link_libraries(test_gguf_parser PRIVATE MyLib)
    add_test(NAME gguf_parser COMMAND test_gguf_parser)

    add_executable(test_verification
        src/test/test_verification.cpp
        )
    target_link_libraries(test_verification PRIVATE MyLib)
    add_test(NAME type_system_verification COMMAND test_verification)

    add_executable(test_dequantize
        src/test/test_dequantize.cpp
        )
    target_link_libraries(test_dequantize PRIVATE MyLib)
    add_test(NAME dequantize COMMAND test_dequantize)

    add_executable(test_tokenizer
        src/test/test_tokenizer.cpp
        )
    target_link_libraries(test_tokenizer PRIVATE MyLib)
    add_test(NAME tokenizer COMMAND test_tokenizer)

    add_executable(test_model_weights
        src/test/test_model_weights.cpp
        )
    target_link_libraries(test_model_weights PRIVATE MyLib)
    add_test(NAME model_weights COMMAND test_model_weights)
endif()

