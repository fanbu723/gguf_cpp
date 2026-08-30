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
    )

add_library(MyLib STATIC 
    src/core/GGUFLoader.cpp 
    src/core/GGUFMmap.cpp 
    src/core/GGMLType.cpp
    src/core/GGMLDequantize.cpp
    src/core/GGUFTokenizer.cpp
    src/core/GGUFModelWeights.cpp
    src/model/GGMLOps.cpp
    src/model/GGMLNorm.cpp
    src/model/GGMLFFN.cpp
    src/model/GGMLRope.cpp
    src/model/GGMLAttention.cpp
    src/model/GGMLTransformer.cpp
    src/model/GGMLSSM.cpp
    src/model/GGMLForward.cpp
    src/model/GGMLSampler.cpp
    src/model/GGMLGenerate.cpp
    src/model/GGMLChat.cpp)

target_include_directories(MyLib PUBLIC 
    ${PROJECT_SOURCE_DIR}/include/core
    ${PROJECT_SOURCE_DIR}/include/model
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

    add_executable(test_model_ops
        src/test/test_model_ops.cpp
        )
    target_link_libraries(test_model_ops PRIVATE MyLib)
    add_test(NAME model_ops COMMAND test_model_ops)

    add_executable(test_transformer
        src/test/test_transformer.cpp
        )
    target_link_libraries(test_transformer PRIVATE MyLib)
    add_test(NAME transformer COMMAND test_transformer)

    add_executable(test_ssm
        src/test/test_ssm.cpp
        )
    target_link_libraries(test_ssm PRIVATE MyLib)
    add_test(NAME ssm COMMAND test_ssm)

    add_executable(test_forward
        src/test/test_forward.cpp
        )
    target_link_libraries(test_forward PRIVATE MyLib)
    add_test(NAME forward COMMAND test_forward)

    add_executable(test_sampler
        src/test/test_sampler.cpp
        )
    target_link_libraries(test_sampler PRIVATE MyLib)
    add_test(NAME sampler COMMAND test_sampler)

    add_executable(test_generate
        src/test/test_generate.cpp
        )
    target_link_libraries(test_generate PRIVATE MyLib)
    add_test(NAME generate COMMAND test_generate)

    add_executable(test_chat
        src/test/test_chat.cpp
        )
    target_link_libraries(test_chat PRIVATE MyLib)
    add_test(NAME chat COMMAND test_chat)

    add_executable(test_tokenizer_ggml
        src/test/test_tokenizer_ggml.cpp
        )
    target_link_libraries(test_tokenizer_ggml PRIVATE MyLib)
    add_test(NAME tokenizer_ggml COMMAND test_tokenizer_ggml)
endif()

