# ---- 单元测试（可被 -DBUILD_TESTING=OFF 关闭）----
if(BUILD_TESTING)
    add_executable(test_gguf_parser test_gguf_parser.cpp)
    target_link_libraries(test_gguf_parser PRIVATE gguf_model)
    add_test(NAME gguf_parser COMMAND test_gguf_parser)

    add_executable(test_verification test_verification.cpp)
    target_link_libraries(test_verification PRIVATE gguf_model)
    add_test(NAME type_system_verification COMMAND test_verification)

    add_executable(test_dequantize test_dequantize.cpp)
    target_link_libraries(test_dequantize PRIVATE gguf_model)
    add_test(NAME dequantize COMMAND test_dequantize)

    add_executable(test_tokenizer test_tokenizer.cpp)
    target_link_libraries(test_tokenizer PRIVATE gguf_model)
    add_test(NAME tokenizer COMMAND test_tokenizer)

    add_executable(test_model_weights test_model_weights.cpp)
    target_link_libraries(test_model_weights PRIVATE gguf_model)
    add_test(NAME model_weights COMMAND test_model_weights)

    add_executable(test_model_ops test_model_ops.cpp)
    target_link_libraries(test_model_ops PRIVATE gguf_model)
    add_test(NAME model_ops COMMAND test_model_ops)

    add_executable(test_transformer test_transformer.cpp)
    target_link_libraries(test_transformer PRIVATE gguf_model)
    add_test(NAME transformer COMMAND test_transformer)

    add_executable(test_ssm test_ssm.cpp)
    target_link_libraries(test_ssm PRIVATE gguf_model)
    add_test(NAME ssm COMMAND test_ssm)

    add_executable(test_forward test_forward.cpp)
    target_link_libraries(test_forward PRIVATE gguf_model)
    add_test(NAME forward COMMAND test_forward)

    add_executable(test_sampler test_sampler.cpp)
    target_link_libraries(test_sampler PRIVATE gguf_model)
    add_test(NAME sampler COMMAND test_sampler)

    add_executable(test_generate test_generate.cpp)
    target_link_libraries(test_generate PRIVATE gguf_model)
    add_test(NAME generate COMMAND test_generate)

    add_executable(test_chat test_chat.cpp)
    target_link_libraries(test_chat PRIVATE gguf_model)
    add_test(NAME chat COMMAND test_chat)

    add_executable(test_tokenizer_ggml test_tokenizer_ggml.cpp)
    target_link_libraries(test_tokenizer_ggml PRIVATE gguf_model)
    add_test(NAME tokenizer_ggml COMMAND test_tokenizer_ggml)
endif()
