# 性能测试（./build/bench [forward_runs]）
add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE gguf_model)
