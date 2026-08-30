cmake_minimum_required(VERSION 4.4.3)

project(gguf_parser VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 开启编译警告，尽早暴露潜在问题
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_executable(main
    src/main.cpp
    )

add_executable(test_gguf_parser
    src/test/test_gguf_parser.cpp
    )

add_library(MyLib STATIC src/core/GGUFLoader.cpp src/core/GGUFMmap.cpp src/core/GGMLType.cpp)

target_include_directories(MyLib PUBLIC 
    ${PROJECT_SOURCE_DIR}/include/core
)

target_link_libraries(main PRIVATE 
    MyLib
)

target_link_libraries(test_gguf_parser PRIVATE 
    MyLib
)
