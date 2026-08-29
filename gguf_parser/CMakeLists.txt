cmake_minimum_required(VERSION 4.4.3)

project(gguf_parser VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(gguf_parser
    src/main.cpp
    )

add_library(MyLib STATIC src/gguf/GGUFLoader.cpp)

target_include_directories(MyLib PUBLIC 
    ${PROJECT_SOURCE_DIR}/include/gguf
)

target_link_libraries(gguf_parser PRIVATE 
    MyLib
)
