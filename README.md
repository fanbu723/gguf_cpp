# gguf_cpp

手写 GGUF 模型解析的学习项目，基于 **C++20 + CMake**，零第三方依赖。

目标是逐步实现一个能读取 GGUF 格式（文件头 → 元数据 KV → 张量信息表 → 张量数据）的解析器，
并最终扩展为可运行的极简推理引擎。

## 仓库结构

```
gguf_cpp/
├── gguf_parser/            # 主项目：GGUF 文件读取器
│   ├── include/gguf/       #   GGUFHeader 结构体 + GGUFLoader 接口
│   ├── src/                #   main 入口 + GGUFLoader 实现
│   ├── CMakeLists.txt      #   CMake 构建配置（可执行 + 静态库 MyLib）
│   ├── .clang-format       #   格式化规则
│   └── README.md           #   子项目说明
├── doc/                    # 设计文档与架构图
├── .clang-format           # 全仓格式化规则
├── .gitignore
└── .vscode/                # VS Code 配置（clang-format 保存自动格式化等）
```

## 构建与运行

环境要求：CMake ≥ 3.20、支持 C++20 的编译器（GCC ≥ 10 / Clang ≥ 12）。

```bash
# 构建
cd gguf_parser
cmake -S . -B build
cmake --build build

# 运行：读取 GGUF 模型文件头并打印
./build/gguf_parser
```

> 当前模型路径硬编码在 `gguf_parser/src/main.cpp` 的 `model_path` 中。

## 当前进度

- [x] 读取 GGUF 文件头（24 字节：magic / version / tensor_count / metadata_kv_count）
- [x] 校验魔数，判断是否为合法 GGUF 文件
- [ ] 解析元数据 KV（含 ARRAY 递归）
- [ ] 解析张量信息表（name / shape / type / offset / size）
- [ ] 读取张量数据并反量化
- [ ] 扩展为可运行的推理引擎

## 代码风格

统一使用 clang-format（规则见根目录 `.clang-format`），VS Code 已配置保存时自动格式化。
