# gguf_parser

手写 GGUF 文件读取器（C++20 + CMake），用于理解 GGUF 模型的二进制格式。

## 功能

- 读取 GGUF 文件头（magic / version / tensor_count / metadata_kv_count，共 24 字节）
- 校验魔数，判断文件是否为合法的 GGUF 模型
- 解析结果通过 `GGUFLoader` 静态方法暴露，便于后续扩展元数据 KV、张量信息表

## GGUF 文件头布局（全部小端字节序）

| 字段               | 大小 | 说明                |
| ------------------ | ---- | ------------------- |
| magic              | 4B   | `"GGUF"` (0x46554747) |
| version            | 4B   | 版本号（当前为 3）  |
| tensor_count       | 8B   | 张量数量            |
| metadata_kv_count  | 8B   | 元数据键值对数量    |

> 文件头之后依次是元数据 KV 区、张量信息表、张量数据（每张量 32 字节对齐）。

## 目录结构

```
gguf_parser/
├── CMakeLists.txt              # 构建配置（可执行 + 静态库 MyLib）
├── .clang-format               # clang-format 格式化规则
├── include/gguf/
│   └── GGUFLoader.hpp          # GGUFHeader 结构体 + GGUFLoader 接口声明
├── src/
│   ├── main.cpp                # 入口：加载模型并打印文件头
│   └── gguf/
│       └── GGUFLoader.cpp      # 解析实现（读取头部、校验魔数）
└── doc/                        # 设计文档与图片
```

## 构建与运行

环境要求：CMake ≥ 3.20、支持 C++20 的编译器（GCC ≥ 10 / Clang ≥ 12）。

```bash
# 方式一：使用项目根目录的构建脚本
bash build.sh

# 方式二：手动 CMake
cd gguf_parser
cmake -S . -B build
cmake --build build

# 运行（读取模型文件头并打印）
./build/gguf_parser
```

> 模型路径目前硬编码在 `src/main.cpp` 的 `model_path` 中，可改为命令行参数传入。

## 代码风格

项目使用 clang-format 统一格式，规则见根目录 `.clang-format`（4 空格缩进、大括号跟随语句、指针/引用靠变量名）。VS Code 已配置保存时自动格式化。

## 后续计划

- [ ] 解析元数据 KV（含 ARRAY 递归读取）
- [ ] 解析张量信息表（name / shape / type / offset / size）
- [ ] 读取张量数据并做反量化
- [ ] 模型路径改为命令行参数 / 配置文件
