#!/usr/bin/env bash
# chat.sh — 交互式多轮对话启动脚本
# 用法：./scripts/chat.sh [model_path] [max_new_tokens]
set -e
cd "$(dirname "$0")/.."

# 确保已构建
cmake -S . -B build >/dev/null
cmake --build build >/dev/null

exec ./build/chat "$@"
