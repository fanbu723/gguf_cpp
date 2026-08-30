#!/usr/bin/env bash
# chat.sh — 交互式多轮对话启动脚本
# 用法：./scripts/chat.sh [model_path] [max_new_tokens]
# 说明：用 Release 预设构建并运行（Debug 无优化，前向慢约 3~4 倍）
set -e
cd "$(dirname "$0")/.."

# 确保已构建（Release 预设）
cmake --preset release >/dev/null
cmake --build build-release >/dev/null

exec ./build-release/chat "$@"
