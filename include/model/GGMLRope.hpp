/*
 * GGMLRope.hpp — RoPE 旋转位置编码（阶段 ⑤）
 *
 * Qwen3.5（qwen35）使用 IMROPE（interleaved M-RoPE），sections=[11,11,10,0]，
 * n_rot=64。多模态时 t/h/w/e 四个位置不同；但本项目是纯文本生成，
 * 四个位置都等于 token 位置 —— 此时 IMROPE 退化为"标准 NEOX RoPE 旋转前 n_rot 维，
 * 其余 head_dim - n_rot 维不旋转"。故这里实现 NEOX 风格即可。
 *
 * NEOX 旋转（每个 head 内部）：
 *   将前 n_rot 维分成两半，配对 (i, i + n_rot/2) 旋转：
 *     y[i]        = x[i]·cos(θ) - x[i+half]·sin(θ)
 *     y[i+half]   = x[i]·sin(θ) + x[i+half]·cos(θ)
 *   其中 θ = pos · freq_base^(-2i/n_rot)，i = 0..half-1。
 */

#pragma once

/**
 * @brief 对一个 head 应用 NEOX 风格 RoPE
 * @param x 输入 head 向量（head_dim 个元素）
 * @param y 输出 head 向量（head_dim 个元素）
 * @param head_dim 每个 head 的维度（如 q=512、k=256）
 * @param n_rot 旋转维数（须为偶数，如 64）
 * @param pos token 位置（从 0 开始）
 * @param freq_base 基频（配置里的 rope.freq_base，如 1e7）
 *
 * 前 n_rot 维按 NEOX 旋转，其余 head_dim - n_rot 维原样复制。
 */
void GGMLRopeNeox(const float *x, float *y, int head_dim, int n_rot, int pos, float freq_base);
