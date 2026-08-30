/*
 * GGMLNorm.hpp — RMSNorm 归一化（阶段 ⑤）
 *
 * RMSNorm（Root Mean Square Layer Normalization）是 Qwen 等模型使用的
 * 归一化层：不做减均值，只按"均方根"缩放，再乘可学习权重 gamma。
 *
 *   y[i] = x[i] / sqrt(mean(x²) + eps) * gamma[i]
 *
 * 相比 LayerNorm 少了减均值，计算更省；eps 用于防止除零。
 */

#pragma once

/**
 * @brief RMSNorm 前向
 * @param x 输入向量（n 个元素）
 * @param gamma 可学习缩放权重（n 个元素，即 attn_norm.weight / output_norm.weight）
 * @param y 输出向量（n 个元素）
 * @param n 向量长度
 * @param eps 防除零小量（配置里的 layer_norm_rms_epsilon）
 */
void GGMLRmsNorm(const float *x, const float *gamma, float *y, int n, float eps);
