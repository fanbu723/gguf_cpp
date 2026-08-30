/*
 * GGMLOps.hpp — 基础数学算子（阶段 ⑤ 第 1 步）
 *
 * 提供矩阵乘（gemv / gemm）与 softmax，是 Transformer 前向的基础。
 *
 * ⚠️ 关于 GGUF 权重的矩阵布局（重要）：
 *    GGUF 张量按 ggml 约定以"列主序"存储 —— 例如 attn_qkv.weight 的
 *    dims=[1024, 6144] 表示 [in_features, out_features]，内存中 in_features
 *    是最内层（连续）。因此该权重可看作"行主序的 [out_features, in_features]"
 *    矩阵（GGUF 的每一列 = 行主序的一行）。
 *    于是 y = W·x 直接用 GGMLGemmVec(weight, x, y, out_features, in_features)。
 */

#pragma once

/**
 * @brief 矩阵向量乘：out = W · x（行主序）
 * @param W 权重，行主序，共 rows×cols 个元素
 * @param x 输入向量，cols 个元素
 * @param out 输出向量，rows 个元素
 * @param rows W 的行数
 * @param cols W 的列数
 *
 * out[i] = Σ_j W[i][j] · x[j]
 */
void GGMLGemmVec(const float *W, const float *x, float *out, int rows, int cols);

/**
 * @brief 通用矩阵乘：C += A · B 或 C = A · B（行主序）
 * @param A 行主序，m×k
 * @param B 行主序，k×n
 * @param C 行主序，m×n（输出）
 * @param m A 的行数
 * @param n B 的列数
 * @param k A 的列数 / B 的行数
 * @param accumulate 为 true 时累加到 C（C 需已初始化），否则覆盖
 *
 * C[i][j] = Σ_t A[i][t] · B[t][j]
 */
void GGMLGemm(const float *A, const float *B, float *C, int m, int n, int k,
              bool accumulate = false);

/**
 * @brief 就地 softmax：x[i] = exp(x[i] - max) / Σexp（数值稳定）
 * @param x 输入输出数组
 * @param n 元素个数
 */
void GGMLSoftmax(float *x, int n);

/**
 * @brief 带掩码的 softmax：mask[i] 为 -inf 的位置输出 0（用于因果注意力）
 * @param x 输入输出数组（就地）
 * @param mask 掩码数组，n 个元素（通常为 0 或 -inf）
 * @param n 元素个数
 */
void GGMLSoftmaxMasked(float *x, const float *mask, int n);
