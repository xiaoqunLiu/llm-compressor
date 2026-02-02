# Quantization Methods Summary

This document summarizes the characteristics, required key parameters, and generated model features of various quantization methods supported by the `unified_quantization.py` script.

## 1. GPTQ (Generalized Post-Training Quantization)

*   **特点**: 
    *   一种经典的逐层量化方法。
    *   利用 Hessian 矩阵（二阶导数信息）来调整权重，以补偿量化带来的误差。
    *   是目前最常用的 W4A16 量化方案之一，推理速度快。
*   **脚本中的参数**:
    *   `scheme="W4A16"`: 权重 4-bit，激活值 16-bit。
    *   `targets="Linear"`: 量化所有线性层。
    *   `ignore=["lm_head"]`: 跳过最后的语言模型头，防止精度大幅下降。
*   **生成模型特点**:
    *   **格式**: W4A16 (权重 int4, 激活 fp16/bf16)。
    *   **优势**: 显存占用大幅减少（约 1/3 - 1/4），推理速度在支持 GPTQ kernel 的框架（如 vLLM, TGI）上很快。
    *   **劣势**: 在极端低比特下可能不如 AutoRound 精确。

## 2. AWQ (Activation-aware Weight Quantization)

*   **特点**: 
    *   基于“激活感知”的量化方法。认为并非所有权重都同等重要，通过观察 Activation 的幅度来保护那 1% 的显著权重（Salient Weights）。
    *   不依赖梯度更新，保留了权重的原始分布特性。
*   **脚本中的参数**:
    *   `scheme="W4A16_ASYM"`: 权重 4-bit 非对称量化。
    *   `duo_scaling="both"`: 使用双重缩放策略以提升精度。
    *   `targets=["Linear"]`: 量化线性层。
*   **生成模型特点**:
    *   **格式**: W4A16。
    *   **优势**: 通常比 GPTQ 具有更好的泛化能力和 PPL (困惑度) 表现，特别是在指令微调模型上。vLLM 对 AWQ 的支持非常好。
    *   **劣势**: 量化过程需要搜索缩放因子，比简单 PTQ 稍慢。

## 3. SmoothQuant

*   **特点**: 
    *   旨在解决 W8A8 量化中的 Activation 离群值（Outliers）问题。
    *   通过数学变换，将 Activation 的量化难度“平滑”迁移到权重上，使得两者都更容易被量化为 INT8。
*   **脚本中的参数**:
    *   `smoothing_strength=0.8`: 控制平滑迁移的强度。
    *   `scheme="W8A8"`: 最终执行 8-bit 权重和 8-bit 激活值的量化。
*   **生成模型特点**:
    *   **格式**: W8A8 (全 INT8 推理)。
    *   **优势**: 真正的全整型矩阵乘法，可以在支持 INT8 硬件（如各种 GPU Tensor Cores, CPU）上获得显著的计算加速，不仅仅是显存节省。
    *   **劣势**: 精度保持难度比 W4A16 大，特别是对于较小的模型。

## 4. AutoRound

*   **特点**: 
    *   基于学习的量化方法。
    *   通过梯度下降（Gradient Descent）来优化权重的舍入（Rounding）值，以最小化层输出的误差。
    *   通常被认为是目前 W4A16 甚至 W2A16 下 SOTA (State-of-the-Art) 的方法。
*   **脚本中的参数**:
    *   `iters=200`: 优化的迭代次数。
    *   `scheme="W4A16"`: 权重 4-bit。
*   **生成模型特点**:
    *   **格式**: W4A16 (通常兼容 GPTQ 格式)。
    *   **优势**: 精度通常最高，几乎无损压缩。
    *   **劣势**: 量化过程非常慢（因为需要迭代优化），对显存和计算资源要求较高。

## 5. Simple PTQ (Basic GPTQ without Dampening)

*   **特点**: 
    *   脚本中实现为基础的 W8A8 量化，没有应用高级的平滑或复杂的校准策略。
    *   仅做简单的统计和舍入。
*   **脚本中的参数**:
    *   `scheme="W8A8"`
    *   `dampening_frac=0.0`
*   **生成模型特点**:
    *   **格式**: W8A8。
    *   **优势**: 转换速度极快。
    *   **劣势**: 精度极有可能大幅下降，通常仅作为基准测试或调试使用，不建议用于生产环境。

---

## 总结表

| 方法 | 精度 (Bit) | 核心思想 | 校准时间 | 推荐场景 |
| :--- | :--- | :--- | :--- | :--- |
| **GPTQ** | W4A16 | Hessian 矩阵补偿 | 快 | 通用，显存受限场景 |
| **AWQ** | W4A16 | 激活感知保护显著权重 | 中 | 高性能推理 (vLLM) |
| **AutoRound** | W4A16 | 梯度优化舍入值 | 慢 | 追求极致精度的 W4 量化 |
| **SmoothQuant** | W8A8 | 平滑迁移激活难度 | 中 | 需要计算加速 (INT8 Compute) |
| **Simple PTQ** | W8A8 | 简单统计映射 | 极快 | 测试流程，非生产 |
