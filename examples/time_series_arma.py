#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARMA 时间序列模型示例
基于仓库中 "AR_MA_ARMA等10类时间序列模型.md" 的实现
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class ARMAModel:
    """ARMA(p, q) 时间序列模型"""

    def __init__(self, ar_params: np.ndarray, ma_params: np.ndarray):
        """
        初始化 ARMA 模型

        Args:
            ar_params: AR 参数 [φ1, φ2, ..., φp]
            ma_params: MA 参数 [θ1, θ2, ..., θq]
        """
        self.ar_params = np.array(ar_params)
        self.ma_params = np.array(ma_params)
        self.p = len(ar_params)
        self.q = len(ma_params)

    def simulate(self, n_samples: int, sigma: float = 1.0, seed: int = 42) -> np.ndarray:
        """
        模拟 ARMA 过程

        Args:
            n_samples: 样本数量
            sigma: 白噪声标准差
            seed: 随机种子

        Returns:
            生成的时间序列
        """
        np.random.seed(seed)

        # 白噪声
        epsilon = np.random.normal(0, sigma, n_samples + max(self.p, self.q))

        # 初始化时间序列
        y = np.zeros(n_samples + max(self.p, self.q))

        # 生成 ARMA 过程
        for t in range(max(self.p, self.q), len(y)):
            # AR 部分
            ar_term = 0
            for i in range(self.p):
                ar_term += self.ar_params[i] * y[t - i - 1]

            # MA 部分
            ma_term = 0
            for j in range(self.q):
                ma_term += self.ma_params[j] * epsilon[t - j - 1]

            y[t] = ar_term + ma_term + epsilon[t]

        return y[max(self.p, self.q):]

    def predict(self, history: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        预测未来值

        Args:
            history: 历史时间序列
            steps: 预测步数

        Returns:
            预测值
        """
        predictions = []
        y_extended = np.concatenate([history, np.zeros(steps)])

        for t in range(len(history), len(y_extended)):
            ar_term = 0
            for i in range(min(self.p, t)):
                ar_term += self.ar_params[i] * y_extended[t - i - 1]

            y_extended[t] = ar_term
            predictions.append(ar_term)

        return np.array(predictions)


def example_arma_11():
    """ARMA(1,1) 模型示例"""
    print("=" * 60)
    print("ARMA(1,1) 时间序列模型示例")
    print("=" * 60)

    # 创建 ARMA(1,1) 模型
    # y_t = 0.7 * y_{t-1} + ε_t + 0.4 * ε_{t-1}
    ar_params = [0.7]
    ma_params = [0.4]

    model = ARMAModel(ar_params, ma_params)

    # 模拟数据
    n_samples = 200
    y = model.simulate(n_samples, sigma=1.0)

    print(f"\n模型参数:")
    print(f"  AR 系数: {ar_params}")
    print(f"  MA 系数: {ma_params}")
    print(f"\n生成数据:")
    print(f"  样本数: {n_samples}")
    print(f"  均值: {y.mean():.3f}")
    print(f"  标准差: {y.std():.3f}")

    # 预测
    history = y[:150]
    predictions = model.predict(history, steps=50)

    print(f"\n预测:")
    print(f"  预测步数: 50")
    print(f"  实际值示例: {y[150:155]}")
    print(f"  预测值示例: {predictions[:5]}")

    # 可视化
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(y, label='ARMA(1,1) 时间序列', alpha=0.8)
    plt.title('ARMA(1,1) 模拟数据')
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(range(150, 200), y[150:], label='实际值', marker='o', markersize=3)
    plt.plot(range(150, 200), predictions, label='预测值', marker='x', markersize=3)
    plt.title('预测 vs 实际')
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('arma_example.png', dpi=100, bbox_inches='tight')
    print(f"\n图表已保存为 'arma_example.png'")


if __name__ == "__main__":
    example_arma_11()
