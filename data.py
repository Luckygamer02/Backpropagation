"""
Data generation module for creating synthetic datasets for neural network training.

This module provides functions to generate the classic XOR (exclusive OR) dataset,
which is a non-linearly separable problem commonly used to demonstrate the need
for hidden layers in neural networks.
"""

import numpy as np


def make_xor(n: int = 500, seed: int = 42):
    """
    Create a non-linear XOR dataset in the unit square [0,1]^2.

    The XOR problem is a classic example where linear classifiers fail.
    Points are classified as class 1 if they are in the top-left or bottom-right
    quadrants (x1 > 0.5 XOR x2 > 0.5), and class 0 otherwise.

    Args:
        n (int): Number of data points to generate. Default is 500.
        seed (int): Random seed for reproducible results. Default is 42.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): Input features of shape (n, 2) with values in [0,1]
            - y (np.ndarray): Binary labels of shape (n, 1) where 1 represents XOR=True

    Example:
        >>> x, y = make_xor(n=100, seed=42)
        >>> print(f"Dataset shape: X={x.shape}, y={y.shape}")
        Dataset shape: X=(100, 2), y=(100, 1)
    """
    # Initialize random number generator with specified seed for reproducibility
    rng = np.random.default_rng(seed)

    # Generate n random points uniformly distributed in the unit square [0,1]^2
    X = rng.random((n, 2))

    # Apply XOR logic: True if exactly one coordinate is > 0.5
    # This creates four regions:
    # - Bottom-left (0,0): both < 0.5 → False → class 0
    # - Top-left (0,1): x1 < 0.5, x2 > 0.5 → True → class 1
    # - Bottom-right (1,0): x1 > 0.5, x2 < 0.5 → True → class 1
    # - Top-right (1,1): both > 0.5 → False → class 0
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(np.float64)

    # Reshape y to column vector format expected by neural networks
    return X, y.reshape(-1, 1)
