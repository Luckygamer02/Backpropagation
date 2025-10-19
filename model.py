"""
Multi-Layer Perceptron (MLP) implementation with manual backpropagation.

This module implements a simple 2-layer neural network (2 → H → 1) designed for binary
classification tasks. The network uses tanh activation in the hidden layer and sigmoid
activation in the output layer, trained using manual backpropagation with binary
cross-entropy loss.

Key Features:
- Xavier/Glorot weight initialization for stable training
- Manual implementation of forward and backward passes
- Binary cross-entropy loss function
- Gradient descent optimization
"""

import numpy as np


class MLP:
    """
    A tiny 2-layer MLP with tanh hidden layer and sigmoid output for binary classification.

    Architecture: Input(2) → Hidden(H, tanh) → Output(1, sigmoid)

    This network is specifically designed to solve non-linearly separable problems
    like the XOR dataset, demonstrating why hidden layers are necessary for such tasks.

    Attributes:
        W1 (np.ndarray): Weight matrix from input to hidden layer, shape (input_dim, hidden_dim)
        b1 (np.ndarray): Bias vector for hidden layer, shape (1, hidden_dim)
        W2 (np.ndarray): Weight matrix from hidden to output layer, shape (hidden_dim, 1)
        b2 (np.ndarray): Bias vector for output layer, shape (1, 1)
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, seed: int = 0):
        """
        Initialize the MLP with Xavier/Glorot weight initialization.

        Xavier initialization helps maintain the variance of activations and gradients
        across layers, leading to more stable training.

        Args:
            input_dim (int): Number of input features. Default is 2 for XOR problem.
            hidden_dim (int): Number of neurons in the hidden layer. Default is 8.
            seed (int): Random seed for reproducible weight initialization. Default is 0.
        """
        # Set up random number generator for reproducible initialization
        rng = np.random.default_rng(seed)

        # Xavier/Glorot initialization for input-to-hidden weights
        # Formula: uniform(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
        # This helps maintain proper variance through the tanh activation
        limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = rng.uniform(-limit1, limit1, size=(input_dim, hidden_dim))
        # Initialize biases to zero (common practice)
        self.b1 = np.zeros((1, hidden_dim))

        # Xavier/Glorot initialization for hidden-to-output weights
        limit2 = np.sqrt(6.0 / (hidden_dim + 1))
        self.W2 = rng.uniform(-limit2, limit2, size=(hidden_dim, 1))
        self.b2 = np.zeros((1, 1))

    @staticmethod
    def _tanh(z):
        """
        Hyperbolic tangent activation function.

        tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

        Properties:
        - Range: (-1, 1)
        - Zero-centered output
        - Stronger gradients than sigmoid (less vanishing gradient problem)

        Args:
            z (np.ndarray): Input values to the activation function

        Returns:
            np.ndarray: Activated values in range (-1, 1)
        """
        return np.tanh(z)

    @staticmethod
    def _dtanh(a):
        """
        Derivative of the tanh function with respect to its input.

        If a = tanh(z), then d(tanh(z))/dz = 1 - tanh²(z) = 1 - a²

        This is used during backpropagation to compute gradients.

        Args:
            a (np.ndarray): Output of tanh function (tanh(z))

        Returns:
            np.ndarray: Derivative values, always positive and ≤ 1
        """
        # Given a = tanh(z), derivative is 1 - a²
        return 1.0 - a ** 2

    @staticmethod
    def _sigmoid(z):
        """
        Sigmoid activation function for binary classification output.

        σ(z) = 1 / (1 + e^(-z))

        Properties:
        - Range: (0, 1) - perfect for binary probabilities
        - Smooth and differentiable
        - Saturates at extremes (can cause vanishing gradients)

        Args:
            z (np.ndarray): Input values to the activation function

        Returns:
            np.ndarray: Activated values in range (0, 1) representing probabilities
        """
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):
        """
        Perform forward propagation through the network.

        This computes the network's prediction for the input data by passing
        it through both layers with their respective activation functions.

        Forward pass computation:
        1. Z1 = X @ W1 + b1          (linear transformation to hidden layer)
        2. A1 = tanh(Z1)             (hidden layer activation)
        3. Z2 = A1 @ W2 + b2         (linear transformation to output)
        4. A2 = sigmoid(Z2)          (output layer activation - probability)

        Args:
            X (np.ndarray): Input data of shape (m, input_dim) where m is batch size

        Returns:
            tuple: A tuple containing:
                - A2 (np.ndarray): Network predictions of shape (m, 1)
                - cache (dict): Cached intermediate values needed for backpropagation
        """
        # Linear transformation: input to hidden layer
        Z1 = X @ self.W1 + self.b1  # Shape: (m, hidden_dim)
        # Hidden layer activation using tanh
        A1 = self._tanh(Z1)  # Shape: (m, hidden_dim)
        # Linear transformation: hidden to output layer
        Z2 = A1 @ self.W2 + self.b2  # Shape: (m, 1)
        # Output layer activation using sigmoid for binary classification
        A2 = self._sigmoid(Z2)  # Shape: (m, 1)

        # Cache all intermediate values needed for backpropagation
        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    @staticmethod
    def bce_loss(y_true, y_hat, eps: float = 1e-10):
        """
        Compute Binary Cross-Entropy (BCE) loss for binary classification.

        BCE Loss = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]

        This loss function is ideal for binary classification as it:
        - Penalizes wrong predictions exponentially
        - Has nice gradient properties with sigmoid activation
        - Outputs probability-like values

        Args:
            y_true (np.ndarray): True binary labels of shape (m, 1)
            y_hat (np.ndarray): Predicted probabilities of shape (m, 1)
            eps (float): Small constant to prevent log(0). Default is 1e-10.

        Returns:
            float: Average binary cross-entropy loss across all samples
        """
        # Clip predictions to prevent numerical instability from log(0) or log(1)
        y_hat = np.clip(y_hat, eps, 1.0 - eps)

        # Compute binary cross-entropy loss
        # First term: -y * log(ŷ) penalizes false negatives
        # Second term: -(1-y) * log(1-ŷ) penalizes false positives
        return -np.mean(y_true * np.log(y_hat) + (1.0 - y_true) * np.log(1.0 - y_hat))

    def backward(self, y_true, cache):
        """
        Perform backward propagation to compute gradients.

        This implements the chain rule to compute gradients of the loss function
        with respect to all parameters (weights and biases) in the network.

        Backpropagation steps:
        1. Compute output layer gradients (sigmoid + BCE has nice derivative)
        2. Propagate gradients back to hidden layer using chain rule
        3. Compute gradients for all weights and biases

        Args:
            y_true (np.ndarray): True binary labels of shape (m, 1)
            cache (dict): Cached values from forward pass

        Returns:
            dict: Dictionary containing gradients for all parameters:
                - dW1: Gradient w.r.t. input-to-hidden weights
                - db1: Gradient w.r.t. hidden layer biases
                - dW2: Gradient w.r.t. hidden-to-output weights
                - db2: Gradient w.r.t. output layer biases
        """
        # Extract cached values from forward pass
        X = cache["X"]  # Input data
        A1 = cache["A1"]  # Hidden layer activations (after tanh)
        A2 = cache["A2"]  # Output layer activations (after sigmoid)
        m = X.shape[0]  # Batch size

        # Output layer gradients
        # For sigmoid + BCE, the gradient simplifies to: dZ2 = A2 - y_true
        # This is a beautiful mathematical result that makes training stable
        dZ2 = A2 - y_true

        # Gradient w.r.t. output layer weights: dW2 = (1/m) * A1^T @ dZ2
        dW2 = (A1.T @ dZ2) / m
        # Gradient w.r.t. output layer bias: db2 = (1/m) * sum(dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer gradients (using chain rule)
        # Gradient flowing back from output layer: dA1 = dZ2 @ W2^T
        dA1 = dZ2 @ self.W2.T
        # Apply derivative of tanh activation: dZ1 = dA1 * dtanh(A1)
        dZ1 = dA1 * self._dtanh(A1)

        # Gradient w.r.t. input layer weights: dW1 = (1/m) * X^T @ dZ1
        dW1 = (X.T @ dZ1) / m
        # Gradient w.r.t. input layer bias: db1 = (1/m) * sum(dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def step(self, grads, lr: float = 0.1):
        """
        Update model parameters using gradient descent.

        This performs the parameter update step:
        θ = θ - α * ∇θ

        where θ represents parameters (W1, b1, W2, b2), α is the learning rate,
        and ∇θ are the computed gradients.

        Args:
            grads (dict): Dictionary containing gradients for all parameters
            lr (float): Learning rate controlling the step size. Default is 0.1.
        """
        # Update weights and biases using gradient descent
        # Subtract learning_rate * gradient from each parameter
        self.W1 -= lr * grads["dW1"]  # Update input-to-hidden weights
        self.b1 -= lr * grads["db1"]  # Update hidden layer biases
        self.W2 -= lr * grads["dW2"]  # Update hidden-to-output weights
        self.b2 -= lr * grads["db2"]  # Update output layer bias

    def predict_proba(self, X):
        """
        Predict class probabilities for input data.

        This is a convenience method that performs only the forward pass
        without caching intermediate values (used for inference).

        Args:
            X (np.ndarray): Input data of shape (m, input_dim)

        Returns:
            np.ndarray: Predicted probabilities of shape (m, 1) in range (0, 1)
        """
        # Perform forward pass and return only predictions (ignore cache)
        y_hat, _ = self.forward(X)
        return y_hat

    def set_params(self, W1, b1, W2, b2):
        """
        Set model parameters to specific values.

        This method is useful for:
        - Restoring model state from snapshots
        - Animation/visualization of training progress
        - Model initialization from pre-trained weights

        Args:
            W1 (np.ndarray): Input-to-hidden weights
            b1 (np.ndarray): Hidden layer biases
            W2 (np.ndarray): Hidden-to-output weights
            b2 (np.ndarray): Output layer bias
        """
        # Copy parameters to avoid unintended modifications to original arrays
        self.W1 = W1.copy()
        self.b1 = b1.copy()
        self.W2 = W2.copy()
        self.b2 = b2.copy()
