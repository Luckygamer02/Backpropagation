"""
Training module for neural network optimization using gradient descent.

This module implements the training loop for the MLP model, including:
- Forward and backward propagation orchestration
- Loss computation and tracking
- Parameter updates via gradient descent
- Model state snapshots for visualization

The training process demonstrates the complete machine learning pipeline
from data processing through model optimization.
"""

import numpy as np


def fit(model, X: np.ndarray, y: np.ndarray,
        epochs: int = 3000, lr: float = 0.5, snapshot_every: int = 50):
    """
    Train a model using manual backpropagation with gradient descent.

    This function implements the complete training loop:
    1. Forward pass: compute predictions and loss
    2. Backward pass: compute gradients via backpropagation
    3. Parameter update: apply gradients using gradient descent
    4. Progress tracking: record loss history and model snapshots

    The training process iteratively minimizes the binary cross-entropy loss
    by adjusting the network's weights and biases based on the computed gradients.

    Args:
        model: MLP instance with forward, backward, and step methods
        X (np.ndarray): Training input data of shape (m, input_dim)
        y (np.ndarray): Training target labels of shape (m, 1)
        epochs (int): Number of complete passes through the dataset. Default is 3000.
        lr (float): Learning rate controlling gradient descent step size. Default is 0.5.
        snapshot_every (int): Frequency of model state snapshots for visualization. Default is 50.

    Returns:
        tuple: A tuple containing:
            - history (dict): Training metrics including loss values for each epoch
            - snapshots (list): Model parameter snapshots at specified intervals

    Note:
        Higher learning rates lead to faster convergence but risk overshooting.
        Lower learning rates are more stable but require more epochs.
        The snapshot mechanism enables visualization of the learning process.
    """
    # Initialize containers for tracking training progress
    history = {"loss": []}  # Store loss value for each epoch
    snapshots = []  # Store model parameters at specified intervals

    # Training loop: iterate through specified number of epochs
    for epoch in range(1, epochs + 1):
        # === FORWARD PASS ===
        # Compute network predictions and cache intermediate values
        y_hat, cache = model.forward(X)

        # Calculate binary cross-entropy loss between predictions and true labels
        loss = model.bce_loss(y, y_hat)

        # Record loss for monitoring training progress
        history["loss"].append(loss)

        # === BACKWARD PASS ===
        # Compute gradients of loss w.r.t. all model parameters using backpropagation
        grads = model.backward(y, cache)

        # === PARAMETER UPDATE ===
        # Apply gradient descent: θ = θ - α * ∇θ
        model.step(grads, lr=lr)

        # === SNAPSHOT COLLECTION ===
        # Periodically save model state for visualization purposes
        # This allows us to see how the decision boundary evolves during training
        if epoch % snapshot_every == 0 or epoch == 1:
            snapshots.append({
                "epoch": epoch,  # Current training epoch
                "W1": model.W1.copy(),  # Input-to-hidden weights (copy to avoid reference issues)
                "b1": model.b1.copy(),  # Hidden layer biases
                "W2": model.W2.copy(),  # Hidden-to-output weights
                "b2": model.b2.copy(),  # Output layer bias
                "loss": float(loss),  # Current loss value (convert to Python float)
            })

    return history, snapshots
