"""
Visualization module for neural network training analysis and results.

This module provides functions to create visual representations of:
- Training progress through loss curves
- Decision boundary evolution during learning
- Animated visualizations showing how the model learns

The visualizations help understand:
- Whether the model is learning (decreasing loss)
- How the decision boundary adapts to separate classes
- The training dynamics and convergence behavior
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def plot_loss(history, outdir: Path):
    """
    Create a line plot showing the training loss over epochs.

    This visualization helps assess training progress by showing:
    - Whether the loss is decreasing (indicating learning)
    - The rate of convergence (steep vs. gradual decline)
    - Potential issues like plateauing or oscillations
    - Overall training stability and success

    Args:
        history (dict): Training history containing 'loss' key with list of loss values
        outdir (Path): Output directory where the plot will be saved

    Returns:
        Path: Full path to the saved loss plot image file

    Note:
        A well-training model should show:
        - Steady decrease in loss over time
        - Eventual convergence to a low, stable value
        - Smooth curve without excessive oscillations
    """
    # Ensure output directory exists
    outdir.mkdir(parents=True, exist_ok=True)

    # Create figure and axis for the loss plot
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot loss values against epoch numbers (starting from 1)
    # X-axis: epoch numbers (1, 2, 3, ...)
    # Y-axis: binary cross-entropy loss values
    ax.plot(np.arange(1, len(history["loss"]) + 1), history["loss"])

    # Set axis labels and title for clarity
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("Training Loss (Backpropagation)")

    # Save plot to file
    out_path = outdir / "loss.png"
    fig.savefig(out_path, bbox_inches="tight")

    # Close figure to free memory
    plt.close(fig)

    return out_path


def animate_decision_boundary(model, snapshots, X, y, outdir: Path,
                              grid_size: int = 200, fps: int = 4,
                              filename: str = "decision_boundary.gif"):
    """
    Create an animated GIF showing how the decision boundary evolves during training.

    This powerful visualization demonstrates:
    - How the neural network learns to separate classes
    - The gradual formation of non-linear decision boundaries
    - Why hidden layers are necessary for XOR-like problems
    - The training dynamics and parameter evolution effects

    The animation shows:
    - Data points colored by true class (circles for class 0, X's for class 1)
    - Background colored by predicted probability (darker = higher probability of class 1)
    - Current epoch and loss value in the title
    - Smooth transitions between training snapshots

    Args:
        model: MLP instance used for making predictions
        snapshots (list): List of model parameter snapshots from training
        X (np.ndarray): Training input data of shape (m, 2)
        y (np.ndarray): Training labels of shape (m, 1)
        outdir (Path): Output directory for saving the animation
        grid_size (int): Resolution of the decision boundary grid. Default is 200.
        fps (int): Frames per second for the animation. Default is 4.
        filename (str): Name of the output GIF file. Default is "decision_boundary.gif".

    Returns:
        Path: Full path to the saved animation file

    Technical Details:
        - Creates a dense grid of points across the input space [0,1]²
        - For each snapshot, computes model predictions on the grid
        - Colors the background based on predicted probabilities
        - Overlays the actual training data points
        - Animates the transition between snapshots
    """
    # Ensure output directory exists
    outdir.mkdir(parents=True, exist_ok=True)

    # Create a dense grid of points for visualizing the decision boundary
    # This grid covers the entire input space [0,1] × [0,1]
    gx = np.linspace(0, 1, grid_size)  # X coordinates
    gy = np.linspace(0, 1, grid_size)  # Y coordinates
    GX, GY = np.meshgrid(gx, gy)  # Create 2D coordinate grids

    # Reshape grid into (grid_size², 2) format for model input
    # Each row represents one point in the 2D space
    grid = np.stack([GX.ravel(), GY.ravel()], axis=1)

    # Set up the matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Plot training data points with different markers for each class
    # Create boolean masks to separate classes
    mask0 = (y.ravel() == 0)  # Points belonging to class 0
    mask1 = (y.ravel() == 1)  # Points belonging to class 1

    # Scatter plot: circles for class 0, X's for class 1
    ax.scatter(X[mask0, 0], X[mask0, 1], marker='o', s=20, label="class 0")
    ax.scatter(X[mask1, 0], X[mask1, 1], marker='x', s=20, label="class 1")

    # Set plot boundaries and styling
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Decision Boundary Evolution")
    ax.legend(loc="upper right")

    # Initialize the background image with first snapshot's predictions
    # This creates the colored decision boundary background
    probs_init = model.predict_proba(grid).reshape(grid_size, grid_size)
    img = ax.imshow(probs_init, origin='lower', extent=(0, 1, 0, 1),
                    alpha=0.5, interpolation='nearest')

    # Create text element for displaying current epoch and loss
    title_text = ax.text(0.02, 1.02, "", transform=ax.transAxes)

    def init():
        """
        Initialize animation elements.

        This function is called once at the beginning of the animation
        to set up the initial state.
        """
        title_text.set_text("")
        return img, title_text

    def animate(i):
        """
        Update function called for each frame of the animation.

        This function:
        1. Restores model parameters from the i-th snapshot
        2. Computes new predictions on the grid
        3. Updates the background image colors
        4. Updates the title with current epoch and loss

        Args:
            i (int): Frame index corresponding to snapshot index

        Returns:
            tuple: Updated matplotlib artists (for blit optimization)
        """
        # Get the current snapshot (model parameters at this training step)
        snap = snapshots[i]

        # Restore model to this snapshot's state
        model.set_params(snap["W1"], snap["b1"], snap["W2"], snap["b2"])

        # Compute predictions on the entire grid to show decision boundary
        probs = model.predict_proba(grid).reshape(grid_size, grid_size)

        # Update the background image with new probability predictions
        # Darker regions indicate higher probability of class 1
        img.set_data(probs)

        # Update title to show current training progress
        title_text.set_text(f"Epoch {snap['epoch']}  |  Loss {snap['loss']:.4f}")

        return img, title_text

    # Create the animation object
    # This orchestrates the frame-by-frame updates
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(snapshots), interval=200, blit=True)

    # Save animation as GIF file
    out_path = outdir / filename
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))

    # Close figure to free memory
    plt.close(fig)

    return out_path


def visualize_backpropagation(model, snapshots, X, y, outdir: Path,
                              filename: str = "backpropagation_flow.gif",
                              fps: int = 2):
    """
    Create an animated visualization showing how gradients flow during backpropagation.

    This visualization demonstrates the backpropagation algorithm by showing:
    - Network architecture with nodes and connections
    - Gradient magnitudes as line thickness and colors
    - How gradients propagate backward from output to input
    - The evolution of gradient patterns during training

    The animation helps understand:
    - How errors flow backward through the network
    - Which connections receive the largest gradient updates
    - How gradient patterns change as the network learns
    - The relationship between forward activations and backward gradients

    Args:
        model: MLP instance for computing gradients
        snapshots (list): List of model parameter snapshots from training
        X (np.ndarray): Training input data for gradient computation
        y (np.ndarray): Training labels for gradient computation
        outdir (Path): Output directory for saving the animation
        filename (str): Name of the output GIF file. Default is "backpropagation_flow.gif".
        fps (int): Frames per second for the animation. Default is 2 (slow for observation).

    Returns:
        Path: Full path to the saved backpropagation animation file

    Visualization Elements:
        - Circles represent neurons (input, hidden, output)
        - Lines represent connections (weights)
        - Line thickness indicates gradient magnitude
        - Line color intensity shows gradient direction/strength
        - Neuron colors show activation levels
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Get actual network dimensions from the model
    input_dim = model.W1.shape[0]
    hidden_dim = model.W1.shape[1]
    output_dim = model.W2.shape[1]

    # Set up the figure for network visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(6, hidden_dim * 0.7 + 2))  # Adjust height based on hidden neurons
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Backpropagation Gradient Flow Visualization", fontsize=16, pad=20)

    # Define network layout positions (dynamic based on actual architecture)
    # Input layer
    input_positions = [(1, 2), (1, 4)] if input_dim == 2 else [(1, i * 0.8 + 1.5) for i in range(input_dim)]
    # Hidden layer - arrange vertically based on actual hidden_dim
    hidden_positions = [(5, i * 0.6 + 1) for i in range(hidden_dim)]
    # Output layer
    output_positions = [(9, 3)]

    # Initialize visualization elements
    input_circles = []
    hidden_circles = []
    output_circles = []
    weight_lines_ih = []  # Input to hidden connections
    weight_lines_ho = []  # Hidden to output connections

    # Create neuron circles
    for pos in input_positions:
        circle = plt.Circle(pos, 0.15, color='lightblue', alpha=0.7)
        ax.add_patch(circle)
        input_circles.append(circle)

    for pos in hidden_positions:
        circle = plt.Circle(pos, 0.12, color='lightgreen', alpha=0.7)
        ax.add_patch(circle)
        hidden_circles.append(circle)

    for pos in output_positions:
        circle = plt.Circle(pos, 0.15, color='lightcoral', alpha=0.7)
        ax.add_patch(circle)
        output_circles.append(circle)

    # Create weight connection lines (based on actual architecture)
    for i, input_pos in enumerate(input_positions):
        for j, hidden_pos in enumerate(hidden_positions):
            line = ax.plot([input_pos[0], hidden_pos[0]],
                           [input_pos[1], hidden_pos[1]],
                           'gray', alpha=0.3, linewidth=0.5)[0]
            weight_lines_ih.append(line)

    for i, hidden_pos in enumerate(hidden_positions):
        for j, output_pos in enumerate(output_positions):
            line = ax.plot([hidden_pos[0], output_pos[0]],
                           [hidden_pos[1], output_pos[1]],
                           'gray', alpha=0.3, linewidth=0.5)[0]
            weight_lines_ho.append(line)

    # Add labels (dynamic based on architecture)
    ax.text(1, 0.5, f'Input Layer\n({input_dim} neurons)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, 0.2, f'Hidden Layer\n({hidden_dim} neurons, tanh)', ha='center', va='center', fontsize=10,
            fontweight='bold')
    ax.text(9, 0.5, f'Output Layer\n({output_dim} neuron, sigmoid)', ha='center', va='center', fontsize=10,
            fontweight='bold')

    # Add gradient flow indicators
    gradient_text = ax.text(5, max(5.5, hidden_dim * 0.6 + 2), '', ha='center', va='center', fontsize=12,
                            fontweight='bold')
    epoch_text = ax.text(0.5, max(5.5, hidden_dim * 0.6 + 2), '', ha='left', va='center', fontsize=12,
                         fontweight='bold')
    loss_text = ax.text(0.5, max(5.2, hidden_dim * 0.6 + 1.7), '', ha='left', va='center', fontsize=11)

    def animate_backprop(frame):
        """
        Update function for each frame of the backpropagation animation.

        This function:
        1. Restores model parameters from the current snapshot
        2. Computes forward pass to get activations
        3. Computes backward pass to get gradients
        4. Updates visualization based on gradient magnitudes
        5. Colors connections and neurons based on gradient flow

        Args:
            frame (int): Current animation frame index

        Returns:
            list: Updated matplotlib artists for animation
        """
        if frame >= len(snapshots):
            return []

        snap = snapshots[frame]

        # Restore model parameters
        model.set_params(snap["W1"], snap["b1"], snap["W2"], snap["b2"])

        # Compute forward pass and gradients on a small batch for visualization
        sample_size = min(50, X.shape[0])  # Use subset for clearer visualization
        X_sample = X[:sample_size]
        y_sample = y[:sample_size]

        # Forward pass
        y_hat, cache = model.forward(X_sample)

        # Backward pass to get gradients
        grads = model.backward(y_sample, cache)

        # Extract activations and gradients
        A1 = cache["A1"]  # Hidden layer activations
        A2 = cache["A2"]  # Output layer activations
        dW1 = grads["dW1"]  # Input-to-hidden weight gradients
        dW2 = grads["dW2"]  # Hidden-to-output weight gradients

        # Update neuron colors based on activation levels
        # Input neurons (show input values)
        for i, circle in enumerate(input_circles):
            if i < X_sample.shape[1]:  # Safety check
                activation = np.mean(X_sample[:, i])
                intensity = activation  # Already in [0,1] for XOR data
                circle.set_color(plt.cm.Blues(0.3 + 0.7 * intensity))

        # Hidden neurons (show tanh activations, map from [-1,1] to [0,1])
        for i, circle in enumerate(hidden_circles):
            if i < A1.shape[1]:  # Safety check for actual hidden layer size
                activation = np.mean(A1[:, i])
                intensity = (activation + 1) / 2  # Map [-1,1] to [0,1]
                circle.set_color(plt.cm.Greens(0.3 + 0.7 * intensity))

        # Output neuron (show sigmoid activation)
        output_activation = np.mean(A2)
        output_circles[0].set_color(plt.cm.Reds(0.3 + 0.7 * output_activation))

        # Update weight lines based on gradient magnitudes
        # Input-to-hidden connections
        max_grad_ih = np.max(np.abs(dW1)) if np.max(np.abs(dW1)) > 0 else 1
        line_idx = 0
        for i in range(input_dim):  # Use actual input dimension
            for j in range(hidden_dim):  # Use actual hidden dimension
                if i < dW1.shape[0] and j < dW1.shape[1]:  # Safety check
                    grad_magnitude = np.abs(dW1[i, j])
                    normalized_grad = grad_magnitude / max_grad_ih

                    # Set line thickness based on gradient magnitude
                    linewidth = 0.5 + 3.0 * normalized_grad
                    weight_lines_ih[line_idx].set_linewidth(linewidth)

                    # Set color based on gradient sign and magnitude
                    if dW1[i, j] > 0:
                        color = plt.cm.Reds(0.3 + 0.7 * normalized_grad)
                    else:
                        color = plt.cm.Blues(0.3 + 0.7 * normalized_grad)
                    weight_lines_ih[line_idx].set_color(color)
                    weight_lines_ih[line_idx].set_alpha(0.4 + 0.6 * normalized_grad)

                line_idx += 1

        # Hidden-to-output connections
        max_grad_ho = np.max(np.abs(dW2)) if np.max(np.abs(dW2)) > 0 else 1
        for i in range(hidden_dim):  # Use actual hidden dimension
            if i < dW2.shape[0]:  # Safety check
                grad_magnitude = np.abs(dW2[i, 0])
                normalized_grad = grad_magnitude / max_grad_ho

                linewidth = 0.5 + 3.0 * normalized_grad
                weight_lines_ho[i].set_linewidth(linewidth)

                if dW2[i, 0] > 0:
                    color = plt.cm.Reds(0.3 + 0.7 * normalized_grad)
                else:
                    color = plt.cm.Blues(0.3 + 0.7 * normalized_grad)
                weight_lines_ho[i].set_color(color)
                weight_lines_ho[i].set_alpha(0.4 + 0.6 * normalized_grad)

        # Update text information
        epoch_text.set_text(f"Epoch: {snap['epoch']}")
        loss_text.set_text(f"Loss: {snap['loss']:.4f}")

        # Create gradient flow description
        avg_grad_magnitude = (np.mean(np.abs(dW1)) + np.mean(np.abs(dW2))) / 2
        if avg_grad_magnitude > 0.1:
            flow_intensity = "High"
            flow_color = "red"
        elif avg_grad_magnitude > 0.01:
            flow_intensity = "Medium"
            flow_color = "orange"
        else:
            flow_intensity = "Low"
            flow_color = "green"

        gradient_text.set_text(f"Gradient Flow: {flow_intensity}")
        gradient_text.set_color(flow_color)

        return (input_circles + hidden_circles + output_circles +
                weight_lines_ih + weight_lines_ho +
                [gradient_text, epoch_text, loss_text])

    # Create animation
    anim = animation.FuncAnimation(fig, animate_backprop, frames=len(snapshots),
                                   interval=500, blit=False, repeat=True)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=3, alpha=0.7, label='Positive Gradients'),
        plt.Line2D([0], [0], color='blue', lw=3, alpha=0.7, label='Negative Gradients'),
        plt.Line2D([0], [0], color='gray', lw=1, label='Low Gradient Magnitude'),
        plt.Line2D([0], [0], color='black', lw=3, label='High Gradient Magnitude')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

    # Save animation
    out_path = outdir / filename
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)

    return out_path
