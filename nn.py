import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

class AdaptivePiecewiseActivation:
    """
    Advanced piecewise activation function with learnable pieces and boundaries.
    Each piece can be a polynomial function with learnable coefficients.
    Boundaries between pieces are also learnable.
    """
    def __init__(self, input_dim=1, num_pieces=3, polynomial_degree=2, init_range=(-3, 3),
                 reg_lambda=0.01):
        # Initialize boundaries (sorted points that separate the pieces)
        self.num_pieces = num_pieces
        self.num_boundaries = num_pieces - 1
        
        # Evenly distribute initial boundaries across the specified range
        if self.num_boundaries > 0:
            self.boundaries = np.linspace(init_range[0], init_range[1], self.num_boundaries + 2)[1:-1]
        else:
            self.boundaries = np.array([])
        
        # Initialize polynomial coefficients for each piece - with smaller initial values
        self.polynomial_degree = polynomial_degree
        self.coefficients = np.zeros((num_pieces, polynomial_degree + 1))
        
        # Initialize with identity function-like segments but with smaller coefficients
        for i in range(num_pieces):
            if i == num_pieces // 2:  # Middle piece as identity
                self.coefficients[i, 1] = 0.5  # x^1 coefficient = 0.5 (reduced from 1.0)
            elif i < num_pieces // 2:  # Left pieces slightly attenuated
                self.coefficients[i, 1] = 0.4  # Reduced from 0.8
                self.coefficients[i, 0] = -0.1  # Reduced from -0.2
            else:  # Right pieces slightly amplified
                self.coefficients[i, 1] = 0.6  # Reduced from 1.2
                self.coefficients[i, 0] = 0.1   # Reduced from 0.2
        
        # Learning rate scaling factors - reduced for better stability
        self.boundary_lr_scale = 0.05  # Reduced from 0.1
        self.coefficient_lr_scale = 0.1  # Reduced from 0.2
        
        # Smoothing for piece transitions (higher = smoother)
        self.smoothing_factor = 3.0  # Reduced from 5.0 for more gradual transitions
        
        # History tracking for visualization
        self.boundary_history = [self.boundaries.copy()]
        self.coefficient_history = [self.coefficients.copy()]
        
        # For adaptive learning rates
        self.piece_usage_count = np.zeros(num_pieces)
        self.boundary_stress = np.zeros(self.num_boundaries)
        
        # Regularization parameter
        self.reg_lambda = reg_lambda
        
    def _get_piece_weights(self, x):
        """Calculate smooth weights for each piece at input x"""
        # Ensure x is 2D (batch_size, 1)
        x = np.clip(x, -10, 10)  # Clip input values to prevent extreme sigmoid inputs
        x = x.reshape(-1, 1)
        batch_size = x.shape[0]
        weights = np.ones((batch_size, self.num_pieces))
        
        for i, boundary in enumerate(self.boundaries):
            # Calculate transition with proper broadcasting
            transition = sigmoid(self.smoothing_factor * (x - boundary)).flatten()
            
            # Update weights for left and right pieces
            for p in range(i + 1):
                weights[:, p] *= (1 - transition)
            for p in range(i + 1, self.num_pieces):
                weights[:, p] *= transition
        
        # Normalize weights and add small epsilon to prevent division by zero
        epsilon = 1e-10
        return weights / (weights.sum(axis=1, keepdims=True) + epsilon)

    def visualize(self, x_range=(-5, 5), num_points=1000, show_boundaries=True, 
                  show_pieces=True, title="Adaptive Piecewise Activation"):
        """Visualize the current state of the activation function"""
        x = np.linspace(x_range[0], x_range[1], num_points).reshape(-1, 1)
        y = self.forward(x)
        
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, 'b-', linewidth=2, label='Activation Function')
        
        if show_boundaries and len(self.boundaries) > 0:
            for boundary in self.boundaries:
                plt.axvline(x=boundary, color='r', linestyle='--', alpha=0.5)
        
        if show_pieces:
            piece_weights = self._get_piece_weights(x)
            for p in range(self.num_pieces):
                piece_output = self._evaluate_polynomial(x, self.coefficients[p])
                weighted_output = piece_output * piece_weights[:, p]
                plt.plot(x, weighted_output, '--', alpha=0.3, 
                         label=f'Piece {p+1} Contribution')
        
        plt.grid(True)
        plt.title(title)
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def _evaluate_polynomial(self, x, coeffs):
        """Evaluate polynomial with given coefficients at x, with safety measures"""
        # Clip x to prevent extreme values in high-power terms
        x_clipped = np.clip(x, -5, 5)
        result = np.zeros_like(x_clipped)
        
        # Use lower powers first, with cumulative multiplication to avoid extreme values
        for i, coef in enumerate(coeffs):
            if i == 0:
                term = coef
            elif i == 1:
                term = coef * x_clipped
            else:
                # For higher powers, use a more numerically stable approach
                # Avoid direct x**i which can explode
                term = coef * np.power(x_clipped, min(i, 4))  # Limit max power to 4
                
            result += term
            
        return np.clip(result.ravel(), -100, 100)  # Clip final result for stability

    def forward(self, x):
        """Forward pass with proper dimension handling and value clipping"""
        x = np.asarray(x).reshape(-1, 1)  # Ensure 2D input
        batch_size = x.shape[0]
        piece_weights = self._get_piece_weights(x)
        
        piece_outputs = np.zeros((batch_size, self.num_pieces))
        for p in range(self.num_pieces):
            piece_output = self._evaluate_polynomial(x, self.coefficients[p])
            piece_outputs[:, p] = piece_output
            
        # Clip output to prevent extreme values
        output = np.sum(piece_weights * piece_outputs, axis=1)
        return np.clip(output, -100, 100)  # Add clipping for stability
    
    def get_derivative(self, x):
        """
        Calculate derivative of the activation function at x.
        This is needed for backpropagation.
        """
        # Clip x for numerical stability
        x = np.clip(x, -10, 10)
        batch_size = x.shape[0]
        piece_weights = self._get_piece_weights(x)
        
        # For each piece, calculate derivative of its polynomial
        piece_derivatives = np.zeros((batch_size, self.num_pieces))
        for p in range(self.num_pieces):
            # Derivative of a polynomial: sum(i * coef[i] * x^(i-1))
            for i in range(1, self.polynomial_degree + 1):
                # Limit power to 3 for derivatives to prevent overflow
                safe_power = min(i-1, 3)
                piece_derivatives[:, p] += i * self.coefficients[p, i] * (np.clip(x, -5, 5) ** safe_power)
        
        # Now calculate derivatives of the weight functions (sigmoid transitions)
        weight_derivatives = np.zeros((batch_size, self.num_pieces))
        
        for i, boundary in enumerate(self.boundaries):
            # Derivative of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            sig = sigmoid(self.smoothing_factor * (x - boundary))
            sig_derivative = self.smoothing_factor * sig * (1 - sig)
            
            # Apply chain rule to update derivatives of weights
            for p in range(self.num_pieces):
                if p <= i:  # Pieces to the left
                    weight_derivatives[:, p] -= sig_derivative
                else:  # Pieces to the right
                    weight_derivatives[:, p] += sig_derivative
        
        # Calculate piece output values
        piece_output_values = np.zeros((batch_size, self.num_pieces))
        for p in range(self.num_pieces):
            piece_output_values[:, p] = self._evaluate_polynomial(x, self.coefficients[p])
        
        # Product rule: d(w*f)/dx = dw/dx * f + w * df/dx
        derivative_from_weights = np.sum(weight_derivatives * piece_output_values, axis=1)
        derivative_from_pieces = np.sum(piece_weights * piece_derivatives, axis=1)
        
        total_derivative = derivative_from_weights + derivative_from_pieces
        # Clip derivatives to prevent gradient explosion
        return np.clip(total_derivative, -10, 10)
    
    def compute_coefficient_gradients(self, x, output_grad):
        """
        Compute gradients for polynomial coefficients based on output gradient.
        Returns gradients for all coefficients in all pieces.
        Now includes L2 regularization term and gradient clipping.
        """
        # Safety - clip input values and output gradient
        x = np.clip(x, -10, 10)
        output_grad = np.clip(output_grad, -10, 10)
        
        batch_size = x.shape[0]
        piece_weights = self._get_piece_weights(x)
        
        # Initialize gradients for all coefficients
        coefficient_grads = np.zeros_like(self.coefficients)
        
        # For each piece and coefficient, compute gradient
        for p in range(self.num_pieces):
            for power in range(self.polynomial_degree + 1):
                # Use a safer computation for x^power
                if power == 0:
                    x_power = np.ones_like(x)
                elif power == 1:
                    x_power = x
                else:
                    # For higher powers, clip more aggressively
                    max_safe_power = min(power, 3)  # Limit to cubic terms
                    x_power = np.power(np.clip(x, -3, 3), max_safe_power)
                
                # Gradient = sum(output_grad * piece_weight * x^power)
                weighted_grad = output_grad * piece_weights[:, p] * x_power.flatten()
                
                # Clip individual gradients before summing
                weighted_grad = np.clip(weighted_grad, -100, 100)
                coefficient_grads[p, power] = np.sum(weighted_grad)
                
                # Add L2 regularization gradient: lambda * coefficient
                coefficient_grads[p, power] += self.reg_lambda * self.coefficients[p, power]
        
        # Global gradient clipping
        coefficient_grads = np.clip(coefficient_grads, -1.0, 1.0)
        return coefficient_grads
    
    def compute_boundary_gradients(self, x, output_grad):
        """
        Compute gradients for boundaries based on output gradient.
        Now with improved numerical stability.
        """
        if self.num_boundaries == 0:
            return np.array([])
        
        # Safety - clip input values and output gradient
        x = np.clip(x, -10, 10)
        output_grad = np.clip(output_grad, -10, 10)
            
        batch_size = x.shape[0]
        boundary_grads = np.zeros(self.num_boundaries)
        
        # Get current piece outputs
        piece_outputs = np.zeros((batch_size, self.num_pieces))
        for p in range(self.num_pieces):
            piece_outputs[:, p] = self._evaluate_polynomial(x, self.coefficients[p])
        
        # For each boundary, compute how changing it affects output
        for b_idx, boundary in enumerate(self.boundaries):
            # Sigmoid transition at this boundary
            sig = sigmoid(self.smoothing_factor * (x - boundary))
            sig_derivative = self.smoothing_factor * sig * (1 - sig)
            
            # Compute effect on each sample
            boundary_effect = np.zeros(batch_size)
            
            # Moving boundary right increases weight of left pieces, decreases right pieces
            for i in range(batch_size):
                left_effect = 0
                right_effect = 0
                
                # Effect on pieces to the left
                for p in range(b_idx + 1):
                    left_effect += piece_outputs[i, p]
                
                # Effect on pieces to the right
                for p in range(b_idx + 1, self.num_pieces):
                    right_effect += piece_outputs[i, p]
                
                # Net effect is the difference between left and right, weighted by sigmoid derivative
                boundary_effect[i] = (left_effect - right_effect) * sig_derivative[i]
            
            # Clip boundary effect to prevent extreme values
            boundary_effect = np.clip(boundary_effect, -10, 10)
            
            # Gradient is sum of effects weighted by output gradient
            boundary_grads[b_idx] = np.sum(output_grad * boundary_effect)
            
            # Track boundary stress for adaptive learning
            self.boundary_stress[b_idx] = np.abs(boundary_grads[b_idx])
        
        # Final gradient clipping
        boundary_grads = np.clip(boundary_grads, -1.0, 1.0)
        return boundary_grads
    
    def update_parameters(self, x, output_grad, global_learning_rate):
        """
        Update all parameters based on gradients and learning rate.
        Uses adaptive learning rates based on piece usage and boundary stress.
        """
        # Clip output gradients for numerical stability
        output_grad = np.clip(output_grad, -10, 10)
        
        # Compute all gradients
        coefficient_grads = self.compute_coefficient_gradients(x, output_grad)
        boundary_grads = self.compute_boundary_gradients(x, output_grad)
        
        # Use piece usage to scale coefficient learning rates
        piece_lr_factors = np.ones(self.num_pieces)
        if np.sum(self.piece_usage_count) > 0:
            normalized_usage = self.piece_usage_count / (np.sum(self.piece_usage_count) + 1e-10)
            # Pieces used less get higher learning rates (to encourage adaptation)
            piece_lr_factors = 1.0 / (normalized_usage + 0.1)
            piece_lr_factors = piece_lr_factors / np.max(piece_lr_factors)  # Normalize
        
        # Update coefficients with adaptive learning rates - use smaller effective learning rate
        for p in range(self.num_pieces):
            effective_lr = global_learning_rate * self.coefficient_lr_scale * piece_lr_factors[p]
            # Apply smaller learning rate for more stability
            effective_lr *= 0.5
            self.coefficients[p] -= effective_lr * coefficient_grads[p]
            
            # Coefficient clipping to prevent overflow (stricter bounds)
            self.coefficients[p] = np.clip(self.coefficients[p], -5.0, 5.0)
        
        # Update boundaries with adaptive learning rates
        if self.num_boundaries > 0:
            boundary_lr_factors = np.ones(self.num_boundaries)
            # Higher stress boundaries get higher learning rates
            if np.sum(self.boundary_stress) > 0:
                normalized_stress = self.boundary_stress / (np.sum(self.boundary_stress) + 1e-10)
                boundary_lr_factors = normalized_stress + 0.5  # Base factor + stress factor
            
            # Update each boundary with reduced learning rate for stability
            for b_idx in range(self.num_boundaries):
                effective_lr = (global_learning_rate * self.boundary_lr_scale * 
                               boundary_lr_factors[b_idx] * 0.5)  # Additional 0.5 factor
                self.boundaries[b_idx] -= effective_lr * boundary_grads[b_idx]
            
            # Sort boundaries to maintain ordering
            self.boundaries = np.sort(self.boundaries)
        
        # Calculate regularization loss (for monitoring)
        reg_loss = 0.5 * self.reg_lambda * np.sum(self.coefficients**2)
        
        # Record history for visualization
        self.boundary_history.append(self.boundaries.copy())
        self.coefficient_history.append(self.coefficients.copy())
        
        # Reset usage counts and stress periodically
        if len(self.boundary_history) % 100 == 0:
            self.piece_usage_count = np.zeros(self.num_pieces)
            self.boundary_stress = np.zeros(self.num_boundaries)
            
        return {
            'coefficient_grads': coefficient_grads,
            'boundary_grads': boundary_grads,
            'piece_usage': self.piece_usage_count,
            'boundary_stress': self.boundary_stress,
            'reg_loss': reg_loss
        }
        
    def get_regularization_loss(self):
        """Calculate the L2 regularization loss"""
        return 0.5 * self.reg_lambda * np.sum(self.coefficients**2)

    def visualize_history(self, step_interval=10, x_range=(-5, 5), num_points=200):
        """Visualize how the activation function evolved during training"""
        # Set up the figure
        plt.figure(figsize=(15, 10))
        
        # Calculate number of snapshots to show
        num_snapshots = len(self.boundary_history) // step_interval
        if num_snapshots > 10:  # Limit number of snapshots for clarity
            step_interval = len(self.boundary_history) // 10
            num_snapshots = 10
        
        # Generate x values for evaluation
        x = np.linspace(x_range[0], x_range[1], num_points).reshape(-1, 1)
        
        # Create a colormap for the snapshots
        cmap = plt.cm.viridis
        colors = [cmap(i / max(1, num_snapshots - 1)) for i in range(num_snapshots)]
        
        # Plot each snapshot
        for i in range(0, len(self.boundary_history), step_interval):
            if i // step_interval >= len(colors):
                break
                
            # Temporarily save current state
            original_boundaries = self.boundaries.copy()
            original_coefficients = self.coefficients.copy()
            
            # Set historical state
            self.boundaries = self.boundary_history[i]
            self.coefficients = self.coefficient_history[i]
            
            # Evaluate function
            y = self.forward(x)
            
            # Plot
            plt.plot(x, y, color=colors[i // step_interval], 
                     alpha=0.7 + 0.3 * (i / len(self.boundary_history)), 
                     label=f'Iteration {i}')
            
            # Restore original state
            self.boundaries = original_boundaries
            self.coefficients = original_coefficients
        
        plt.grid(True)
        plt.title("Evolution of Activation Function During Training")
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.legend()
        plt.tight_layout()
        plt.show()


class NeuronWithAdaptiveActivation:
    """A single neuron with an adaptive piecewise activation function"""
    def __init__(self, input_size, 
                 num_pieces=3, 
                 polynomial_degree=2,
                 activation_range=(-3, 3),
                 reg_lambda=0.01):
        # Initialize weights and bias with smaller values
        self.input_size = input_size
        self.weights = np.random.randn(input_size) * np.sqrt(0.5 / input_size)  # Reduced scaling factor
        self.bias = np.zeros(1)
        
        # Initialize the adaptive activation function
        self.activation = AdaptivePiecewiseActivation(
            num_pieces=num_pieces,
            polynomial_degree=polynomial_degree,
            init_range=activation_range,
            reg_lambda=reg_lambda
        )
        
        # For tracking during backprop
        self.last_input = None
        self.last_z = None  # Pre-activation
        self.last_output = None
        
        # Metrics
        self.weight_update_magnitude_history = []
        self.activation_update_magnitude_history = []
        self.reg_loss_history = []  # Track regularization loss
        
    def forward(self, x):
        """Forward pass through the neuron with value clipping for stability"""
        self.last_input = x
        # Compute weighted input with clipping to prevent overflow
        self.last_z = np.clip(np.dot(x, self.weights) + self.bias, -10, 10)
        self.last_output = self.activation.forward(self.last_z)
        return self.last_output
    
    def backward(self, output_grad, learning_rate):
        """
        Backward pass with combined weight and activation function update.
        Now with gradient clipping for numerical stability.
        Returns gradient with respect to inputs.
        """
        # Clip output gradients
        output_grad = np.clip(output_grad, -10, 10)
        
        # Get the derivative of the activation function at the pre-activation values
        activation_derivative = self.activation.get_derivative(self.last_z)
        
        # Gradient through the activation function
        pre_activation_grad = output_grad * activation_derivative
        
        # Clip gradients to prevent extreme values
        pre_activation_grad = np.clip(pre_activation_grad, -10, 10)
        
        # Gradients for weights and bias
        input_m = self.last_input.shape[0]  # Batch size
        weight_grad = (1/input_m) * np.dot(self.last_input.T, pre_activation_grad)
        bias_grad = (1/input_m) * np.sum(pre_activation_grad)
        
        # Clip weight and bias gradients
        weight_grad = np.clip(weight_grad, -1.0, 1.0)
        bias_grad = np.clip(bias_grad, -1.0, 1.0)
        
        # Update weights and bias with smaller learning rate
        reduced_lr = learning_rate * 0.5  # Add safety factor
        weight_update = reduced_lr * weight_grad
        self.weights -= weight_update 
        self.bias -= reduced_lr * bias_grad
        
        # Clip weights and bias to prevent extreme values
        self.weights = np.clip(self.weights, -5.0, 5.0)
        self.bias = np.clip(self.bias, -5.0, 5.0)
        
        # Update the activation function parameters
        activation_update_info = self.activation.update_parameters(
            self.last_z, output_grad, reduced_lr)
        
        # Track update magnitudes for analysis
        self.weight_update_magnitude_history.append(np.linalg.norm(weight_update))
        if 'coefficient_grads' in activation_update_info:
            coef_update_norm = np.linalg.norm(activation_update_info['coefficient_grads']) 
            self.activation_update_magnitude_history.append(coef_update_norm)
        
        # Track regularization loss
        if 'reg_loss' in activation_update_info:
            self.reg_loss_history.append(activation_update_info['reg_loss'])
        
        # Gradient with respect to inputs (for upstream propagation)
        input_grad = np.outer(pre_activation_grad, self.weights)
        
        # Clip input gradient
        input_grad = np.clip(input_grad, -10, 10)
        
        return input_grad
    
    def visualize_updates(self):
        """Visualize the relative magnitudes of weight vs activation updates"""
        plt.figure(figsize=(12, 6))
        
        # Plot weight update magnitudes
        plt.subplot(1, 2, 1)
        plt.plot(self.weight_update_magnitude_history, 'b-', label='Weight Updates')
        plt.title('Weight Update Magnitudes')
        plt.xlabel('Training Step')
        plt.ylabel('Update Magnitude')
        plt.grid(True)
        
        # Plot activation update magnitudes if available
        if self.activation_update_magnitude_history:
            plt.subplot(1, 2, 2)
            plt.plot(self.activation_update_magnitude_history, 'r-', 
                    label='Activation Fn Updates')
            plt.title('Activation Function Update Magnitudes')
            plt.xlabel('Training Step')
            plt.ylabel('Update Magnitude')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot regularization loss if available
        if self.reg_loss_history:
            plt.figure(figsize=(10, 5))
            plt.plot(self.reg_loss_history, 'g-', label='Regularization Loss')
            plt.title('Regularization Loss Over Time')
            plt.xlabel('Training Step')
            plt.ylabel('L2 Loss')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
    def visualize_activation(self, title="Current Activation Function"):
        """Visualize with proper dimension handling"""
        x = np.linspace(-5, 5, 1000).reshape(-1, 1)
        y = self.activation.forward(x)
        
        plt.figure(figsize=(12, 6))
        plt.plot(x.ravel(), y.ravel(), 'b-', linewidth=2, label='Activation Function')  # Use ravel()
        
        if self.activation.boundaries is not None:
            for boundary in self.activation.boundaries:
                plt.axvline(x=boundary, color='r', linestyle='--', alpha=0.5)
        
        plt.title(title)
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def visualize_activation_evolution(self):
        """Visualize how the activation function evolved during training"""
        self.activation.visualize_history()
        
    def print_activation_details(self):
        """Print activation details with proper formatting"""
        print("\nLearned Activation Function Details:")
        print(f"Number of pieces: {self.activation.num_pieces}")
        print(f"Polynomial degree: {self.activation.polynomial_degree}")
        print(f"Regularization lambda: {self.activation.reg_lambda}")
        
        for i, coeffs in enumerate(self.activation.coefficients):
            poly_str = " + ".join([f"{c:.3f}x^{p}" for p, c in enumerate(coeffs)])
            print(f"\nPiece {i+1}:")
            print(f"  Polynomial: {poly_str}")
            if i < len(self.activation.boundaries):
                print(f"  Right boundary: {self.activation.boundaries[i]:.3f}")

# Example usage
def train_neuron_on_complex_function(reg_lambda=0.01):
    """Example: Train a neuron with adaptive activation on a complex function"""
    # Generate synthetic data: y = sin(x) * exp(-0.1*x^2) + noise
    np.random.seed(42)
    x_train = np.random.uniform(-5, 5, 1000).reshape(-1, 1)
    y_true = np.sin(x_train) * np.exp(-0.1 * x_train**2)
    y_train = y_true + 0.05 * np.random.randn(len(x_train), 1)
    
    # Normalize data for better numerical stability
    x_mean = np.mean(x_train)
    x_std = np.std(x_train) 
    x_train_normalized = (x_train - x_mean) / x_std
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train_normalized = (y_train - y_mean) / y_std
    
    print(f"Data statistics before normalization:")
    print(f"  X mean: {x_mean:.4f}, std: {x_std:.4f}")
    print(f"  Y mean: {y_mean:.4f}, std: {y_std:.4f}")
    
    # Create a neuron with adaptive activation and regularization
    # Use smaller polynomial degree for better stability
    neuron = NeuronWithAdaptiveActivation(
        input_size=1, 
        num_pieces=3,  # Reduced from 5
        polynomial_degree=2,  # Reduced from 3
        reg_lambda=reg_lambda
    )
    
    # Visualize initial activation function
    print("Initial activation function:")
    neuron.visualize_activation("Initial Activation Function")
    
    # Training parameters - use smaller learning rate and batch size
    epochs = 2000
    learning_rate = 0.005  # Reduced from 0.01
    batch_size = 16  # Reduced from 32
    
    # Training loop
    losses = []
    mse_losses = []  # Track MSE loss separately from regularization
    reg_losses = []  # Track regularization losses
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(x_train_normalized))
        x_shuffled = x_train_normalized[indices]
        y_shuffled = y_train_normalized[indices]
        
        # Process in mini-batches
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_reg_loss = 0
        
        for i in range(0, len(x_train_normalized), batch_size):
            x_batch = x_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            # Forward pass
            y_pred = neuron.forward(x_batch)
            
            # Compute MSE loss
            mse_loss = np.mean((y_pred - y_batch.ravel())**2)
            
            # Compute output gradient (derivative of MSE loss)
            output_grad = 2 * (y_pred - y_batch.ravel()) / batch_size
            
            # Backward pass
            neuron.backward(output_grad, learning_rate)
            
            # Get regularization loss
            reg_loss = neuron.activation.get_regularization_loss()
            
            # Accumulate losses
            batch_total_loss = mse_loss + reg_loss
            epoch_loss += batch_total_loss * len(x_batch)
            epoch_mse_loss += mse_loss * len(x_batch)
            epoch_reg_loss += reg_loss * len(x_batch)
        
        # Calculate average epoch losses
        epoch_loss /= len(x_train_normalized)
        epoch_mse_loss /= len(x_train_normalized)
        epoch_reg_loss /= len(x_train_normalized)
        
        losses.append(epoch_loss)
        mse_losses.append(epoch_mse_loss)
        reg_losses.append(epoch_reg_loss)
        
        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:.4f} (MSE: {epoch_mse_loss:.4f}, Reg: {epoch_reg_loss:.4f})")
    
    # Visualization after training
    print("\nFinal activation function:")
    neuron.visualize_activation("Trained Activation Function")
    neuron.visualize_activation_evolution()
    neuron.visualize_updates()
    
    # Plot training curves
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label='Total Loss')
    plt.plot(mse_losses, label='MSE Loss')
    plt.plot(reg_losses, label='Reg Loss')
    plt.legend()
    plt.title("Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

class GELUActivation:
    """
    Implementation of the Gaussian Error Linear Unit (GELU) activation function.
    GELU(x) = x * Φ(x) where Φ is the standard Gaussian CDF.
    
    We use the approximation: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x^3)))
    """
    def __init__(self):
        # No parameters to learn for GELU
        pass
    
    def forward(self, x):
        """Forward pass through GELU activation"""
        # Use the approximation for GELU that's more computationally efficient
        sqrt_2_over_pi = np.sqrt(2 / np.pi)
        x = np.asarray(x).reshape(-1, 1)
        inner = sqrt_2_over_pi * (x + 0.044715 * np.power(x, 3))
        return 0.5 * x.flatten() * (1 + np.tanh(inner).flatten())
    
    def get_derivative(self, x):
        """
        Calculate derivative of GELU.
        Using the approximation form's derivative.
        """
        sqrt_2_over_pi = np.sqrt(2 / np.pi)
        x = np.asarray(x).reshape(-1, 1)
        
        # Compute intermediate values for clarity
        term1 = 0.044715 * np.power(x, 3)
        inner = sqrt_2_over_pi * (x + term1)
        tanh_inner = np.tanh(inner)
        
        # The derivative has multiple terms
        deriv_term1 = 0.5 * (1 + tanh_inner.flatten())
        
        # Derivative of tanh(x) is (1 - tanh^2(x))
        sech_squared = 1 - np.power(tanh_inner, 2)
        deriv_term2 = 0.5 * x.flatten() * sech_squared.flatten() * sqrt_2_over_pi * (1 + 3 * 0.044715 * np.power(x, 2)).flatten()
        
        return deriv_term1 + deriv_term2


class NeuronWithGELU:
    """A single neuron with GELU activation function"""
    def __init__(self, input_size):
        # Initialize weights and bias
        self.input_size = input_size
        self.weights = np.random.randn(input_size) * np.sqrt(0.5 / input_size)
        self.bias = np.zeros(1)
        
        # Initialize the GELU activation function
        self.activation = GELUActivation()
        
        # For tracking during backprop
        self.last_input = None
        self.last_z = None  # Pre-activation
        self.last_output = None
        
        # Metrics
        self.weight_update_magnitude_history = []
        
    def forward(self, x):
        """Forward pass through the neuron with value clipping for stability"""
        self.last_input = x
        # Compute weighted input with clipping to prevent overflow
        self.last_z = np.clip(np.dot(x, self.weights) + self.bias, -10, 10)
        self.last_output = self.activation.forward(self.last_z)
        return self.last_output
    
    def backward(self, output_grad, learning_rate):
        """
        Backward pass with weight updates.
        Returns gradient with respect to inputs.
        """
        # Clip output gradients
        output_grad = np.clip(output_grad, -10, 10)
        
        # Get the derivative of the activation function at the pre-activation values
        activation_derivative = self.activation.get_derivative(self.last_z)
        
        # Gradient through the activation function
        pre_activation_grad = output_grad * activation_derivative
        
        # Clip gradients to prevent extreme values
        pre_activation_grad = np.clip(pre_activation_grad, -10, 10)
        
        # Gradients for weights and bias
        input_m = self.last_input.shape[0]  # Batch size
        weight_grad = (1/input_m) * np.dot(self.last_input.T, pre_activation_grad)
        bias_grad = (1/input_m) * np.sum(pre_activation_grad)
        
        # Clip weight and bias gradients
        weight_grad = np.clip(weight_grad, -1.0, 1.0)
        bias_grad = np.clip(bias_grad, -1.0, 1.0)
        
        # Update weights and bias with smaller learning rate
        reduced_lr = learning_rate * 0.5  # Add safety factor
        weight_update = reduced_lr * weight_grad
        self.weights -= weight_update 
        self.bias -= reduced_lr * bias_grad
        
        # Clip weights and bias to prevent extreme values
        self.weights = np.clip(self.weights, -5.0, 5.0)
        self.bias = np.clip(self.bias, -5.0, 5.0)
        
        # Track update magnitudes for analysis
        self.weight_update_magnitude_history.append(np.linalg.norm(weight_update))
        
        # Gradient with respect to inputs (for upstream propagation)
        input_grad = np.outer(pre_activation_grad, self.weights)
        
        # Clip input gradient
        input_grad = np.clip(input_grad, -10, 10)
        
        return input_grad
    
    def visualize_updates(self):
        """Visualize the magnitudes of weight updates"""
        plt.figure(figsize=(10, 5))
        
        # Plot weight update magnitudes
        plt.plot(self.weight_update_magnitude_history, 'b-', label='Weight Updates')
        plt.title('Weight Update Magnitudes')
        plt.xlabel('Training Step')
        plt.ylabel('Update Magnitude')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def visualize_activation(self, x_range=(-5, 5), num_points=1000, title="GELU Activation Function"):
        """Visualize GELU activation function"""
        x = np.linspace(x_range[0], x_range[1], num_points).reshape(-1, 1)
        y = self.activation.forward(x)
        
        plt.figure(figsize=(10, 5))
        plt.plot(x.ravel(), y, 'b-', linewidth=2, label='GELU')
        plt.title(title)
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def compare_activations(reg_lambda=0.01):
    """Compare AdaptivePiecewise activation to GELU on the same complex function"""
    # Generate synthetic data: y = sin(x) * exp(-0.1*x^2) + noise
    np.random.seed(42)
    x_train = np.random.uniform(-5, 5, 1000).reshape(-1, 1)
    y_true = np.sin(x_train) * np.exp(-0.1 * x_train**2)
    y_train = y_true + 0.05 * np.random.randn(len(x_train), 1)
    
    # Normalize data for better numerical stability
    x_mean = np.mean(x_train)
    x_std = np.std(x_train) 
    x_train_normalized = (x_train - x_mean) / x_std
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train_normalized = (y_train - y_mean) / y_std
    
    print("Data statistics:")
    print(f"  X mean: {x_mean:.4f}, std: {x_std:.4f}")
    print(f"  Y mean: {y_mean:.4f}, std: {y_std:.4f}")
    
    # Create both models
    adaptive_neuron = NeuronWithAdaptiveActivation(
        input_size=1, 
        num_pieces=3,
        polynomial_degree=2,
        reg_lambda=reg_lambda
    )
    
    gelu_neuron = NeuronWithGELU(input_size=1)
    
    # Visualize both initial activation functions
    print("\nInitial activation functions:")
    adaptive_neuron.visualize_activation("Initial Adaptive Piecewise Activation")
    gelu_neuron.visualize_activation(title="GELU Activation Function")
    
    # Training parameters - use same parameters for both models
    epochs = 2000
    learning_rate = 0.005
    batch_size = 16
    
    # Prepare for tracking results
    adaptive_losses = []
    gelu_losses = []
    
    # Training loop for both models
    for epoch in range(epochs):
        # Shuffle data (use the same shuffle for both models for fair comparison)
        indices = np.random.permutation(len(x_train_normalized))
        x_shuffled = x_train_normalized[indices]
        y_shuffled = y_train_normalized[indices]
        
        # Process in mini-batches
        adaptive_epoch_loss = 0
        gelu_epoch_loss = 0
        
        for i in range(0, len(x_train_normalized), batch_size):
            x_batch = x_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Train Adaptive model
            adaptive_pred = adaptive_neuron.forward(x_batch)
            adaptive_mse = np.mean((adaptive_pred - y_batch.ravel())**2)
            adaptive_output_grad = 2 * (adaptive_pred - y_batch.ravel()) / batch_size
            adaptive_neuron.backward(adaptive_output_grad, learning_rate)
            adaptive_reg_loss = adaptive_neuron.activation.get_regularization_loss()
            adaptive_total_loss = adaptive_mse + adaptive_reg_loss
            adaptive_epoch_loss += adaptive_total_loss * len(x_batch)
            
            # Train GELU model
            gelu_pred = gelu_neuron.forward(x_batch)
            gelu_mse = np.mean((gelu_pred - y_batch.ravel())**2)
            gelu_output_grad = 2 * (gelu_pred - y_batch.ravel()) / batch_size
            gelu_neuron.backward(gelu_output_grad, learning_rate)
            gelu_epoch_loss += gelu_mse * len(x_batch)
        
        # Calculate average epoch losses
        adaptive_epoch_loss /= len(x_train_normalized)
        gelu_epoch_loss /= len(x_train_normalized)
        
        adaptive_losses.append(adaptive_epoch_loss)
        gelu_losses.append(gelu_epoch_loss)
        
        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Adaptive Loss = {adaptive_epoch_loss:.6f}, GELU Loss = {gelu_epoch_loss:.6f}")
    
    # Visualization after training
    print("\nFinal activation functions:")
    adaptive_neuron.visualize_activation("Trained Adaptive Piecewise Activation")
    gelu_neuron.visualize_activation(title="GELU Activation Function (Unchanged)")
    
    # Plot training curves for comparison
    plt.figure(figsize=(12, 6))
    plt.plot(adaptive_losses, 'b-', label='Adaptive Piecewise')
    plt.plot(gelu_losses, 'r-', label='GELU')
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale to better see differences
    plt.show()
    
    # Compute final test metrics
    # Create a grid of test points for visualization
    x_test = np.linspace(-5, 5, 200).reshape(-1, 1)
    y_test_true = np.sin(x_test) * np.exp(-0.1 * x_test**2)
    
    # Normalize test data using training stats
    x_test_normalized = (x_test - x_mean) / x_std
    
    # Get predictions from both models
    adaptive_preds = adaptive_neuron.forward(x_test_normalized)
    gelu_preds = gelu_neuron.forward(x_test_normalized)
    
    # Denormalize predictions
    adaptive_preds_denorm = adaptive_preds * y_std + y_mean
    gelu_preds_denorm = gelu_preds * y_std + y_mean
    
    # Calculate MSE on test data
    adaptive_test_mse = np.mean((adaptive_preds_denorm - y_test_true.ravel())**2)
    gelu_test_mse = np.mean((gelu_preds_denorm - y_test_true.ravel())**2)
    
    print(f"\nTest MSE for Adaptive Piecewise: {adaptive_test_mse:.6f}")
    print(f"Test MSE for GELU: {gelu_test_mse:.6f}")
    
    # Visualize predictions
    plt.figure(figsize=(12, 8))
    plt.plot(x_test, y_test_true, 'k-', linewidth=2, label='True Function')
    plt.plot(x_test, adaptive_preds_denorm, 'b-', label=f'Adaptive Piecewise (MSE={adaptive_test_mse:.6f})')
    plt.plot(x_test, gelu_preds_denorm, 'r-', label=f'GELU (MSE={gelu_test_mse:.6f})')
    plt.title("Model Predictions Comparison")
    plt.xlabel("Input x")
    plt.ylabel("Output y")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Return final metrics as dictionary
    return {
        'adaptive_final_loss': adaptive_losses[-1],
        'gelu_final_loss': gelu_losses[-1],
        'adaptive_test_mse': adaptive_test_mse,
        'gelu_test_mse': gelu_test_mse,
        'improvement_percentage': ((gelu_test_mse - adaptive_test_mse) / gelu_test_mse) * 100
    }


# Optional: Add another activation function comparison with tanh as smoothing
def modify_with_tanh_smoothing():
    """
    This function would modify the AdaptivePiecewiseActivation class 
    to use tanh instead of sigmoid for smoothing transitions between pieces.
    
    Implementation:
    1. Replace sigmoid with tanh in _get_piece_weights
    2. Adjust smoothing_factor as tanh has different curvature
    3. Update gradient calculations accordingly
    """
    print("To implement tanh smoothing:")
    print("1. Replace sigmoid with tanh in _get_piece_weights method")
    print("2. In _get_piece_weights, use: transition = np.tanh(self.smoothing_factor * (x - boundary))")
    print("3. Update derivative calculations in get_derivative (derivative of tanh(x) = 1 - tanh^2(x))")
    print("4. Adjust smoothing_factor (typically needs to be higher with tanh)")


# Run the comparison
if __name__ == "__main__":
    print("Comparing Adaptive Piecewise Activation with GELU...")
    results = compare_activations(reg_lambda=0.01)
    
    print("\nFinal Results Summary:")
    print(f"Adaptive Piecewise Final Loss: {results['adaptive_final_loss']:.6f}")
    print(f"GELU Final Loss: {results['gelu_final_loss']:.6f}")
    print(f"Adaptive Piecewise Test MSE: {results['adaptive_test_mse']:.6f}")
    print(f"GELU Test MSE: {results['gelu_test_mse']:.6f}")
    print(f"Improvement from using Adaptive: {results['improvement_percentage']:.2f}%")
    
    if results['improvement_percentage'] > 0:
        print("\nThe Adaptive Piecewise Activation performed better than GELU!")
    else:
        print("\nGELU performed better than the Adaptive Piecewise Activation.")
        print("Consider using tanh for smoothing instead of sigmoid.")
        modify_with_tanh_smoothing()
