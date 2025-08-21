Adaptive Piecewise Activation Functions

This project implements a learnable piecewise polynomial activation function that adapts during training, along with a neuron wrapper that integrates this adaptive activation.
Unlike fixed nonlinearities (ReLU, tanh, GELU), this activation function evolves to better approximate complex nonlinear mappings.


Features

AdaptivePiecewiseActivation

Splits the input space into learnable regions (pieces).

Each region uses a polynomial function with trainable coefficients.

Boundaries between regions are learnable and smoothed via sigmoid transitions.

Includes gradient clipping, coefficient regularization, and boundary stabilization for numerical stability.

Tracks history of boundaries and coefficients for visualization.

NeuronWithAdaptiveActivation

Single neuron model with learnable weights, bias, and adaptive activation.

Supports forward and backward passes with full gradient updates.

Includes visualization tools for:

Weight vs activation function updates.

Regularization losses.

Activation evolution across training.

Training Example

Trains a neuron with adaptive activation on a synthetic target function.

Demonstrates how the activation function evolves to fit nonlinear patterns.

Provides plots for activation function updates, training loss, and regularization effects.


Requirements

Install dependencies with:

pip install numpy matplotlib scipy


Usage

Run the example training script:

python adaptive_piecewise_activation.py


During execution, you will see:

Initial vs. trained activation functions.

Evolution of the activation function across epochs.

Training curves (MSE + regularization loss).
