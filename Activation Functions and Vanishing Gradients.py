"""
Objective for this notebook
Define sigmoid activation function and its derivative
Define ReLu activation function and its derivative
Calculate the Sigmoid and ReLU activations and their derivatives for a range of input values
Plot the output values
"""

import numpy as np
import matplotlib.pyplot as plt

# create sigmoid function
def sigmoid(z):
  return 1 / (1 + np.exp(-z))


# define sigmoid derivative function
def sigmoid_derivative(z):
  return sigmoid(z) * (1 - sigmoid(z))

# define relu function
def relu(z):
  return np.maximum(0, z)
"""
Define the ReLU derivative (ReLU gradient)
Let's create a function relu_derivative to get the output of the Relu derivative (gradient) The function would take a value z
and give the corresponding relu derivative by returning the value 1 for all z greater than 0.
"""
def relu_derivative(z):
    return np.where(z > 0, 1, 0)

# Define the number of input values
#Generate 400 synthetic values between -10 and 10. The gradient would be calculated for each of these values.

# Generate a range of input values
z = np.linspace(-10, 10, 400)

# Next, let's compute the derivatives for Sigmoid and Relu activations and assign the values to sigmoid_grad and relu_grad respectively

sigmoid_grad = sigmoid_derivative(z)
relu_grad = relu_derivative(z)

# Awesome! So now, let's visualize the sigmoid and relu activations and their gradients by plotting their output values
# Plot the activation functions
plt.figure(figsize=(12, 6))

# Plot Sigmoid and its derivative
plt.subplot(1, 2, 1)
plt.plot(z, sigmoid(z), label='Sigmoid Activation', color='b')
plt.plot(z, sigmoid_grad, label="Sigmoid Derivative", color='r', linestyle='--')
plt.title('Sigmoid Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

# Plot ReLU and its derivative
plt.subplot(1, 2, 2)
plt.plot(z, relu(z), label='ReLU Activation', color='g')
plt.plot(z, relu_grad, label="ReLU Derivative", color='r', linestyle='--')
plt.title('ReLU Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

plt.tight_layout()
plt.show()

"""
Practice Exercise 1
In this practice exercise, let's try to create the hyperbolic tangent function and it's derivative.
"""
def tanh(z):
    return np.tanh(z)

def tan_derivative(z):
    return 1 - np.tanh(z) ** 2

"""
Practice Exercise 2¶
Now, let's plot and compare the output of tanH function with the ReLU function for 100 synthetic values between -5 to 5
"""

# Generate a range of input values
z = np.linspace(-5, 5, 100)

# Get the Relu and tanH gradient values
tanh_grad = tan_derivative(z)
relu_grad = relu_derivative(z)

# Plot the activation functions
plt.figure(figsize=(12, 6))

# Plot Relu and its derivative
plt.subplot(1, 2, 1)
plt.plot(z, relu(z), label='ReLU Activation', color='g')
plt.plot(z, relu_grad, label="ReLU Derivative", color='r', linestyle='--')
plt.title('ReLU Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()


# Plot tanH and its derivative
plt.subplot(1, 2, 2)
plt.plot(z, tanh(z), label='tanH Activation', color='g')
plt.plot(z, tanh_grad, label="tanH Derivative", color='r', linestyle='--')
plt.title('tanH Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

plt.tight_layout()
plt.show()


