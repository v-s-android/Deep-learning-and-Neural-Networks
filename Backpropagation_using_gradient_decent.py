"""
Objective for this notebook
Train a Neural Network to Solve the XOR Problem
Implement Backpropagation for Neural Network Training
Demonstrate the Use of Activation Functions
Understand the Learning Process Over Multiple Epochs
Demonstrate Weight and Bias Adjustments via Gradient Descent
Evaluate the Model's Performance After Training
Monitor and Analyze the Training Process

#!pip install numpy==1.26.4
#!pip install matplotlib==3.5.2
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Initialize Inputs
Define the input and expected output for a XOR gate problem
"""

# Defining inputs and expected output (XOR truth table)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # 2x4 matrix, each column is a training example
d = np.array([0, 1, 1, 0])  # Expected output for XOR

"""
Declare the network parameters & the weights
Define the network parameters such as

number of input neurons
hidden layer neurons
output neurons
learning rate
number of epochs
Also, Declare the weights for the neurons. The initial weights are taken as random numbers which are then optimized by the backpropagation
algorithm inside a function parameter initialize_network_parameters()
"""
