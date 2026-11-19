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

def initialize_network_parameters():
    # Network parameters
    inputSize = 2      # Number of input neurons (x1, x2)
    hiddenSize = 2     # Number of hidden neurons
    outputSize = 1     # Number of output neurons
    lr = 0.1           # Learning rate
    epochs = 180000    # Number of training epochs

    # Initialize weights and biases randomly within the range [-1, 1]
    w1 = np.random.rand(hiddenSize, inputSize) * 2 - 1  # Weights from input to hidden layer
    b1 = np.random.rand(hiddenSize, 1) * 2 - 1          # Bias for hidden layer
    w2 = np.random.rand(outputSize, hiddenSize) * 2 - 1 # Weights from hidden to output layer
    b2 = np.random.rand(outputSize, 1) * 2 - 1          # Bias for output layer

    return w1, b1, w2, b2, lr, epochs
"""
Training the Neural Network¶
The neural network works in 5 stages:

Forward pass

The input X is multiplied by the weights w1 and passed through the first layer, followed by the application of the sigmoid or ReLU activation function. This gives the output for the hidden layer.
The output of the hidden layer is then passed through the second set of weights w2 to compute the final output. Again, a sigmoid activation function is used to generate the final output a2.
Error calculation

The error is computed as the difference between the expected output (d) and the actual output (a2).
Backward pass

Output Layer: The derivative of the sigmoid activation function is applied to the error, producing the gradient for the output layer (da2). This is used to calculate how much the weights in the output layer need to be adjusted.
Hidden Layer: The error is then propagated backward to the hidden layer. The gradient at the hidden layer (da1) is computed by taking the dot product of the transpose of the weights (w2.T) and the gradient from the output layer. The derivative of the activation function (sigmoid or ReLU) is used to adjust this error.
Weights and bias updates

After computing the gradients (dz1, dz2), the weights (w1, w2) and biases (b1, b2) are updated using the learning rate (lr) and the gradients. The updates are done to minimize the error and improve the model’s predictions.
Training:

This entire process is repeated over many iterations (epochs). During each epoch, the model adjusts its weights and biases to reduce the error. Over time, the network learns to approximate the XOR function. Forward Pass:
"""

# Get initialized parameters
w1, b1, w2, b2, lr, epochs = initialize_network_parameters()

# Training the network using backpropagation
error_list = []
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(w1, X) + b1  # Weighted sum for hidden layer
    a1 = 1 / (1 + np.exp(-z1))  # Sigmoid activation for hidden layer

    z2 = np.dot(w2, a1) + b2  # Weighted sum for output layer
    a2 = 1 / (1 + np.exp(-z2))  # Sigmoid activation for output layer

    # Error calculation and backpropagation
    error = d - a2  # Difference between expected and actual output
    da2 = error * (a2 * (1 - a2))  # Derivative for output layer
    dz2 = da2  # Gradient for output layer

    # Propagate error to hidden layer
    da1 = np.dot(w2.T, dz2)  # Gradient for hidden layer
    dz1 = da1 * (a1 * (1 - a1))  # Derivative for hidden layer

    # Update weights and biases
    w2 += lr * np.dot(dz2, a1.T)  # Update weights from hidden to output layer
    b2 += lr * np.sum(dz2, axis=1, keepdims=True)  # Update bias for output layer

    w1 += lr * np.dot(dz1, X.T)  # Update weights from input to hidden layer
    b1 += lr * np.sum(dz1, axis=1, keepdims=True)  # Update bias for hidden layer
    if (epoch+1)%10000 == 0:
        print("Epoch: %d, Average error: %0.05f"%(epoch, np.average(abs(error))))
        error_list.append(np.average(abs(error)))

"""
Epoch: 9999, Average error: 0.06195
Epoch: 19999, Average error: 0.03225
Epoch: 29999, Average error: 0.02418
Epoch: 39999, Average error: 0.02008
Epoch: 49999, Average error: 0.01751
Epoch: 59999, Average error: 0.01572
Epoch: 69999, Average error: 0.01437
Epoch: 79999, Average error: 0.01331
Epoch: 89999, Average error: 0.01245
Epoch: 99999, Average error: 0.01174
Epoch: 109999, Average error: 0.01113
Epoch: 119999, Average error: 0.01061
Epoch: 129999, Average error: 0.01016
Epoch: 139999, Average error: 0.00975
Epoch: 149999, Average error: 0.00939
Epoch: 159999, Average error: 0.00907
Epoch: 169999, Average error: 0.00878
Epoch: 179999, Average error: 0.00851
"""

"""
Testing the Network
After training, you can now test the neural network to verify that it has learned the XOR function and outputs the correct values close to [0, 1, 1, 0]
"""
# Testing the trained network
z1 = np.dot(w1, X) + b1  # Weighted sum for hidden layer
a1 = 1 / (1 + np.exp(-z1))  # Sigmoid activation for hidden layer

z2 = np.dot(w2, a1) + b2  # Weighted sum for output layer
a2 = 1 / (1 + np.exp(-z2))  # Sigmoid activation for output layer

# Print results
print('Final output after training:', a2)
print('Ground truth', d)
print('Error after training:', error)
print('Average error: %0.05f'%np.average(abs(error)))

"""
Final output after training: [[0.00785585 0.99109928 0.99109823 0.00831761]]
Ground truth [0 1 1 0]
Error after training: [[-0.00785587  0.00890075  0.0089018  -0.00831764]]
Average error: 0.00849
"""

"""
Plot the error
Here, we plot the error as a function of epochs. This shows how error changed over multiple iterations of forward and backward passes and how the network learnt over time
"""
# Plot error
plt.plot(error_list)
plt.title('Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

"""
Practice exercise 1
Implement backpropagation for AND problem using similar input as used for XOR problem above and plot the error
"""

# Write your code here
X = np.array([[0,0],[0,1],[1,0],[1,1]]).T # 2x4 matrix, it is transpose
d = np.array([0, 0, 0, 1]) # expected output for AND

# Initialize the network
w1 ,b1, w2, b2 , lr, epochs = initialize_network_parameters()

error_list = [] # for plotting the graph

#Training the network using backpropagation
for epoch in range(epochs):
    # forward propagation
    z1 = np.dot(w1, X) + b1 # weighted sum for hidden layer
    a1 = 1 / (1 + np.exp(-z1)) # sigmoid activation for hidden layer

    z2 = np.dot(w2, a1) + b2 # weighted sum for output layer
    a2 = 1 / (1 + np.exp(-z2)) # sigmoid activation for outputlayer

    # Error calculation
    error = d - a2 # difference between expected and actual ouput
    der_a2 = error * (a2 * (1 - a2)) # Derivative for output layer
    der_z2 = der_a2 # Gradient for output layer

    # Backpropagation - propagate error to hidden layer
    der_a1 = np.dot(w2.T , der_z2) # Gradient for hidden layer
    der_z1 = der_a1 * (a1 * (1 - a1)) # Derivative of hidden layer

    # Update weights and biases
    w2 += lr * np.dot(der_z2, a1.T)  # Update weights from hidden to output layer
    b2 += lr * np.sum(der_z2, axis=1, keepdims=True)  # Update bias for output layer

    w1 += lr * np.dot(der_z1, X.T)  # Update weights from input to hidden layer
    b1 += lr * np.sum(der_z1, axis=1, keepdims=True)  # Update bias for hidden layer
    if (epoch+1)%10000 == 0:
        print("Epoch: %d, Average error: %0.05f"%(epoch, np.average(abs(error))))
        error_list.append(np.average(abs(error)))

# Testing the trained network
z1 = np.dot(w1, X) + b1  # Weighted sum for hidden layer
a1 = 1 / (1 + np.exp(-z1))  # Sigmoid activation for hidden layer

z2 = np.dot(w2, a1) + b2  # Weighted sum for output layer
a2 = 1 / (1 + np.exp(-z2))  # Sigmoid activation for output layer

# Print results
print('\nFinal output after training:', a2)
print('Ground truth', d)
print('Error after training:', error)
print('Average error: %0.05f'%np.average(abs(error)))

# Plot error
plt.plot(error_list)
plt.title('Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()
