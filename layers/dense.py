import random
import math
from layers.layer import Layer

'''
A fully connected layer.

input_size: the size of the input
output_size: the size of the output
'''
class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        limit = math.sqrt(2.0 / input_size)  # He initialization variant
        self.weights = [[random.uniform(-limit, limit) for _ in range(self.input_size)] for _ in range(self.output_size)]
        self.bias = [0.01 for _ in range(self.output_size)]  # Small positive bias

    def forward(self, input):
        self.input = input
        output = []

        for i in range(self.output_size):
            parameter = 0

            # o[i] = sum(i[j] * w[i][j] for j in range(self.input_size)) + b[i]
            for j in range(self.input_size):
                parameter += self.input[j] * self.weights[i][j]

            parameter += self.bias[i]

            output.append(parameter)

        return output

    def backward(self, grad_output, learning_rate=0.01):
        # Compute gradients for input
        grad_input = [0] * self.input_size
        for i in range(self.output_size):
            for j in range(self.input_size):
                grad_input[j] += grad_output[i] * self.weights[i][j]
        
        # Update weights and biases
        for i in range(self.output_size):
            # Update bias
            self.bias[i] -= learning_rate * grad_output[i]
            
            # Update weights
            for j in range(self.input_size):
                self.weights[i][j] -= learning_rate * grad_output[i] * self.input[j]
        
        return grad_input