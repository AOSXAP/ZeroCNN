import random
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
        
        # Initialize weights and biases
        self.weights = [[random.uniform(-0.5, 0.5) for _ in range(self.input_size)] for _ in range(self.output_size)]
        self.bias = [random.uniform(-0.5, 0.5) for _ in range(self.output_size)]

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

    def backward(self, grad_output):
        grad_input = [0] * self.input_size
        for i in range(self.output_size):
            for j in range(self.input_size):
                grad_input[j] += grad_output[i] * self.weights[i][j]
        return grad_input