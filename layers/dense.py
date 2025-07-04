import random

class DenseLayer:
    def __init__(self, input, output_size):
        self.input = input
        self.input_size = len(input)
        self.output_size = output_size
        self.output = [0] * self.output_size

        random.seed(1)
        self.weights = [[random.random() for _ in range(self.input_size)] for _ in range(self.output_size)]
        self.bias = [random.random() for _ in range(self.output_size)]

def forward(self):
    output = []

    for i in range(self.output_size):
        parameter = 0

        # o[i] = sum(i[j] * w[i][j] for j in range(self.input_size)) + b[i]
        for j in range(self.input_size):
            parameter += self.input[j] * self.weights[i][j]

        parameter += self.bias[i]

        output.append(parameter)

    return output
