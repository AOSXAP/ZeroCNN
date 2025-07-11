from utils.matrix import get_input_region, matrix_to_vector
from layers.layer import Layer

'''
Applies a max pooling operation to the input.

pool_size: the size of the pooling window
stride: the number of pixels the pooling window moves after each pooling
'''
class MaxPool2DLayer(Layer):
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        self.input_height = len(input)
        self.input_width = len(input[0])
        
        output = []
        for i in range(0, self.input_height - self.pool_size + 1, self.stride):
            row = []
            for j in range(0, self.input_width - self.pool_size + 1, self.stride):
                input_region = get_input_region(self.input, i, j, self.pool_size)
                row.append(max(matrix_to_vector(input_region)))
            output.append(row)
        return output

    def backward(self, grad_output):
        '''
        The gradient of the max pooling layer is kept for the first max value in the pooling window and 0 for the rest.
        '''
        grad_input = [[0 for _ in range(self.input_width)] for _ in range(self.input_height)]
        for i in range(0, self.input_height - self.pool_size + 1, self.stride):
            for j in range(0, self.input_width - self.pool_size + 1, self.stride):
                input_region = get_input_region(self.input, i, j, self.pool_size)
                max_value = max(matrix_to_vector(input_region))
                for n in range(self.pool_size):
                    for m in range(self.pool_size):
                        if input_region[n][m] == max_value:
                            grad_input[i + n][j + m] += grad_output[i // self.stride][j // self.stride]
                            break
        return grad_input