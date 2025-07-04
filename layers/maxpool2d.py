from utils.matrix import get_input_region, matrix_to_vector

'''
Applies a max pooling operation to the input.

input_size: the size of the input
pool_size: the size of the pooling window
stride: the number of pixels the pooling window moves after each pooling
'''
class MaxPool2DLayer:
    def __init__(self, input, pool_size, stride):
        self.input = input
        self.input_height = len(input)
        self.input_width = len(input[0])

        self.pool_size = pool_size
        self.stride = stride

    def forward(self):
        output = []
        for i in range(0, self.input_height - self.pool_size + 1, self.stride):
            row = []
            for j in range(0, self.input_width - self.pool_size + 1, self.stride):
                input_region = get_input_region(self.input, i, j, self.pool_size)
                row.append(max(matrix_to_vector(input_region)))
            output.append(row)
        return output