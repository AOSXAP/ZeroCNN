from utils.matrix import get_input_region, matrix_to_vector
from utils.maths import dot_product   

'''
Applies filters (kernels) to the input image or feature map.
Each filter slides (convolves) across the input, computing dot products between the filter and local regions.
Produces an output feature map that highlights certain patterns (e.g., edges, textures).

input_size: the size of the input
kernel_size: the size of the filter
stride: the number of pixels the filter moves after each convolution
padding: the number of pixels added to the edges of the input to maintain the output size
'''
class Convolutional2DLayer:
    def __init__(self, input, kernel, stride, padding):
        self.input = input
        self.input_size = len(input)

        self.kernel = kernel
        self.kernel_size = len(kernel)
        self.kernel_vector = matrix_to_vector(kernel)

        self.stride = stride
        self.padding = padding

        self.apply_padding()
        self.input_size = len(self.input)

    def forward(self):
        output = []
        for i in range(0, self.input_size - self.kernel_size + 1, self.stride):
            row = []
            for j in range(0, self.input_size - self.kernel_size + 1, self.stride):
                input_region = get_input_region(self.input, i, j, self.kernel_size)
                input_region_vector = matrix_to_vector(input_region)
                row.append(dot_product(input_region_vector, self.kernel_vector))
            output.append(row)
        return output

    def apply_padding(self):
        padded_input = []
        for i in range(self.padding):
            padded_input.append([0] * (self.input_size + 2 * self.padding))
        for row in self.input:
            padded_input.append([0] * self.padding + row + [0] * self.padding)
        for i in range(self.padding):
            padded_input.append([0] * (self.input_size + 2 * self.padding))
        self.input = padded_input