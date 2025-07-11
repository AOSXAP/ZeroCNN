from layers.layer import Layer

'''
ReLU (Rectified Linear Unit) activation layer.
Applies the ReLU function: f(x) = max(0, x)
'''
class ReLULayer(Layer):
    def __init__(self):
        pass
    
    def forward(self, input):
        self.input = input
        # Apply ReLU: max(0, x) for each element
        if isinstance(input[0], list):  # 2D input (feature maps)
            output = []
            for row in input:
                output_row = []
                for val in row:
                    output_row.append(max(0, val))
                output.append(output_row)
        else:  # 1D input (vector)
            output = [max(0, val) for val in input]
        
        return output
    
    def backward(self, grad_output):
        # ReLU derivative: 1 if input > 0, else 0
        if isinstance(self.input[0], list):  # 2D input
            grad_input = []
            for i, row in enumerate(self.input):
                grad_row = []
                for j, val in enumerate(row):
                    if val > 0:
                        grad_row.append(grad_output[i][j])
                    else:
                        grad_row.append(0)
                grad_input.append(grad_row)
        else:  # 1D input
            grad_input = []
            for i, val in enumerate(self.input):
                if val > 0:
                    grad_input.append(grad_output[i])
                else:
                    grad_input.append(0)
        
        return grad_input 