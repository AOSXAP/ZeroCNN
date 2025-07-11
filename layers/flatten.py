from layers.layer import Layer

'''
Flattens the input into a 1D vector.
'''
class FlattenLayer(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        self.input_height = len(input)
        self.input_width = len(input[0])
        
        self.output = [item for sublist in input for item in sublist]
        return self.output

    def backward(self, grad_output):
        '''
        grad_output is a 1D list; reshape it to (input_height, input_width)
        '''
        return [
            grad_output[i * self.input_width : (i + 1) * self.input_width]
            for i in range(self.input_height)
        ]