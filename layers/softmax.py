import math
from layers.layer import Layer

'''
Applies the softmax function to the input.
'''
class SoftmaxLayer(Layer):
    def forward(self, input):
        self.input = input
        max_input = max(input)
        exps = [math.exp(x - max_input) for x in input]
        total_sum = sum(exps)
        
        self.output = [e / total_sum for e in exps]
        
        return self.output

    def backward(self, grad_output):
        # For classification, grad_output is typically the difference between predicted and true labels
        # For simplicity, we'll just pass the gradient through
        return grad_output