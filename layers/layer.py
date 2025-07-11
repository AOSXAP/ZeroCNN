'''
A layer is a building block of a neural network.
It takes an input and produces an output.
It also has a backward method that computes the gradient of the input with respect to the output.
'''

class Layer:
    def __init__(self):
        pass

    def forward(self, input):
        raise NotImplementedError("Subclasses must implement forward method")

    def backward(self, grad_output):
        raise NotImplementedError("Subclasses must implement backward method")