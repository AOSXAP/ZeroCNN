import math

'''
Applies the softmax function to the input.

input: the input to the softmax layer
'''
class SoftmaxLayer:
    def __init__(self, input):
        self.input = input

    def forward(self):
        total_sum = sum(math.exp(x) for x in self.input)
        return [math.exp(x) / total_sum for x in self.input]
