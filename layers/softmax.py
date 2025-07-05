import math

'''
Applies the softmax function to the input.
'''
class SoftmaxLayer(Layer):
    def forward(self, input):
        max_input = max(input)
        exps = [math.exp(x - max_input) for x in input]
        total_sum = sum(exps)
        
        self.output = [e / total_sum for e in exps]
        
        return self.output

    def backward(self, y_true):
        return [o - t for o, t in zip(self.output, y_true)]

if __name__ == "__main__":
    softmax_layer = SoftmaxLayer()
    print(softmax_layer.forward([1, 2, 3]))
    print(softmax_layer.backward([0, 0, 1]))