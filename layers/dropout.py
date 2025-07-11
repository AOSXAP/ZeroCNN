import random
from layers.layer import Layer

'''
Dropout layer for regularization.
Randomly sets a fraction of input units to 0 during training.
'''
class DropoutLayer(Layer):
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.training = True  # Flag to distinguish training vs inference
        self.mask = None
    
    def set_training(self, training):
        """Set whether the layer is in training mode"""
        self.training = training
    
    def forward(self, input):
        self.input = input
        
        if self.training:
            # Create dropout mask
            self.mask = [1 if random.random() > self.dropout_rate else 0 for _ in input]
            # Apply mask and scale by 1/(1-dropout_rate) to maintain expected output
            scale = 1.0 / (1.0 - self.dropout_rate)
            output = [x * mask * scale for x, mask in zip(input, self.mask)]
        else:
            # During inference, use all neurons
            output = input
        
        return output
    
    def backward(self, grad_output):
        if self.training and self.mask:
            # Apply the same mask to gradients
            scale = 1.0 / (1.0 - self.dropout_rate)
            grad_input = [grad * mask * scale for grad, mask in zip(grad_output, self.mask)]
        else:
            grad_input = grad_output
        
        return grad_input 