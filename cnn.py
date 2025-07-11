from layers.layer import Layer
from layers.conv2d import Convolutional2DLayer
from layers.maxpool2d import MaxPool2DLayer
from layers.flatten import FlattenLayer
from layers.dense import DenseLayer
from layers.softmax import SoftmaxLayer
from utils.maths import relu_matrix
import math

class ZeroCNN:
    def __init__(self):
        self.layers = []  

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, input):
        current_input = input
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

    def backward(self, grad_output):
        # Implement backward pass through all layers
        current_grad = grad_output
        for layer in reversed(self.layers):
            current_grad = layer.backward(current_grad)
        return current_grad

    def predict(self, input):
        """Make a prediction and return the predicted class"""
        output = self.forward(input)
        return output.index(max(output))

    def train_step(self, input, target, learning_rate=0.01):
        """Perform one training step"""
        # Forward pass
        output = self.forward(input)
        
        # Calculate loss (cross-entropy)
        loss = -math.log(output[target] + 1e-8)  # Add small epsilon to prevent log(0)
        
        # Create one-hot encoded target
        target_one_hot = [0] * len(output)
        target_one_hot[target] = 1
        
        # Backward pass
        grad_output = target_one_hot
        self.backward(grad_output)
        
        return loss

def create_mnist_cnn():
    """Create a CNN architecture suitable for MNIST classification"""
    cnn = ZeroCNN()
    
    # First convolutional layer with edge detection kernel
    # MNIST images are 28x28, so we use a 3x3 kernel
    conv1_kernel = [
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ]
    cnn.add_layer(Convolutional2DLayer(conv1_kernel, stride=1, padding=1))
    
    # First max pooling layer (reduces 28x28 to 14x14)
    cnn.add_layer(MaxPool2DLayer(pool_size=2, stride=2))
    
    # Second convolutional layer with different kernel
    conv2_kernel = [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]
    cnn.add_layer(Convolutional2DLayer(conv2_kernel, stride=1, padding=1))
    
    # Second max pooling layer (reduces 14x14 to 7x7)
    cnn.add_layer(MaxPool2DLayer(pool_size=2, stride=2))
    
    # Flatten layer to convert 7x7 feature map to 1D vector
    cnn.add_layer(FlattenLayer())
    
    # Dense layer (fully connected) - 7x7 = 49 inputs to 128 hidden units
    cnn.add_layer(DenseLayer(49, 128))
    
    # Output layer - 128 inputs to 10 outputs (for 10 digit classes)
    cnn.add_layer(DenseLayer(128, 10))
    
    # Softmax layer for probability distribution
    cnn.add_layer(SoftmaxLayer())
    
    return cnn

def test_cnn():
    """Test the CNN with a simple example"""
    # Create a simple 28x28 test image (all zeros with some pattern)
    test_image = [[0.0 for _ in range(28)] for _ in range(28)]
    
    # Add some pattern to make it interesting
    for i in range(10, 18):
        for j in range(10, 18):
            test_image[i][j] = 1.0
    
    # Create CNN
    cnn = create_mnist_cnn()
    
    # Test forward pass
    output = cnn.forward(test_image)
    print("CNN Output:", output)
    print("Predicted class:", cnn.predict(test_image))
    
    return cnn

if __name__ == "__main__":
    print("Testing CNN...")
    cnn = test_cnn()
    print("CNN created and tested successfully!")