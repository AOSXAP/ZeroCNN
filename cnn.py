from layers.layer import Layer
from layers.conv2d import Convolutional2DLayer
from layers.maxpool2d import MaxPool2DLayer
from layers.flatten import FlattenLayer
from layers.dense import DenseLayer
from layers.softmax import SoftmaxLayer
from layers.dropout import DropoutLayer
from utils.maths import relu_matrix
from mnist import read_images, read_labels
import math
import random

class ZeroCNN:
    def __init__(self):
        self.layers = []  # Fixed: properly initialize as instance variable

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def set_training(self, training):
        """Set training mode for all layers"""
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(training)

    def forward(self, input):
        current_input = input
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

    def backward(self, grad_output, learning_rate=0.01):
        # Implement backward pass through all layers
        current_grad = grad_output
        for layer in reversed(self.layers):
            # Pass learning rate to Dense layers for weight updates
            if hasattr(layer, 'weights'):  # Dense layer
                current_grad = layer.backward(current_grad, learning_rate)
            else:  # Other layers
                current_grad = layer.backward(current_grad)
        return current_grad

    def predict(self, input):
        """Make a prediction and return the predicted class"""
        # Set to inference mode
        self.set_training(False)
        output = self.forward(input)
        return output.index(max(output))

    def train_step(self, input, target, learning_rate=0.01):
        """Perform one training step"""
        # Set to training mode
        self.set_training(True)
        
        # Forward pass
        output = self.forward(input)
        
        # Calculate loss (cross-entropy)
        loss = -math.log(output[target] + 1e-8)  # Add small epsilon to prevent log(0)
        
        # Create gradient for softmax layer (difference between predicted and true)
        grad_output = [0] * len(output)
        for i in range(len(output)):
            if i == target:
                grad_output[i] = output[i] - 1  # For true class
            else:
                grad_output[i] = output[i]      # For other classes
        
        # Backward pass with learning rate
        self.backward(grad_output, learning_rate)
        
        return loss

    def evaluate(self, test_images, test_labels):
        """Evaluate the model on test data"""
        # Set to inference mode
        self.set_training(False)
        
        correct = 0
        total = len(test_images)
        
        for i in range(total):
            prediction = self.predict(test_images[i])
            if prediction == test_labels[i]:
                correct += 1
        
        accuracy = correct / total
        return accuracy

def create_mnist_cnn():
    """Create a CNN architecture suitable for MNIST classification"""
    cnn = ZeroCNN()
    
    # First convolutional layer with edge detection kernel
    # MNIST images are 28x28, so we use a 3x3 kernel
    conv1_kernel = [
        [0.1, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.1]
    ]
    cnn.add_layer(Convolutional2DLayer(conv1_kernel, stride=1, padding=1))
    
    # First max pooling layer (reduces 28x28 to 14x14)
    cnn.add_layer(MaxPool2DLayer(pool_size=2, stride=2))
    
    # Second convolutional layer with different kernel
    conv2_kernel = [
        [0.2, 0.0, -0.2],
        [0.4, 0.0, -0.4],
        [0.2, 0.0, -0.2]
    ]
    cnn.add_layer(Convolutional2DLayer(conv2_kernel, stride=1, padding=1))
    
    # Second max pooling layer (reduces 14x14 to 7x7)
    cnn.add_layer(MaxPool2DLayer(pool_size=2, stride=2))
    
    # Flatten layer to convert 7x7 feature map to 1D vector
    cnn.add_layer(FlattenLayer())
    
    # Dense layer (fully connected) - 7x7 = 49 inputs to 64 hidden units
    cnn.add_layer(DenseLayer(49, 64))
    
    # Dropout layer to prevent overfitting
    cnn.add_layer(DropoutLayer(dropout_rate=0.5))
    
    # Output layer - 64 inputs to 10 outputs (for 10 digit classes)
    cnn.add_layer(DenseLayer(64, 10))
    
    # Dropout layer before final classification (lighter dropout)
    cnn.add_layer(DropoutLayer(dropout_rate=0.3))
    
    # Softmax layer for probability distribution
    cnn.add_layer(SoftmaxLayer())
    
    return cnn

def train_mnist_cnn():
    """Train the CNN on MNIST data"""
    print("Loading MNIST data...")
    train_images = read_images('data_mnist/train-images.idx3-ubyte')
    train_labels = read_labels('data_mnist/train-labels.idx1-ubyte')
    test_images = read_images('data_mnist/t10k-images.idx3-ubyte')
    test_labels = read_labels('data_mnist/t10k-labels.idx1-ubyte')
    
    print(f"Training images: {len(train_images)}")
    print(f"Test images: {len(test_images)}")
    
    # Create CNN
    cnn = create_mnist_cnn()
    
    # Training parameters
    epochs = 10 
    batch_size = 100  
    learning_rate = 0.01 
    
    # Early stopping parameters
    best_accuracy = 0.0
    patience = 3
    patience_counter = 0
    
    print("Starting training...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Shuffle training data
        combined = list(zip(train_images, train_labels))
        random.shuffle(combined)
        train_images_shuffled, train_labels_shuffled = zip(*combined)
        
        total_loss = 0
        num_batches = len(train_images_shuffled) // batch_size
        
        for batch_idx in range(min(20, num_batches)):  # Limit batches for reasonable training time
            batch_loss = 0
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            for i in range(start_idx, end_idx):
                loss = cnn.train_step(train_images_shuffled[i], train_labels_shuffled[i], learning_rate)
                batch_loss += loss
            
            avg_batch_loss = batch_loss / batch_size
            total_loss += avg_batch_loss
            
            if batch_idx % 5 == 0:  # Print every 5 batches
                print(f"Batch {batch_idx + 1}/{min(20, num_batches)}, Loss: {avg_batch_loss:.4f}")
        
        # Evaluate
        accuracy = cnn.evaluate(test_images, test_labels)
        
        print(f"Epoch {epoch + 1} - Test accuracy: {accuracy:.4f}")
        
        # Early stopping logic
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            print(f"New best accuracy: {best_accuracy:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
        
        # Reduce learning rate if no improvement
        if patience_counter >= 2:
            learning_rate *= 0.5
            print(f"Reducing learning rate to: {learning_rate:.4f}")
    
    print(f"\nTraining completed! Best accuracy: {best_accuracy:.4f}")
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
    print("ZeroCNN - A CNN implementation from scratch")
    print("=" * 50)
    
    choice = input("Choose an option:\n1. Test CNN with dummy data\n2. Train CNN on MNIST data\nEnter choice (1 or 2): ")
    
    if choice == "1":
        print("\nTesting CNN with dummy data...")
        cnn = test_cnn()
        print("CNN created and tested successfully!")
    elif choice == "2":
        print("\nTraining CNN on MNIST data...")
        cnn = train_mnist_cnn()
        print("Training completed!")
    else:
        print("Invalid choice. Running test by default...")
        cnn = test_cnn()
        print("CNN created and tested successfully!")