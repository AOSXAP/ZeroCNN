# ZeroCNN - A CNN Implementation from Scratch

A complete Convolutional Neural Network built from scratch in Python. No external ML libraries used.
*Tested on MNIST dataset, managed to reach 88% accuracy on test set.*

## Features

- **Pure Python**: No TensorFlow, PyTorch, or NumPy
- **Complete CNN Pipeline**: Conv → ReLU → Pool → Dense → Dropout → Softmax
- **Training Features**: Early stopping, learning rate decay, regularization
- **Educational Focus**: Clear, readable code for learning

## Architecture

1. **Conv Layer 1** (3x3) + ReLU + MaxPool (28x28 → 14x14)
2. **Conv Layer 2** (3x3) + ReLU + MaxPool (14x14 → 7x7)
3. **Flatten** → **Dense** (49→64) + ReLU + Dropout (30%)
4. **Dense** (64→32) + ReLU + Dropout (20%)
5. **Dense** (32→10) + **Softmax**

## Usage

```bash
python cnn.py
# Choose option 1: Test with dummy data
# Choose option 2: Train on MNIST data

python tests.py  # Run layer tests
```

## Files

```
ZeroCNN/
├── cnn.py              # Main CNN and training
├── layers/             # Layer implementations
│   ├── conv2d.py       # Convolutional layer
│   ├── maxpool2d.py    # Max pooling layer
│   ├── dense.py        # Fully connected layer
│   ├── relu.py         # ReLU activation
│   ├── dropout.py      # Dropout regularization
│   └── softmax.py      # Softmax activation
├── utils/              # Matrix and math utilities
├── mnist.py            # MNIST data loading
└── data_mnist/         # MNIST dataset files
```

## MNIST Data
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

## Key Features

- **Proper Gradient Flow**: Backpropagation through all layers
- **Weight Updates**: Gradient descent with He initialization
- **Regularization**: Dropout prevents overfitting
- **Training Control**: Early stopping and learning rate decay

## Educational Purpose

Learn how CNNs work at a fundamental level:
- Convolution and pooling operations
- Forward/backward propagation
- Weight updates and optimization
- Regularization techniques

---

*Built for learning and education*

