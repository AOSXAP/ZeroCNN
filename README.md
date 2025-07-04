Input (e.g., 28x28)
↓
Conv2D (3x3 kernel, stride=1, padding=1, 8 filters)
↓
ReLU
↓
MaxPool2D (2x2, stride=2)
↓
Flatten
↓
Dense (128 units)
↓
ReLU
↓
Dense (10 units)
↓
Softmax (for classification)
