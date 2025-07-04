# ZeroCNN - A CNN from scratch

## Project Structure

- `layers/`: Contains the implementation of each layer.
- `utils/`: Contains utility functions.

## Roadmap

1. **Convolutional2DLayer**
   - Implement 2D convolution operation (with support for stride and padding).
   - Handle multiple input and output channels (filters).
   - Include forward and backward propagation.

2. **MaxPool2DLayer**
   - Implement 2D max pooling operation (with configurable pool size and stride).
   - Support forward and backward passes.

3. **FlattenLayer**
   - Implement a layer to flatten multi-dimensional input into a 1D vector.
   - Ensure compatibility with both forward and backward passes.

4. **DenseLayer**
   - Implement a fully connected (dense) layer.
   - Support arbitrary input and output sizes.
   - Include forward and backward propagation.

5. **SoftmaxLayer**
   - Implement the softmax activation for output classification.
   - Include cross-entropy loss calculation and gradient computation.