from layers.layer import Layer

class ZeroCNN:
    def __init__(self):
        layers: list[Layer] = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, input):
        pass