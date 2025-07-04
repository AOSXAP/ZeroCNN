'''
Flattens the input into a 1D vector.
'''
class FlattenLayer:
    def __init__(self, input):
        self.input = input
        self.input_height = len(input)
        self.input_width = len(input[0])

    def forward(self):
        return [item for sublist in self.input for item in sublist]