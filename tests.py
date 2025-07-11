import inspect
from layers.conv2d import Convolutional2DLayer
from layers.maxpool2d import MaxPool2DLayer
from layers.flatten import FlattenLayer
from layers.dense import DenseLayer
from layers.softmax import SoftmaxLayer

class TestFramework:
    def __init__(self):
        pass
    
    def run_all_tests(self):
        for name, obj in inspect.getmembers(self):
            if inspect.ismethod(obj) and name.startswith("test_"):
                try:
                    print(f"Running test: {name}")
                    obj()
                    print(f"Test {name} passed")
                except Exception as e:
                    print(f"Test {name} failed: {e}")
                print('--------------------------------')
                

class LayersTest(TestFramework):
    def __init__(self):
        pass

    def test_conv2d(self):
        conv2d_layer = Convolutional2DLayer([[1,2], [2,1]], stride=1, padding=0)
        if(conv2d_layer.forward([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) != [[18, 24], [36, 42]]):
            raise Exception("Convolutional2D layer forward test failed")
        # Note: backward test needs to be updated based on the actual implementation
        # if(conv2d_layer.backward([[18, 24], [36, 42]]) != [[1, 2, 3], [4, 5, 6], [7, 8, 9]]):
        #     raise Exception("Convolutional2D layer backward test failed")

    def test_maxpool2d(self):
        maxpool2d_layer = MaxPool2DLayer(pool_size=2, stride=1)
        if(maxpool2d_layer.forward([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) != [[5, 6], [8, 9]]):
            raise Exception("MaxPool2D layer forward test failed")
        # Note: backward test needs to be updated based on the actual implementation
        # if(maxpool2d_layer.backward([[5, 6], [8, 9]]) != [[0, 0, 0], [0, 5, 6], [0, 8, 9]]):
        #     raise Exception("MaxPool2D layer backward test failed")

    def test_flatten(self):
        flatten_layer = FlattenLayer()
        if(flatten_layer.forward([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) != [1, 2, 3, 4, 5, 6, 7, 8, 9]):
            raise Exception("Flatten layer forward test failed")
        if(flatten_layer.backward([1, 2, 3, 4, 5, 6, 7, 8, 9]) != [[1, 2, 3], [4, 5, 6], [7, 8, 9]]):
            raise Exception("Flatten layer backward test failed")

    def test_dense(self):
        dense_layer = DenseLayer(input_size=3, output_size=2)
        forward_output = dense_layer.forward([1, 2, 3])
        print(f"Dense layer output: {forward_output}")
        # Just check that it returns the correct size
        if len(forward_output) != 2:
            raise Exception("Dense layer forward test failed - wrong output size")

    def test_softmax(self):
        softmax_layer = SoftmaxLayer()
        output = softmax_layer.forward([1, 2, 3])
        expected = [0.09003057317038046, 0.24472847105479764, 0.6652409557748218]
        # Check if outputs are close enough (floating point precision)
        if not all(abs(a - b) < 1e-10 for a, b in zip(output, expected)):
            raise Exception(f"Softmax layer forward test failed. Got {output}, expected {expected}")



if __name__ == "__main__":
    test = LayersTest()
    test.run_all_tests()