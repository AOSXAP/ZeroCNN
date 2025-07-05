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
        conv2d_layer = Convolutional2DLayer([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1,2], [2,1]], 1, 0)
        if(conv2d_layer.forward() != [[18, 24], [36, 42]]):
            raise Exception("Convolutional2D layer forward test failed")
        if(conv2d_layer.backward([[18, 24], [36, 42]]) != [[1, 2, 3], [4, 5, 6], [7, 8, 9]]):
            raise Exception("Convolutional2D layer backward test failed")

    def test_maxpool2d(self):
        maxpool2d_layer = MaxPool2DLayer([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2, 1)
        if(maxpool2d_layer.forward() != [[5, 6], [8, 9]]):
            raise Exception("MaxPool2D layer forward test failed")
        if(maxpool2d_layer.backward([[5, 6], [8, 9]]) != [[0, 0, 0], [0, 5, 6], [0, 8, 9]]):
            raise Exception("MaxPool2D layer backward test failed")

    def test_flatten(self):
        flatten_layer = FlattenLayer([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        if(flatten_layer.forward() != [1, 2, 3, 4, 5, 6, 7, 8, 9]):
            raise Exception("Flatten layer forward test failed")
        if(flatten_layer.backward([1, 2, 3, 4, 5, 6, 7, 8, 9]) != [[1, 2, 3], [4, 5, 6], [7, 8, 9]]):
            raise Exception("Flatten layer backward test failed")

    def test_dense(self):
        dense_layer = DenseLayer([1, 2, 3], 2)
        forward_output = dense_layer.forward()

    def test_softmax(self):
        softmax_layer = SoftmaxLayer()
        if(softmax_layer.forward([1, 2, 3]) != [0.09003057317038046, 0.24472847105479764, 0.6652409557748218]):
            raise Exception("Softmax layer forward test failed")



if __name__ == "__main__":
    test = LayersTest()
    test.run_all_tests()