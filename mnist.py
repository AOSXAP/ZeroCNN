import struct

def read_images(filename):
    with open(filename, 'rb') as f:
        # >IIII: big endian, 4 ints
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = []
        for _ in range(num):
            img = list(f.read(rows * cols))
            # Normalize pixel values to [0, 1]
            img = [pixel / 255.0 for pixel in img]
            # Reshape to 2D
            img = [img[i * cols:(i + 1) * cols] for i in range(rows)]
            images.append(img)
        return images

def read_labels(filename):
    with open(filename, 'rb') as f:
        # >II: big endian, 2 ints
        magic, num = struct.unpack(">II", f.read(8))
        labels = list(f.read(num))
        return labels

def main():
    train_images = read_images('data_mnist/train-images.idx3-ubyte')
    train_labels = read_labels('data_mnist/train-labels.idx1-ubyte')
    test_images = read_images('data_mnist/t10k-images.idx3-ubyte')
    test_labels = read_labels('data_mnist/t10k-labels.idx1-ubyte')

    print(len(train_images))
    print(len(train_labels))
    print(len(test_images))
    print(len(test_labels))

if __name__ == "__main__":
    main()
