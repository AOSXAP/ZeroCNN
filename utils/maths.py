def matrix_mult(A, B):
    """
    Multiplies two matrices.
    """
    C = []

    for i in range(len(A)):
        C.append([])
        for k in range(len(B[0])):
            sumx = 0
            for j in range(len(A[0])):
                sumx += A[i][j] * B[j][k]
            C[i].append(sumx)

    print(C)

def dot_product(A, B):
    """
    Calculates the dot product of two vectors.
    """
    return sum(a * b for a, b in zip(A, B))

def relu(x):
    """
    Applies the ReLU activation function to a value.
    """
    return max(0, x)

def relu_matrix(matrix):
    """
    Applies the ReLU activation function to a matrix.
    """
    return [[relu(x) for x in row] for row in matrix]