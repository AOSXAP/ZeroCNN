def get_input_region(input, i, j, size):
    '''
    Gets a region of the input matrix.

    input: the input matrix
    i: the row index of the top-left corner of the region
    j: the column index of the top-left corner of the region
    size: the size of the region
    '''
    region = []
    for row in range(i, i+size):
        region.append([])
        for col in range(j, j+size):
            region[row-i].append(input[row][col])
    return region

def matrix_to_vector(matrix):
    '''
    Converts a matrix to a vector.

    matrix: the matrix to convert
    '''
    vector = []
    for row in matrix:
        for col in row:
            vector.append(col)
    return vector