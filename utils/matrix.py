def get_input_region(input, i, j, size):
    region = []
    for row in range(i, i+size):
        region.append([])
        for col in range(j, j+size):
            region[row-i].append(input[row][col])
    return region

def matrix_to_vector(matrix):
    vector = []
    for row in matrix:
        for col in row:
            vector.append(col)
    return vector

def main():
    input = [[1,2,3,4,5],
             [6,7,8,9,10],
             [11,12,13,14,15],
             [16,17,18,19,20],
             [21,22,23,24,25]]
    print(get_input_region(input, 2, 2, 3))

if __name__ == "__main__":
    main()