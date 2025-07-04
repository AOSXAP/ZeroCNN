kernels = [
    [[1, 0, -1], [1, 0, -1], [1, 0, -1]],   # detects vertical edges (Sobel)
    [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],   # detects horizontal edges
    [[0, 1, 0], [1, -4, 1], [0, 1, 0]]      # Laplacian
]
