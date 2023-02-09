import numpy as np
import matplotlib.pyplot as plt

def ShowMatrix(matrix, path):
    if len(matrix.shape) == 1:
        matrix = np.reshape(matrix, (-1, matrix.shape[0]))
    matrix = np.absolute(matrix)
    plt.matshow(matrix, cmap=plt.cm.gray)
    plt.colorbar()
    plt.savefig(path)
    plt.clf()