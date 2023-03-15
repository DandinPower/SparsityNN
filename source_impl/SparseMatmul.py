import numpy as np

class CRS_Matrix:
    def __init__(self, rowPtrs, col_indices, data, shape):
        self.rowPtrs = rowPtrs
        self.col_indices = col_indices
        self.data = data
        self.shape = shape

def _csr_matmul(mat1, mat2):
    c = np.zeros((mat1.shape[0], mat2.shape[1]), dtype=int)
    for row in range(mat1.shape[0]):
        for col in range(mat2.shape[1]):
            sum = 0
            for k in range(mat1.rowPtrs[row], mat1.rowPtrs[row + 1]):
                mat1_col = mat1.col_indices[k]
                for j in range(mat2.rowPtrs[mat1_col], mat2.rowPtrs[mat1_col + 1] -1):
                    print(mat1.data[k], mat2.data[j])
                    '''
                    if mat2.col_indices[j] == col:
                        sum += mat1.data[k] * mat2.data[j]
                        break 
                    '''
            if sum != 0:
                c[row][col] = sum 
    return c

if __name__ == "__main__":
    a = CRS_Matrix([0,2,2,3], [0,2,1], [1,2,4], [3,3])
    b = CRS_Matrix([0,1,2,4], [1,0,2], [3,1,2], [3,2])
    c = _csr_matmul(a, b)
    print(c)  