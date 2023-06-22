import numpy as np

class Dense_Matrix:
    def __init__(self, data):
        self.data = data 
    
    def __repr__(self):
        output = '['
        for index, row in enumerate(self.data):
            if index != 0: output += '\n '
            output += f'{row},'
        output = output[:-1]
        output += ']'
        return output

    def __mul__(self, other):
        myData = np.array(self.data)
        otherData = np.array(other.data)
        answerData = np.matmul(myData, otherData)
        return Dense_Matrix(answerData)

class CSR_Matrix:
    def __init__(self, rowPtrs, colIndices, data, shape):
        self.rowPtrs = rowPtrs
        self.colIndices = colIndices
        self.data = data
        self.shape = shape

    def __repr__(self):
        return f'rowPtrs: {self.rowPtrs}, \ncolIndices: {self.colIndices}, \ndata: {self.data}, \nshape: {self.shape}'

    def to_dense(self):
        nRows = self.shape[0]
        nCols = self.shape[1]
        dense_matrix = [[0] * nCols for _ in range(nRows)]

        for i in range(nRows):
            for j in range(self.rowPtrs[i], self.rowPtrs[i + 1]):
                col = self.colIndices[j]
                value = self.data[j]
                dense_matrix[i][col] = value
        dense_matrix = Dense_Matrix(dense_matrix)
        return dense_matrix
       
def _csr_matmult_maxnnz(nRows: int, nCols: int, Ap: list[int], Aj: list[int], Bp: list[int], Bj: list[int]) -> int:
    '''
    implement python version to calculate the number of non-zero value of result matrix 
    '''
    mask = np.full((nCols,), -1)
    nnz = 0
    for i in range(nRows):
        row_nnz = 0
        for jj in range(Ap[i], Ap[i + 1]):
            j = Aj[jj]
            for kk in range(Bp[j], Bp[j + 1]):
                k = Bj[kk]
                if mask[k] != i:
                    mask[k] = i
                    row_nnz += 1
        nnz += row_nnz
    return nnz

def _csr_matmul(nRows: int, nCols: int, A: CSR_Matrix, B: CSR_Matrix, ans: CSR_Matrix) -> None:
    '''
    The actual _csr_matmult function takes in the row pointers, column indices, and non-zero values 
    for both input matrices in CSR format, as well as preallocated arrays for the row pointers, column indices, 
    and non-zero values of the output matrix.

    The main steps of the algorithm in the actual implementation are as follows:

    1. Initialize variables next and sums to store intermediate values during the computation.
    2. Iterate over the rows of the first matrix, A.
    3. For each row, iterate over its non-zero values, and for each non-zero value, iterate over the corresponding row in the second matrix, B.
    4. Update the sums for each column in the resulting matrix, C.
    5. If a new non-zero value is encountered in the resulting matrix, update the next array to keep track of non-zero values in the current row.
    6. After processing a row in matrix A, store the non-zero values and their column indices in the output matrix, C. 
    The column indices are sorted before being stored, as they are not guaranteed to be sorted in the linked list used during the computation.
    7. Update the row pointers for the output matrix, C.
    8. This implementation is more optimized than the simplified version I provided earlier, and it leverages linked lists, sorting, 
    and other techniques to achieve better performance during the matrix multiplication.
    '''
    next = np.full(nCols, -1, dtype=int)
    sums = np.zeros(nCols, dtype=int)
    nnz = 0
    ans.rowPtrs[0] = 0
    for i in range(nRows):
        head = -2
        length = 0
        for jj in range(A.rowPtrs[i], A.rowPtrs[i + 1]):
            j = A.colIndices[jj]
            v = A.data[jj]
            for kk in range(B.rowPtrs[j], B.rowPtrs[j + 1]):
                k = B.colIndices[kk]
                sums[k] += v * B.data[kk]
                if next[k] == -1:
                    next[k] = head
                    head = k
                    length += 1
        for _ in range(length):
            ans.colIndices[nnz] = head
            ans.data[nnz] = sums[head]
            nnz += 1
            temp = head
            head = next[head]
            next[temp] = -1
            sums[temp] = 0
        # Sort column indices and corresponding values.
        col_indices_values = sorted(zip(ans.colIndices[nnz - length:nnz], ans.data[nnz - length:nnz]), key=lambda x: x[0])
        if len(col_indices_values):
            ans.colIndices[nnz - length:nnz], ans.data[nnz - length:nnz] = zip(*col_indices_values)
        ans.rowPtrs[i + 1] = nnz

def _csr_matmul_my(nRows: int, nCols: int, A: CSR_Matrix, B: CSR_Matrix, ans: CSR_Matrix) -> None:
    nnz = 0
    ans.rowPtrs[0] = 0
    next = np.full(nCols, -1, dtype=int)
    sums = np.zeros(nCols, dtype=int)
    for row in range(nRows):    
        head = -2
        length = 0
        dataIndexInEachARow = range(A.rowPtrs[row], A.rowPtrs[row + 1])
        for dataIndexInA in  dataIndexInEachARow:
            currentDataCol = A.colIndices[dataIndexInA]
            currentDataValue = A.data[dataIndexInA]
            dataIndexInEachCorrespondBRow = range(B.rowPtrs[currentDataCol], B.rowPtrs[currentDataCol + 1])
            for dataIndexInB in dataIndexInEachCorrespondBRow:
                ansCol = B.colIndices[dataIndexInB]
                sums[ansCol] += currentDataValue * B.data[dataIndexInB]
                if next[ansCol] == -1:
                    next[ansCol] = head
                    head = ansCol
                    length += 1
        for _ in range(length):
            ans.data[nnz] = sums[head]
            ans.colIndices[nnz] = head
            nnz += 1
            temp = head
            head = next[head]   
            next[temp] = -1
            sums[temp] = 0
        colIndicesValuesSort = sorted(zip(ans.colIndices[nnz - length:nnz], ans.data[nnz - length:nnz]), key=lambda x: x[0])
        if len(colIndicesValuesSort):
            ans.colIndices[nnz - length:nnz], ans.data[nnz - length:nnz] = zip(*colIndicesValuesSort)
        ans.rowPtrs[row + 1] = nnz
            
def CsrMM(A: CSR_Matrix, B: CSR_Matrix) -> CSR_Matrix:
    nRows = A.shape[0]
    nCols = B.shape[1]
    numNonZeroInC = _csr_matmult_maxnnz(nRows, nCols, A.rowPtrs, A.colIndices, B.rowPtrs, B.colIndices)
    cRowPtrs = np.zeros(nRows + 1, dtype=int)
    cColInddices = np.zeros(numNonZeroInC , dtype=int)
    cData = np.zeros(numNonZeroInC , dtype=int)
    C = CSR_Matrix(cRowPtrs, cColInddices, cData, (nRows, nCols))
    _csr_matmul_my(nRows, nCols, A, B, C)
    return C

def test_1():
    A = CSR_Matrix([0, 2, 2, 3], [0, 2, 1], [1, 2, 4], (3, 3))
    B = CSR_Matrix([0, 2, 2, 2], [1, 2], [3, 4], (3, 3))
    print(A, end='\n\n')
    print(A.to_dense(), end='\n\n')
    print(B, end='\n\n')
    print(B.to_dense(), end='\n\n')
    C = CsrMM(A, B)
    print(C, end='\n\n')
    print(C.to_dense(), end='\n\n')

def test_2():
    A = CSR_Matrix([0, 1, 3, 3], [2, 0, 2], [3, 4, 5], (3, 3))
    B = CSR_Matrix([0, 1, 1, 3], [1, 0, 1], [4, 3, 5], (3, 3))
    C = CsrMM(A, B)
    print(C, end='\n\n')
    print(C.to_dense(), end='\n\n')

def test_3():
    A = CSR_Matrix([0, 2, 5, 7, 9, 10, 12, 14, 16, 16, 17],  [0, 3, 1, 4, 9, 0, 5, 2, 7, 8, 3, 6, 5, 7, 9, 1, 5],  [3, 4, 1, 2, 7, 3, 5, 8, 6, 1, 3, 9, 1, 2, 5, 2, 6], (10, 10))
    B = CSR_Matrix([0, 2, 4, 6, 8, 10, 12, 14, 16, 16, 18], [0, 6, 2, 4, 0, 3, 1, 4, 3, 7, 0, 5, 1, 6, 2, 5, 6, 7], [4, 2, 7, 3, 5, 1, 6, 3, 4, 8, 1, 2, 5, 7, 9, 2, 6, 1], (10, 8))
    print(A, end='\n\n')
    print(A.to_dense(), end='\n\n')
    print(B, end='\n\n')
    print(B.to_dense(), end='\n\n')
    C = CsrMM(A, B)
    print(C, end='\n\n')
    print(C.to_dense(), end='\n\n')
    DenseC = A.to_dense() * B.to_dense()
    print(DenseC)

if __name__ == "__main__":
    #test_1()
    test_2()
    #test_3()