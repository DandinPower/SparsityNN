/*
Simplified Version : 
The _csr_matmult function performs the actual sparse matrix multiplication 
using the CSR format. It first extracts the CSR data (row pointers, column indices, and values) 
from the input tensors and then iterates over the rows of the first matrix (mat1) 
and the columns of the second matrix (mat2) to compute the resulting matrix. 
During this process, it accumulates the product of the elements that have the same inner indices.
*/
template <typename scalar_t>
void _csr_matmult(const Tensor& r_, const Tensor& mat1, const Tensor& mat2) {
  // Extract CSR data (row pointers, column indices, and values) from input tensors
  // ...

  for (int64_t row = 0; row < mat1.size(0); row++) {
    for (int64_t col = 0; col < mat2.size(1); col++) {
      scalar_t sum = 0;

      for (int64_t k = row_ptr1[row]; k < row_ptr1[row + 1]; k++) {
        int64_t mat1_col = col_indices1[k];

        for (int64_t l = row_ptr2[mat1_col]; l < row_ptr2[mat1_col + 1]; l++) {
          if (col_indices2[l] == col) {
            sum += values1[k] * values2[l];
            break;
          }
        }
      }

      if (sum != 0) {
        // Add the computed sum to the resulting sparse tensor (r_)
        // ...
      }
    }
  }
}

/*
Actual Version : 
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
*/
template<class scalar_t>
void _csr_matmult(
    const int64_t n_row,
    const int64_t n_col,
    const int64_t Ap[],
    const int64_t Aj[],
    const scalar_t Ax[],
    const int64_t Bp[],
    const int64_t Bj[],
    const scalar_t Bx[],
    int64_t Cp[],
    int64_t Cj[],
    scalar_t Cx[]) {
  /*
    Compute CSR entries for matrix C = A@B.
    The matrices `A` and 'B' should be in proper CSR structure, and their dimensions
    should be compatible.
    Inputs:
      `n_row`         - number of row in A
      `n_col`         - number of columns in B
      `Ap[n_row+1]`   - row pointer
      `Aj[nnz(A)]`    - column indices
      `Ax[nnz(A)]     - nonzeros
      `Bp[?]`         - row pointer
      `Bj[nnz(B)]`    - column indices
      `Bx[nnz(B)]`    - nonzeros
    Outputs:
      `Cp[n_row+1]` - row pointer
      `Cj[nnz(C)]`  - column indices
      `Cx[nnz(C)]`  - nonzeros
    Note:
      Output arrays Cp, Cj, and Cx must be preallocated
  */
  std::vector<int64_t> next(n_col, -1);
  std::vector<scalar_t> sums(n_col, 0);

  int64_t nnz = 0;

  Cp[0] = 0;

  for (const auto i : c10::irange(n_row)) {
    int64_t head = -2;
    int64_t length = 0;

    int64_t jj_start = Ap[i];
    int64_t jj_end = Ap[i + 1];
    for (const auto jj : c10::irange(jj_start, jj_end)) {
      int64_t j = Aj[jj];
      scalar_t v = Ax[jj];

      int64_t kk_start = Bp[j];
      int64_t kk_end = Bp[j + 1];
      for (const auto kk : c10::irange(kk_start, kk_end)) {
        int64_t k = Bj[kk];

        sums[k] += v * Bx[kk];

        if (next[k] == -1) {
          next[k] = head;
          head = k;
          length++;
        }
      }
    }

    for (const auto jj : c10::irange(length)) {
      (void)jj; //Suppress unused variable warning

      // NOTE: the linked list that encodes col indices
      // is not guaranteed to be sorted.
      Cj[nnz] = head;
      Cx[nnz] = sums[head];
      nnz++;

      int64_t temp = head;
      head = next[head];

      next[temp] = -1; // clear arrays
      sums[temp] = 0;
    }

    // Make sure that col indices are sorted.
    // TODO: a better approach is to implement a CSR @ CSC kernel.
    auto col_indices_accessor = StridedRandomAccessor<int64_t>(Cj + nnz - length, 1);
    auto val_accessor = StridedRandomAccessor<scalar_t>(Cx + nnz - length, 1);
    auto kv_accessor = CompositeRandomAccessorCPU<
      decltype(col_indices_accessor), decltype(val_accessor)
    >(col_indices_accessor, val_accessor);
    std::sort(kv_accessor, kv_accessor + length, [](const auto& lhs, const auto& rhs) -> bool {
        return get<0>(lhs) < get<0>(rhs);
    });

    Cp[i + 1] = nnz;
  }
}

