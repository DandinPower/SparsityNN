import torch

# Define a sparse matrix in COO format
indices = torch.tensor([
    [0, 0, 2],
    [0, 2, 1],
], dtype=torch.long)
values = torch.tensor([1, 2, 4], dtype=torch.float32)
sparse_shape = (3, 3)
sparse_matrix = torch.sparse_coo_tensor(indices, values, sparse_shape)

# Define a sparse matrix in COO format
indices = torch.tensor([
    [0, 1, 2],
    [1, 0, 1],
], dtype=torch.long)
values = torch.tensor([3, 1, 2], dtype=torch.float32)
sparse_shape = (3, 2)

sparse_matrix2 = torch.sparse_coo_tensor(indices, values, sparse_shape)

print(sparse_matrix.to_dense(), end="\n\n")
print(sparse_matrix2.to_dense(), end="\n\n")

result = torch.sparse.mm(sparse_matrix, sparse_matrix2)
print(result, end="\n\n")
print(result.to_dense(), end="\n\n")

