import time
import torch
import matplotlib.pyplot as plt

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the sizes of the matrices to test
sizes = [(pow(2, size), pow(2, size)) for size in range(5,11)]
plot_sizes = [x for x,_ in sizes]

# Define the sparsity percentage of the matrices
sparsity = 0.8

# Define the workloads to test
workloads = ["dense", "sparse"]

# Define lists to store the performance results
dense_times = []
sparse_times = []

# Test each workload for each matrix size
for size in sizes:
    num_nonzeros = int(size[0] * size[1] * sparsity)

    # Generate random indices and values for the non-zero elements
    indices = torch.rand(2, num_nonzeros, device=device) * torch.tensor(size, device=device).reshape(2, 1)
    values = torch.randn(num_nonzeros, device=device)

    sparse_tensor1 = torch.sparse_coo_tensor(indices, values, size, device=device)
    sparse_tensor2 = torch.sparse_coo_tensor(indices.flip(0), values, size[::-1], device=device)

    dense_tensor1 = sparse_tensor1.to_dense().to(device)
    dense_tensor2 = sparse_tensor2.to_dense().to(device)

    dense_time = 0
    sparse_time = 0

    for workload in workloads:
        for i in range(len(sizes)):
            if workload == "dense":
                result = torch.mm(dense_tensor1, dense_tensor2)
            elif workload == "sparse":
                result = torch.sparse.mm(sparse_tensor1, sparse_tensor2)

    # Time the execution of matrix multiplication
    if workload == "dense":
        start = time.time()
        result = torch.mm(dense_tensor1, dense_tensor2)
        dense_time = time.time() - start
    elif workload == "sparse":
        start = time.time()
        result = torch.sparse.mm(sparse_tensor1, sparse_tensor2)
        sparse_time = time.time() - start

    dense_times.append(dense_time)
    sparse_times.append(sparse_time)

print(dense_times)
print(sparse_times)

# Plot the results using Matplotlib
plt.plot(plot_sizes, dense_times, label="Dense")
plt.plot(plot_sizes, sparse_times, label="Sparse")
plt.legend()
plt.title("Matrix Multiplication Performance")
plt.xlabel("Matrix Size")
plt.ylabel("Time (s)")
plt.savefig('comparison_gpu.png')
plt.clf()