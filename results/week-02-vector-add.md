# Week 02 Vector Add

## Environment
- Machine: HP OMEN-35L (x86-64)
- OS: Ubuntu 24.04.4 LTS
- Python:3.14.7
- Numpy: 2.5.4
- GPU: RTX 5060 TI (16GB), CUDA 13.2
- `nvcc --version`: Cuda compilation tools, release 13.2, V13.2.86

## Python Reference

Input: `a = np.array([1.0, 2.0, 3.0], dtype=np.float64)` and `np.array([4.0, 5.0, 6.0], dtype=np.float64)`

Output: `[5. 7. 9.]` and `expected: [5. 7. 9.]`

What this proves: It proves that the output result is correct i.e. what we expected.

## CUDA Starter

Compile command: `nvcc -O2 -std=c++14 cuda/vector_add.cu -o build/vector_add

Run output: For command `./build/vector_add`, the output was `vector_add passed for 1048576 elements`


## Kernel Mapping

Explain `idx = blockIdx.x * blockDim.x + threadIdx.x`

Unlike CPU, GPU can use thousands of thread to perform an operation on a data parallel task such as - vector addition where each element is independent i.e. it doesn't matter in which order the elements are added. Each thread performs the addition on a single element i.e. each thread much have a unique id so that it can access the element of a vector in that position. Here `idx` contains that unique id and `blockIdx`, `blockDim`, `threadIdx` are reserved variables. 

Threads are organized into blocks and `blockDim` represents the number of threads in a block. If there are $16$ threads in a block, then its dimension will be $16$. Suppose we want to add two vectors of length $16$, then we can use $1$ block containing $16$ threads and so the `blockDim` will be $16$. In this case, there will be just $1$ block and so its index is $0$.

***Blocks can be 1D or 2D or 3D.***


## Bounds Check

Explain why `if (idx < n)` is needed.

The number of elements that need to added may not be exactly multiple of the block size. So, the bound check prevents extra threads from reading or writing outside the array. For example -

Suppose we want to add two vector containing $30$ elements and for that we're using $2$ blocks each containing $16$ threads. Therefore, we'll have $32$ threads in total. So, we need to have a bound check so that the extra two threads don't perform any operation.

## What I Would Debug First

If the results were wrong, I would inspect the input array elements first and then the copy operation from host to device.
