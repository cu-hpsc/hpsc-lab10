# HPSC Lab 10
2019-11-01

Click to [make your own repo](TO DO).

The goals for this lab are:
* Become more familiar with CUDA syntax;
* Run code on a RMACC Summit GPU.
* Explore restructure code for optimizations.

-----

On Wendesday we went over some of the common syntax for CUDA.  In this lab's directory you will find two implementations of kernels for matrix multiplication on the GPU device.

`matmul.cu` contains a straightforward implementation, where each thread independently reads a row of left matrix A and a column of right matrix B to compute the vector product to be placed in the matrix C element corresponding to that thread.

![](matrix-multiplication-without-shared-memory.png)

This means that A is read B.width times from global memory and B is read A.height times.

As with lab 7, to compile on Summit you will need to ssh in, ssh to a compile node, and load the correct modules:
```
ssh 
ssh scompile
module load gcc/6.1.0 cuda
```

Then invoke the cuda compiler:
```
nvcc matmul.cu -o matmul
```

*Depending on how they've set up the compile nodes, this is probably compiling for compute capability 3.0 instead of 3.7.*

`jobscript` is provided for you to run this executable.  The main difference from normal Summit jobscripts is the partition option.  Submit as normal through Slurm, and examine the output file matching the job number.
```
sbatch jobscript
```

-----

`matmul_shared.cu` contains a slightly smarter implementation where blocks of threads first load blocks of A and B into shared memory, and then compute those blocks' contributions to C.

![](matrix-multiplication-with-shared-memory.png)

Recall from Wednesday that shared memory is much faster than global memory.  By allowing the threads within a block to reuse the same memory, the A matrix is only read (B.width / block_size) times from global memory and B is only read (A.height / block_size) times.

The equivalent compile and submission commands are:
```
nvcc matmul_shared.cu -o matmul_shared
sbatch jobscript_shared
```

-----

Looking through the `matmul_shared.cu`, you will see some new syntax from what was covered Wednesday.  Attend to the following:

* `__device__` function specifiers in place of `__global__`.  Recall that `__global__` tells the compiler that the function will be executed on the device and can be called by the host.  `__device__` tells the compiler that the function will be executed on the device and only called from the device.  Older compute capabilities couldn't call functions from the device.
* `__shared__` variable specifiers.  This tells the compiler to allocate the variable in shared memory.  How sensible!
* `__syncthreads()` function call.  This syncs the threads across all warps of a block.

-----

What other optimization considerations discussed on Wednesday are you interested in exploring?  Add and push some observations back to your local repo.