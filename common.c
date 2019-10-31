#include <stdio.h>

#include "rdtsc.h"

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;


// Get a matrix element on device
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element on device
__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}


// Thread block size
#define BLOCK_SIZE 32

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);


// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
      Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    d_A.stride = A.stride;
    size_t size = A.stride * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    d_B.stride = B.stride;
    size = B.stride * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    d_C.stride = C.stride;
       size = C.stride * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}


/////////////////////////////////////////////////////////////////////////////
// Boring setup stuff

#define DIFFICULTY (256)
#define A_M (BLOCK_SIZE*DIFFICULTY)
#define A_N (BLOCK_SIZE*DIFFICULTY)
#define B_M (A_N)
#define B_N (BLOCK_SIZE*DIFFICULTY)
#define C_M (A_M)
#define C_N (B_N)

int main()
{
  // Allocate A, B, and C in host memory
  Matrix A;
  A.height = A_M;
  A.width = A_N;
  A.stride = A_N;
    A.elements = (float*) malloc(A.stride * A.height * sizeof(float));
  Matrix B;
  B.height = B_M;
  B.width = B_N;
  B.stride = B_N;
  B.elements = (float*) malloc(B.stride * B.height * sizeof(float));
  Matrix C;
  C.height = C_M;
  C.width = C_N;
  C.stride = C_N;
  C.elements = (float*) malloc(C.stride * C.height * sizeof(float));

  // initialize A and B (arbitrary values)
  for (int m = 0; m < A_M; ++m) {
    for (int n = 0; n < A_N; ++n) {
      A.elements[m*A.stride+n] = (float)(m+1)/(n+2);
    }
  }
  for (int m = 0; m < B_M; ++m) {
       for (int n = 0; n < B_N; ++n) {
      B.elements[m*B.stride+n] = (float)(B_N-n)/(B_M-m+1);
    }
  }

  // multiply
  ticks_t lasttick = rdtsc();
  MatMul(A,B,C);
  printf("Kernel+transfer took %ld ticks\n",rdtsc()-lasttick);

  // free host memory
  free(A.elements);
  free(B.elements);
  free(C.elements);
  return 0;
}            
