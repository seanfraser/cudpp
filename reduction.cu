// ---*- C++ -*---
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include <cassert>
#include <algorithm>

#include <cuda.h>
#include <cooperative_groups.h>

// Handy macro borrowed to report CUDA errors, borrowed from:
// https://devtalk.nvidia.com/default/topic/1025474/cuda-programming-and-performance/kernel-launched-via-cudalaunchcooperativekernel-runs-in-different-stream/
#define CUERR {								\
    cudaError_t err;							\
    if ((err = cudaGetLastError()) != cudaSuccess) {			\
      printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(1);								\
    }									\
  }

static inline double tdiff(struct timeval* a, struct timeval* b)
// Effect: Compute the difference of b and a, that is $b-a$
//  as a double with units in seconds
{
  return a->tv_sec - b->tv_sec + 1e-6 * (a->tv_usec - b->tv_usec);
}

using namespace cooperative_groups;

// The sum_kernel_block, reduce_sum, and thread_sum codes were given
// in the NVIDIA blog post on cooperative thread groups:
// https://devblogs.nvidia.com/cooperative-groups/
__device__ float reduce_sum(thread_group g, float *temp, float val) {
  int lane = g.thread_rank();	

  // Each iteration halves the number of active threads
  // Each thread adds its partial sum[i] to sum[lane+i]
  for (int i = g.size() / 2; i > 0; i /= 2) {
    temp[lane] = val;
    g.sync(); // wait for all threads to store
    if (lane < i)
      val += temp[lane + i];
    g.sync(); // wait for all threads to load
  }
  return val; // note: only thread 0 will return full sum
}


// The thread_sum_vec code is the actual version of thread_sum in the
// NVIDIA blog post on cooperative thread groups, but it appears to be
// buggy.  The following thread_sum code removes the vector operations
// and thereby fixes the bug.

__device__ float thread_sum(float *input, int n) {
  float sum = 0.0;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += blockDim.x * gridDim.x) {
    sum += input[i];
  }
  return sum;
}

// __device__ int thread_sum_vec(int *input, int n) {
//   int sum = 0;

//   for (int i = blockIdx.x * blockDim.x + threadIdx.x;
//        i < n / 4;
//        i += blockDim.x * gridDim.x) {
//     int4 in = ((int4*)input)[i];
//     sum += in.x + in.y + in.z + in.w;
//   }
//   return sum;
// }

__global__ void sum_kernel_block(float *sum, float *input, int n) {
  float my_sum = thread_sum(input, n);

  extern __shared__ float temp[];
  auto g = this_thread_block();
  float block_sum = reduce_sum(g, temp, my_sum);

  if (g.thread_rank() == 0)
    atomicAdd(sum, block_sum);
    // *sum = block_sum;
}


// Upsweep algorithm adapted from cache-efficient prefix-sum
// computation for CPU's.
__device__ void upsweep(float *A, int y) {
  int delta = 1;
  int repeat = y;
  do {
    // printf("y = %d, delta = %d\n", y, delta);
    A[y] += A[y - delta];
    if ((repeat & 2) == 0) {
      break;
    } else {
      delta <<= 1;
      repeat >>= 1;
    }
  } while (true);
}

// Top-level kernel code that uses the upsweep algorithm to compute
// the sum of all elements in an array of power-of-2 size.
const int blockSize = 1024;
__global__ void upsweep_kernel_block(float *sum, float *input, int n) {
  __shared__ float temp[blockSize / 32];
  // Apparently calling this_grid() is VERY expensive.
  // auto g = this_grid();
  auto blk = this_thread_block();

  // Get the ID of this thread within its thread block and on the
  // device overall.
  int bThreadID = blk.thread_rank();
  int gThreadID = bThreadID + (blk.group_index().x * blockSize);
  // Get a 32-thread (1-warp) tile for each block.  These threads will
  // operate in lock step on a given input.
  auto tile = tiled_partition(blk, 32);

  // This code uses 1/2 as many threads as there are elements in the
  // input array.
  if (gThreadID < n / 2) {
    // Perform an upsweep on an aligned chunk of 64 array elements.
    upsweep(&input[(gThreadID / 32) * 64], (tile.thread_rank() * 2) + 1);

    // Save the final summation into the shared temporary array.  This
    // code relies on arrays of power-of-2 size.
    if (bThreadID % 32 == 31 || gThreadID == (n / 2) - 1) {
      temp[bThreadID / 32] = input[(2 * gThreadID) + 1];
    }
  }

  // Sync the warps
  blk.sync();

  // If we're dealing with small inputs, no more upsweeps are needed.
  if (n < 128) {
    *sum = temp[0];
    return;
  }

  // For arrays of more than 2048 elements, the following code relies
  // on power-of-2 sized inputs.

  // In each thread block, perform an upsweep on the elements in the
  // shared temporary array.
  if (bThreadID < min(n, 2048) / 128)
    upsweep(temp, (bThreadID * 2) + 1);

  // Synchronization primitives that work across the thread blocks in
  // the grid seem somewhat lacking.  The simplest way to accumulate
  // these sums is with an atomic addition.  This code only matters
  // when we're dealing with arrays of more than 2048 elements;
  // otherwise the entire reduction is handled in a single thread
  // block.
  if (bThreadID == (min(n, 2048) / 128) - 1)
    atomicAdd(sum, temp[(bThreadID * 2) + 1]);
}

// __global__ void upsweep_kernel_block(float *sum, float *input, int n) {
//   extern __shared__ float temp[];
//   // auto g = this_grid();
//   auto g = this_thread_block();
//   auto tile = tiled_partition(this_thread_block(), 32);
//   // int my_n = n;
//   float *my_input = input;
//   float *my_temp = &temp[0];
//   for (int my_n = n; my_n > 1; my_n /= 64) {
//     // printf("my_n = %d\n", my_n);
//     if (g.thread_rank() < my_n / 2) {
//       upsweep(&my_input[(g.thread_rank() / 32) * 64],
// 	      (tile.thread_rank() * 2) + 1);

//       if (g.thread_rank() % 32 == 31 || g.thread_rank() == (my_n / 2) - 1) {
// 	my_temp[g.thread_rank() / 32] = my_input[(2 * g.thread_rank()) + 1];
// 	// printf("temp[%d] = %f\n", g.thread_rank() / 32, temp[g.thread_rank() / 32]);
//       }
//       my_input = &my_temp[0];
//       my_temp = &my_temp[(my_n + 64 - 1) / 64];
//     }
//   }
//   // g.sync();
//   if (g.thread_rank() == 0)
//     *sum = my_input[0];
// }

// Initialize the input data.
void initialize(float *sum, float *data, int n) {
  cudaMemset(sum, 0.0, sizeof(float)); CUERR;
  // Initialize data[] such that data[i] = (-1^i) * i / 10.0.  This
  // array allows us to check the result of the sum easily and observe
  // errors that arise in the floating-point calculation.
  for (int i = 0; i < n; ++i) {
    data[i] = ((float)i) / 10.0;
    if (i % 2)
      data[i] *= -1.0;
  }
}

int main(int argc, char *argv[]) {
  struct timeval tm_begin, tm_end;
  int n = 20;
  if (argc > 1)
    n = atoi(argv[1]);

  int deviceId = 0;
  {
    // Get information about the GPU device.  Verify that it supports
    // cooperative thread groups.
    cudaSetDevice(deviceId);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId); CUERR;

    int pi = 0;
    CUdevice dev;
    cuDeviceGet(&dev,0); // get handle to device 0
    cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, dev);
    assert(pi && "Cooperative launch is not supported\n");
  }

  const int blockSize = 1024;
  const int nBlocks = (n + blockSize - 1) / blockSize;

  // Allocate sum and data[] in CUDA unified memory, so they can be
  // accessed by both the host CPU and the GPU.
  float *sum, *data;
  cudaMallocManaged(&sum, sizeof(float)); CUERR;
  cudaMallocManaged(&data, n * sizeof(float)); CUERR;

  // Hard-coded 5 trials for now.  
  for (int trials = 0; trials < 5; ++trials) {
    //------------------------------------------------------------------------
    // initialize data
    initialize(sum, data, n);

    // Attempting to prefetch the data onto the GPU, but these calls
    // don't seem to affect performance much.
    cudaMemPrefetchAsync(data, n * sizeof(float), deviceId);
    cudaMemPrefetchAsync(sum, sizeof(float), deviceId);

    gettimeofday(&tm_begin, 0);
    {
      // // Launch the kernel, if we're not using cooperative thread groups.
      // sum_kernel_block<<<nBlocks, blockSize, sharedBytes>>>(sum, data, n);

      // Launch the kernel using cooperative thread groups.
      void *args[3] = {(void*)&sum, (void*)&data, (void*)&n};
      cudaLaunchCooperativeKernel((void *)sum_kernel_block, nBlocks, blockSize,
				  args, /*sharedBytes=*/blockSize * sizeof(float)); CUERR;
      // Wait for the GPU to finish.
      cudaDeviceSynchronize();
    }
    gettimeofday(&tm_end, 0);
    printf("sum = %f\n", *sum);

    // The timing results measured by gettimeofday() appear very
    // noisy.  I suspect the problem lies in the variable performance
    // of offloading the kernel and data onto the GPU over the PCI
    // bus.  The profile results measured by nvprof appear more
    // stable.
    printf("consumed time: %9.6f ms for sum_kernel_block\n", tdiff(&tm_end, &tm_begin) * 1.0e3);

    //------------------------------------------------------------------------
    // initialize data
    initialize(sum, data, n);

    // Attempting to prefetch the data onto the GPU, but these calls
    // don't seem to affect performance much.
    cudaMemPrefetchAsync(data, n * sizeof(float), deviceId); CUERR;
    cudaMemPrefetchAsync(sum, sizeof(float), deviceId); CUERR;

    gettimeofday(&tm_begin, 0);
    {
      // // Luanch the kernel, if we're not using cooperative thread groups.
      // upsweep_kernel<<<1, 2, nBlocks * 2 * blockSize * sizeof(float) / 64>>>(sum, data, n); CUERR;

      // Launch the kernel using cooperative thread groups.
      void *args[3] = {(void*)&sum, (void*)&data, (void*)&n};
      cudaLaunchCooperativeKernel((void *)upsweep_kernel_block,
				  ((n / 2) + blockSize - 1) / blockSize, blockSize, args); CUERR;
      cudaDeviceSynchronize();
    }
    gettimeofday(&tm_end, 0);
    printf("sum = %f\n", *sum);

    // The timing results measured by gettimeofday() appear very
    // noisy.  I suspect the problem lies in the variable performance
    // of offloading the kernel and data onto the GPU over the PCI
    // bus.  The profile results measured by nvprof appear more
    // stable.
    printf("consumed time: %9.6f ms for upsweep_cuda\n", tdiff(&tm_end, &tm_begin) * 1.0e3);
  }

  cudaFree(data);
  cudaFree(sum);

  return 0;
}

