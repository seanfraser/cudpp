// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/*
 * This is a basic example of how to use the CUDPP library.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <float.h>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include "tensor.h"
#include <vector>
#include <numeric>

// includes, project
#include "cudpp.h"

#include <string>

typedef double dtype;


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C"
void computeSumScanGold( dtype *reference, const dtype *idata, 
                        const unsigned int len,
                        const CUDPPConfiguration &config);

extern "C"
void computeRMSError(dtype* h_idata, dtype* h_odata, size_t numElements);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    if (argc > 1) {
        std::string arg = argv[1];
        size_t pos = arg.find("=");
        if (arg.find("device") && pos != std::string::npos) {
            dev = atoi(arg.c_str() + (pos + 1));
        }
    }
    if (dev < 0) dev = 0;
    if (dev > deviceCount-1) dev = deviceCount - 1;
    cudaSetDevice(dev);

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               prop.name, (int)prop.totalGlobalMem, (int)prop.major, 
               (int)prop.minor, (int)prop.clockRate);
    }

    // distribution params
    int params[3][2] = {{0,1}, {1, 1}, {0, 1}};

    unsigned int power_of_2 = 25;

    printf ("input size: 2^%d\n", power_of_2);

    unsigned int numElements = 1 << power_of_2;
    unsigned int memSize = sizeof( dtype) * numElements;
    size_t N = numElements;

    // 0 for uniform, 1 for exp, 2 for normal
    int distr_flag = 0;

    printf ("distribution: %d (0: uniform, 1: exp, 2: normal)\n", distr_flag);

    printf("data type: %lu-bits floating-point\n", sizeof(dtype) * 8);

    const std::vector<size_t> dim_lens{N};

    exsum_tensor::Tensor<dtype, 1> tensor(dim_lens);
    tensor.RandFill(params[distr_flag][0], params[distr_flag][1], distr_flag);

    // allocate host memory
    dtype* h_idata = tensor.data();

    // allocate device memory
    dtype* d_idata;
    cudaError_t result = cudaMalloc( (void**) &d_idata, memSize);
    if (result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }

    // copy host memory to device
    result = cudaMemcpy( d_idata, h_idata, memSize, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }

    // allocate device memory for result
    dtype* d_odata;
    result = cudaMalloc( (void**) &d_odata, memSize);
    if (result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }

    // Initialize the CUDPP Library
    CUDPPHandle theCudpp;
    cudppCreate(&theCudpp);

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_DOUBLE;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

    CUDPPHandle scanplan = 0;
    CUDPPResult res = cudppPlan(theCudpp, &scanplan, config, numElements, 1, 0);  

    if (CUDPP_SUCCESS != res)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }

    // start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Run the scan
    res = cudppScan(scanplan, d_odata, d_idata, numElements);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error in cudppScan()\n");
        exit(-1);
    }

    // end timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("consumed time: %9.6f ms for cudppScan\n", elapsedTime);

    // allocate mem for the result on host side
    dtype* h_odata = (dtype*) malloc( memSize);
    // copy result from device to host
    result = cudaMemcpy( h_odata, d_odata, memSize, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }

    printf("last element from GPU device: %.6f\n", h_odata[numElements-1]);

    // check accuracy
    computeRMSError(h_idata, h_odata, numElements);

    res = cudppDestroyPlan(scanplan);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }

    // shut down the CUDPP library
    cudppDestroy(theCudpp);

    free( h_idata);
    free( h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    exit(0);
}
