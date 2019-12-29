
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <cstdlib> 
#include <curand.h>
using namespace std;

#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <random>

#define Npoints 40000

__global__ void setup_kernel(curandState* state, unsigned long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, i, 0, &state[i]);
}

__global__ void generate(curandState* globalState, float* devRez)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = globalState[i];
    double x = curand_normal_double(&localState);
    double y = curand_uniform_double(&localState);
    if ((x*x + y*y) <= 1) {
        devRez[i] = 1.0;
    }
    globalState[i] = localState;
    
}

int main(int argc, char** argv)
{
    curandState* devStates;
    float* rez = new float[Npoints];
    float* devRez;

    cudaSetDevice(0);
    cudaMalloc(&devStates, Npoints * sizeof(curandState));
    setup_kernel << < Npoints/1024 + 1, 1024 >> > (devStates, time(0));

    cudaMalloc(&devRez, Npoints * sizeof(*rez));
    float gpuTime;

    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start, 0);
    
    generate << < Npoints / 1024 + 1, 1024 >> > (devStates, devRez);

    cudaEventRecord(gpu_stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&gpuTime, gpu_start, gpu_stop);
    cudaMemcpy(rez, devRez, Npoints * sizeof(*rez), cudaMemcpyDeviceToHost);
    double NincircleGpu = 0;
    cout << NincircleGpu << endl;
    for (int i = 0; i < Npoints; i++)
    {
        if (rez[i] == 1) {
            NincircleGpu++;
        }
    }
    cout << NincircleGpu << endl;
   double piGPU = (double)(NincircleGpu * 4.0 / (double)Npoints);
   printf("PI gpu: %.5f \n", piGPU);
   cudaFree(devRez);
   cudaFree(devStates);

   double Nincircle = 0;
   double x;
   double y;
   unsigned int cpu_start = clock();
   srand(time(0));
       for (int i = 0; i < Npoints; i++) {
           x = 0.01 * (rand() % 101);
           y = 0.01 * (rand() % 101);

           if (x*x + y*y <= 1) {
               Nincircle++;
           }
       }
   unsigned int cpu_end = clock();
   double pi = (double)(Nincircle * 4.0 / (double)Npoints);
   
   printf("PI cpu %.5f \n", pi);

   printf("gpu time= %.5f seconds\n", gpuTime / CLOCKS_PER_SEC);
   printf("cpu time= %.5f seconds\n", (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC);
   delete rez;
    return 0;
}



