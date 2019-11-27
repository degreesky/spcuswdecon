#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <cufft.h>
#include <cuComplex.h>
#include <assert.h>

//Include external files
#include "../headers/errormessage.h"

//forward declaration
__global__ void vecScale(float *array, float scale, int n);

extern "C" void __device__fft_gather(int fft_type, float scale, float *data, cufftComplex *dataw, int ntw, int nttr, int nt) {
	int rank = 1, inembed = { 0 }, onembed = { 0 }, istride = 1, idist = nt, ostride = 1, odist = ntw;
	int iDev = 0;  //only one device at the moment...
	int blockSize;   // The launch configurator returned block size
	int minGridSize; // The minimum grid size needed to achieve the
	                 // maximum occupancy for a full device launch
	int gridSize;    // The actual grid size needed, based on input size
	int arrayCount = nt*nttr;

	cufftHandle planfwcu = { 0 }, planbwcu = { 0 };
	cufftReal *hostInputData, *deviceInputData,*deviceOutputData;
	cufftComplex *hostInputDataw, *deviceInputDataw, *deviceOutputDataw;
	cudaError_t error_id;

	// For 1D transform:   input[ b * idist + x * istride]
	//                    output[ b * odist + x * ostride]

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, iDev);

	if (fft_type == CUFFT_FORWARD) {
		//****************Host methods***********
		//Host side Input data allocation
		hostInputData = (cufftReal*) data; //pointer to the original loaded data

		//Host side output data allocation
		//cufftComplex *hostOutputData = (cufftComplex*)malloc(ntw * nttr * sizeof(cufftComplex));

		//****************Device methods***********
		//Device side input data allocation and initialization
		CHECK_CUDA_ERRORS(cudaMalloc((void** )&deviceInputData,nt * nttr * sizeof(cufftReal)));
		//Some later adaptation here to make the memory copy from the Host to Device to a chunk size
		CHECK_CUDA_ERRORS(cudaMemcpy(deviceInputData, hostInputData,nt * nttr * sizeof(cufftReal), cudaMemcpyHostToDevice));
		//Device side output data allocation
		CHECK_CUDA_ERRORS(cudaMalloc((void** )&deviceOutputDataw,ntw * nttr * sizeof(cufftComplex)));

		//Make a plan
		CHECK_CUFFT_ERRORS(cufftPlanMany(&planfwcu, rank, &nt, &inembed, istride, idist,&onembed, ostride, odist, CUFFT_R2C, nttr));

		fprintf(stderr, "CUFFT FORWARD\n");
		CHECK_CUFFT_ERRORS(cufftExecR2C(planfwcu, deviceInputData, deviceOutputDataw));

		cudaDeviceSynchronize(); //kernel is guaranteed to finish
		//Can do printing here...

		//Device->Host copy of the results
		CHECK_CUDA_ERRORS(cudaMemcpy(dataw, deviceOutputDataw,ntw * nttr * sizeof(cufftComplex),cudaMemcpyDeviceToHost));

		//Destroy the plan
		CHECK_CUFFT_ERRORS(cufftDestroy(planfwcu));

		//Release device memory
		CHECK_CUDA_ERRORS(cudaFree(deviceInputData));
		CHECK_CUDA_ERRORS(cudaFree(deviceOutputDataw));

	} else if (fft_type == CUFFT_INVERSE) {
		//Device properties:
		//fprintf(stderr,"  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
	    //fprintf(stderr,"Warp size:                                       %d\n",  deviceProp.warpSize);
	    //fprintf(stderr,"Maximum # Threads per MultiProcessor:            %d\n",deviceProp.maxThreadsPerMultiProcessor);

		//redefinition of stride I/O for inverse FFT plan
		idist = ntw, odist = nt;

		//****************Host methods***********
		//Host side Input data allocation
		hostInputDataw = dataw;    //pointer to the original loaded data

		//Host side output data allocation
		cufftReal *hostOutputData = (cufftReal*)data;
		//hostOutputData = (cufftReal*) malloc(nt * nttr * sizeof(cufftReal));

		//****************Device methods***********
		//Device side input data allocation and initialization
		CHECK_CUDA_ERRORS(cudaMalloc((void** )&deviceInputDataw,ntw * nttr * sizeof(cufftComplex)));

		//Some later adaptation here to make the memory copy from the Host to Device to a chunk size
		// check for error
		CHECK_CUDA_ERRORS(cudaMemcpy(deviceInputDataw, hostInputDataw,ntw * nttr * sizeof(cufftComplex),cudaMemcpyHostToDevice));

		//Device side output data allocation
		CHECK_CUDA_ERRORS(cudaMalloc((void** )&deviceOutputData,nt * nttr * sizeof(cufftReal)));

		//make a plan using CUFFT
		//CHECK_CUFFT_ERRORS(cufftPlanMany(&planbwcu, rank, &nt, &inembed, istride, idist, &onembed, ostride, odist, CUFFT_C2R, nttr)) This is not working
		CHECK_CUFFT_ERRORS(cufftPlanMany(&planbwcu, rank, &nt, &onembed, istride, idist,&onembed, ostride, odist, CUFFT_C2R, nttr));

		//fprintf(stderr, "CUFFT INVERSE\n");

		CHECK_CUFFT_ERRORS(cufftExecC2R(planbwcu, deviceInputDataw, deviceOutputData));

		//SCALE THE OUTPUT DATA BY 1/NT
		//Find out the maximum occupancy
		cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,vecScale, 0, 0);

		// Round up according to array size
		gridSize = (arrayCount + blockSize - 1) / blockSize;

		//fprintf(stderr,"gridSize=%d blockSize=%d \n",gridSize,blockSize);

		// calculate theoretical occupancy
		int maxActiveBlocks;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks,vecScale, blockSize,0);
		/*float occupancy = (maxActiveBlocks * blockSize / deviceProp.warpSize) /
						(float)(deviceProp.maxThreadsPerMultiProcessor /
								deviceProp.warpSize); */

		//fprintf(stderr,"maxActiveBlocks=%d Launched blocks of size %d. Theoretical occupancy: %f\n", maxActiveBlocks, blockSize, occupancy);

		//Launch the scaling Kernel
		vecScale<<<gridSize, blockSize>>>(deviceOutputData, scale, arrayCount);
		error_id=cudaGetLastError();
		if (error_id != cudaSuccess) {
			fprintf(stderr,"GPU kernel assert: %s %s %d\n", cudaGetErrorString(error_id), __FILE__, __LINE__);
			assert(0);
		}

		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());  //kernel is guaranteed to finish
		//Can do printing here...

		//Device->Host copy of the results
		CHECK_CUDA_ERRORS(cudaMemcpy(hostOutputData, deviceOutputData,nt * nttr * sizeof(cufftReal), cudaMemcpyDeviceToHost));

		data = (float*)hostOutputData;

		//Destroy the plan
		cufftDestroy(planbwcu);

		//Release device memory
		CHECK_CUDA_ERRORS(cudaFree(deviceInputDataw));
		CHECK_CUDA_ERRORS(cudaFree(deviceOutputData));
	}

	//Release Host memory
	//cudaFree(hostOutputData);

}

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecScale(float *array, float scale, int n) {
	// Get our global thread ID
	int idx = blockIdx.x*blockDim.x + threadIdx.x; //blockId.x: which block is it?
												   //blockDim.x: How many threads there are in a block?
	 	 	 	 	 	 	 	 	 	 	 	   //threadId.x: Which thread it is inside that block?

	 // Make sure we do not go out of bounds
	 if (idx < n)
		 array[idx] *= scale;
}

static const char *_cudaGetErrorEnum(cufftResult error) {
	switch (error) {
	case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";

	case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";

	case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";

	case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";

	case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";

	case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";

	case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";

	case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";

	case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";

	case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";

	case CUFFT_INCOMPLETE_PARAMETER_LIST:
		return "CUFFT_INCOMPLETE_PARAMETER_LIST";

	case CUFFT_INVALID_DEVICE:
		return "CUFFT_INVALID_DEVICE";

	case CUFFT_PARSE_ERROR:
		return "CUFFT_PARSE_ERROR";

	case CUFFT_NO_WORKSPACE:
		return "CUFFT_NO_WORKSPACE";

	case CUFFT_NOT_IMPLEMENTED:
		return "CUFFT_NOT_IMPLEMENTED";

	case CUFFT_LICENSE_ERROR:
		return "CUFFT_LICENSE_ERROR";

	case CUFFT_NOT_SUPPORTED:
		return "CUFFT_NOT_SUPPORTED";

	}

	return "<unknown>";
}
