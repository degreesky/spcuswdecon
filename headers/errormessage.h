/*
 * errormessage.h
 *
 *  Created on: Sep 9, 2019
 *      Author: mathieu
 */

#ifndef ERRORMESSAGE_H_
#define ERRORMESSAGE_H_

#include <cufft.h>
#include <assert.h>

// Functions here...

inline cufftResult_t CHECK_CUFFT_ERRORS(cufftResult_t err)
{
	#if defined(DEBUG) || defined(_DEBUG)
	if(err != CUFFT_SUCCESS){
        fprintf(stderr, "cuFFT error %d:%s at %s:%d\n", err, _cudaGetErrorEnum(err), __FILE__, __LINE__);
        assert(result == cudaSuccess);
	}
#endif
  return err;
}

static const char *_cudaGetErrorEnum(cufftResult error);

inline cudaError_t CHECK_CUDA_ERRORS(cudaError_t result)
{
	#if defined(DEBUG) || defined(_DEBUG)
	  if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s %s %d\n", cudaGetErrorString(result), __FILE__, __LINE__);
		assert(result == cudaSuccess);
	  }
	#endif
	  return result;
}


#endif /* ERRORMESSAGE_H_ */
