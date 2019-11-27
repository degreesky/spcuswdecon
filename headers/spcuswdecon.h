/*
 * spcuswdecon.h
 *
 *  Created on: Jul 31, 2019
 *      Author: mathieu
 */

#ifndef SPCUSWDECON_H_
#define SPCUSWDECON_H_

/*Includes here*/

/*Declarations here*/
static int cuda,nwin;
#define DEFAULT_GATHER_KEY "fldr"
#define DEFAULT_NUM_COMPONENTS 3
#define DEFAULT_FFT_PLAN 0
#define DEFAULT_CUDA 1
#define DEFAULT_NWIN 15

void __FreeMemory(int cuda,float *data, float *datas, segyhdr *hdrs,fftwf_plan mplanfw,fftwf_complex *dataw);
void __Reallocate(int NTR, int NT, float **data, float **datas, segyhdr **hdrs);
void *safe_malloc(size_t n);
void *safe_fftwf_malloc(size_t n);

void __device__fft_gather(int fft_type, float scale, float *data, cufftComplex *dataw, int ntw, int nttr, int nt);
//void __printmessage(void);


#endif /* SPCUSWDECON_H_ */
