/*
 * fftop.h
 *
 *  Created on: Aug 5, 2019
 *      Author: mathieu
 */

#ifndef FFTOP_H_
#define FFTOP_H_

//Includes here...
#include <fftw3.h>

//declaration here...
void __host__fft_gather(int fft_type, float scale, fftwf_plan plan, float *data, fftwf_complex *dataw, int nt, int nttr);

#endif /* FFTOP_H_ */
