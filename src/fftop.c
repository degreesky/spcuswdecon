/*
 * fftop.c
 *
 *  Created on: Aug 5, 2019
 *      Author: mathieu
 */

//comments here...
//includes here...
#include "../headers/fftop.h"

//function here...

void __host__fft_gather(int fft_type, float scale, fftwf_plan plan, float *data, fftwf_complex *dataw, int nt, int nttr) {
    int rowcounter = 0;

    if (fft_type == FFTW_FORWARD) {
        fftwf_execute_dft_r2c(plan, data, dataw);

    } else {
        fftwf_execute_dft_c2r(plan, dataw, data);
            for ( int intr = 0; intr < nttr; ++intr) {
                for (int ins = 0; ins < nt; ++ins) {
                    data[rowcounter + ins] *= scale;
                }
                rowcounter += nt;
            }
    }
    return;
}
