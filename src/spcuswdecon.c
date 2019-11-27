#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include "su.h"
#include "segy.h"
#include "header.h"
#include "segyhdr.h"
#include <complex.h>  /* standard C complex library */
#include <fftw3.h>
#include <assert.h>

//#include <helper_cuda.h>

/*Include external file*/
#include "../headers/spcuswdecon.h"
#include "../headers/iogather.h"
#include "../headers/fftop.h"
#include "../headers/filter.h"

//global declaration
//Create a typdef alias (gathersInfo)
typedef struct gathersInfo gathersInfo;
gathersInfo gInfo;

void __device__1DMedianReal( gathersInfo *gInfo,float *h_filter);    //musty be here because gatherInfo
void __device__2DMedianReal( gathersInfo *gInfo,float *h_filter);    //musty be here because gatherInfo

char *sdoc[] = {
    "                                                                                                                    ",
    "SPCUSWDECON SEMBLANCE WEIGTED DECONVOLUTION APPLIED TO VSP ARRAY                                                    ",
    "         THE OUTPUT RETURN TWO SEMBLANCE VOLUMES, PSV AND PSH DECONVOLVED DATA                                      ",
    "                                                                                                                    ",
    " key=fldr                                Key header that defines a shot gather                                      ",
    " nc=3                                    Define the number of components                                            ",
    "                                         The input shot gather must be sorted in this manner:                       ",
    "                                         if nc=3 ==>> rcv1[ZXY],rcv2[ZXY],rcv3[ZXY],rcv4[ZXY]....                   ",
    "                                         if nc=4 ==>> rcv1[ZXYH],rcv2[ZXYH],rcv3[ZXYH],rcv4[ZXYH]....               ",
    "                                                                                                                    ",
    " plan=0                                  flag for finding out best FFTW algorithm                                   ",
    "                                         =0  ESTIMATE, more for single test use                                     ",
    "                                         =1  MEASURE, for repeated transform                                        ",
    "                                         =2  PATIENT, for mass production (may lead to segmentation fault)          ",
    " cuda=1                                  =1 use CUFFT, else FFTW algorithm                                          ",
    "                                                                                                                    ",
    " nwin=15                                 window lenght for the estimation of the downgoing wave operator            ",
    NULL
};



int main( int argc, char* argv[] )
{
	/*Variable definition*/
	Value val; /* value of key*/
	segy *gather=NULL, *gatherout=NULL;

	int index,ngather,nt,previous_nt,ntr=0,nttr=0,previous_nttr=0,previous_ntr=0,ntrc,first=0,nc,ntw,plan,flag;
	float dt,previous_dt;

	//FFT variables for FFTW and CUFFT
	int rank=1,inembed={0},onembed={0},istride=1,idist=0,ostride=1,odist=0;

	cwp_String key; /* gather number key */
	cwp_String type; /* type of key	*/

	/* FFTW variables*/
	fftwf_plan mplanfw=NULL;
	//rescaling factor after backward fft
	float fscale=0;

	/*Seismic unix documentation call if no argument given*/
	initargs(argc, argv);
	requestdoc(1);


	///// INPUT KEY PARAMETERS
	//For the key shot gather
	if (!getparstring("key", &key)) key = DEFAULT_GATHER_KEY; /*header key word for storing gather number */
	type = hdtype(key); /* get the gather key type */
	index = getindex(key); /* get its index */

	/* read first gather */
	ngather = 1; /* gather counter */
	gather = get_newgather(&key, &type, &val, &nt, &ntr, &dt, &first); //return ntr,dt and nt
	nttr=ntr;

	if (!getparint("nc", &nc)) nc = DEFAULT_NUM_COMPONENTS;
	fprintf(stderr,"Number of component(s) used: %d\n",nc);

    if (!getparint("plan", &plan)) plan = DEFAULT_FFT_PLAN;
    switch (plan) {
    case 0:
        flag = FFTW_ESTIMATE;
        break;
    case 1:
        flag = FFTW_MEASURE;
        break;
    case 2:
        flag = FFTW_PATIENT;
        break;
    default:
        flag = FFTW_PATIENT;
        break;
    }


    if (!getparint("nwin", &nwin)) nwin = DEFAULT_NWIN;
    if (nwin % 2 == 0) {
        ++nwin;
        warn("window size must be odd number\nnew winsize = %d", nwin);
    }

    //CUDA flag
    if (!getparint("cuda", &cuda)) cuda = DEFAULT_CUDA;

	//ALLOCATE MEMORY FOR THE PROCESSING
	float *data = safe_malloc(sizeof (*data) * nt * nttr);    //input data from stream
	segyhdr *hdrs = safe_malloc(nttr * HDRBYTES);             //trace headers for all traces within gather
	float *datas = safe_malloc(sizeof (*datas) * nt * nttr);  //ouput data from inverse FFT

    fftwf_complex *dataw=NULL;
    cufftComplex *hostOutputDataw=NULL;



	//read all possible subsequent gathers in the memory of the host
	do{
	//	fprintf(stderr,"ngather=%d ntr=%d ntrc=%d nttr=%d \n",ngather,ntr,ntrc,nttr);

		//number of trace per component
	    ntrc = ntr/nc;

		//copy the gather dataset into the data samples and trace headers
		for(int intr=0;intr<ntr;intr++){
			memcpy(&hdrs[intr+(ngather-1)*ntr], &gather[intr], HDRBYTES);  // copy all headers within gather
			memcpy(&data[intr*nt + (ngather-1)*nt*ntr], &gather[intr].data, sizeof(float)*nt);  // copy all samples  within gather
		}
		gather=NULL;

		previous_nttr=nttr;
		previous_ntr = ntr;
		previous_dt = dt;
		previous_nt = nt;

		gather = get_newgather(&key, &type, &val, &nt, &ntr, &dt, &first);
		++ngather;   //gather counter
		nttr+=ntr;   //Accumulated Total Number of Traces read in

		if( previous_ntr != ntr && ntr != 0){
			fprintf(stderr,"The shot gather %d contains different number of traces (%d) than the previous one (%d)\n", ngather,previous_ntr,ntr);
			return EXIT_FAILURE;}

		//fprintf(stderr,"nttr=%d previous_nttr=%d ntr=%d nt=%d\n",nttr,previous_nttr,ntr,nt);

		//Reallocate the memory for larger array
		if(nttr>previous_nttr){
			if(nt){
				warn("Reallocate\n");
				__Reallocate(nttr,nt,&data,&datas,&hdrs);
			}
		}

	}while (ntr);

	//Redefine the basic trace sample information
	dt=previous_dt; nt=previous_nt; ntr=previous_ntr;

	fscale=1.0f/nt;  		//scaling for backward FFTW
	ntw = (int)(nt/2+1);	//Number of frequency elements in the FFT

	//reset the FFT parameters
	idist=nt,ostride=1,odist=ntw;

	warn("nttr=%d\n",nttr);

	//Forward FFT
	if(!cuda){
		dataw = (fftwf_complex*) safe_fftwf_malloc(sizeof (*dataw) * ntw * nttr);
		//do a forward FFT with FFTW on host
	    mplanfw = fftwf_plan_many_dft_r2c(rank,&nt,nttr,data,&inembed,istride,idist,dataw,&onembed,ostride,odist,flag);    	//Make a plan

	    //fftw_plan_many_dft_r2c(int rank(1), const int *n(&nt), int howmany(nttr),
	    //                                 double *in(data), const int *inembed(NULL),
	    //                                 int istride(1), int idist(nt),
	    //                                 fftw_complex *out(dataw), const int *onembed(NULL),
	    //                                 int ostride(1), int odist(ntw),
	    //                                 unsigned flags)


	    warn("FFTW nttr=%d nt=%d ntw=%d\n",nttr,nt,ntw);
		__host__fft_gather(FFTW_FORWARD,1,mplanfw,data,dataw,nt,nttr);

		warn("DONE");

	}
	else{
	    warn("CUFFT FORWARD nttr=%d nt=%d ntw=%d\n",nttr,nt,ntw);
		//Host side output data allocation
	    hostOutputDataw = safe_malloc(ntw * nttr * sizeof(cufftComplex));

		__device__fft_gather(CUFFT_FORWARD,1.f,data,hostOutputDataw,ntw,nttr,nt);

		//point dataw to hostOutputDataw with FFTW type-casting
		dataw = (fftwf_complex*) hostOutputDataw;
	}

	//Backward cuFFT
	hostOutputDataw = (cufftComplex*) dataw;

	__device__fft_gather(CUFFT_INVERSE,fscale,data,hostOutputDataw,ntw,nttr,nt);

	//Hold gather info into structure
	gInfo.Nttr=nttr;
	gInfo.Ntr=ntr;
	gInfo.Nt=nt;
	gInfo.Ncomp=nc;
	gInfo.Ntw=ntw;
	gInfo.Data=data;
	gInfo.Nshot=gInfo.Nttr/gInfo.Ntr;

	int ind;
	//fix the input data, the data in row major
	for(int ishot=0;ishot<gInfo.Nshot;ishot++)
	{
		for(int intr=0;intr<gInfo.Ntr;intr++)
		{
			for(int ins=0;ins<gInfo.Nt;ins++){
				ind = ins+intr*gInfo.Nt+ishot*gInfo.Nt*gInfo.Ntr;
				gInfo.Data[ind] = (float) ind;
				//fprintf(stderr,"index=%i value=%f\n",ind,gInfo.Data[ind]);
			}
		}
	}

	//  Create a 1DMedian kernel for CUDA
	__device__2DMedianReal(&gInfo,datas);

	//write gather to disk
    //gatherout = (segy*)malloc(sizeof(*gatherout));

	//writedata(gatherout, nttr, nt, datas, hdrs);

	//Free memory
	//__FreeMemory(cuda,data,datas,hdrs,mplanfw,dataw);
	free(gather); gather=NULL;

	return EXIT_SUCCESS;
}

void *safe_malloc(size_t n){
	void *p = malloc(n);
	if(p == NULL){
		fprintf(stderr,"Cannot Malloc @ %s %d\n",__FILE__,__LINE__); p=NULL;
		assert(0);
	}
	return p;
}

void *safe_fftwf_malloc(size_t n){
	void *p = fftwf_malloc(n);
	if(p == NULL){
		fprintf(stderr,"Cannot fftwf_malloc @ %s %d\n",__FILE__,__LINE__); p=NULL;
		assert(0);
	}
	return p;
}

void __Reallocate(int NTTR, int NT, float **data, float **datas, segyhdr **hdrs){
//NTTR : Accumulated Number Total TRaces in the dataset
//NT : Number of Time Sample per trace

    //Trace headers
    segyhdr *temphdrs = (segyhdr*) realloc(*hdrs,NTTR * HDRBYTES);
    if(temphdrs == NULL){
        warn("Cannot reallocate memory @ temphdrs from(%s,%d)\n",__FILE__,__LINE__); temphdrs=NULL; exit(EXIT_FAILURE);
        free(temphdrs);
        temphdrs=NULL;
        exit(EXIT_FAILURE);
    }
    *hdrs=temphdrs;

    /*REAL ARRAY*/
	float *tempdata = (float*) realloc(*data,sizeof(float)*NT*NTTR);
	if(tempdata == NULL){
		warn("Cannot reallocate memory @ tempdata from(%s,%d)\n",__FILE__,__LINE__); tempdata=NULL; exit(EXIT_FAILURE);
		free(tempdata);
		tempdata=NULL;
		exit(EXIT_FAILURE);
	}
	*data=tempdata;

	float *tempdatas = (float*) realloc(*datas,sizeof(float)*NT*NTTR);
	if(tempdatas == NULL){
		warn("Cannot reallocate memory @ tempdatas from(%s,%d)\n",__FILE__,__LINE__); tempdatas=NULL; exit(EXIT_FAILURE);
		free(tempdatas);
		tempdatas=NULL;
		exit(EXIT_FAILURE);
	}
	*datas=tempdatas;

	return;
}

void __FreeMemory(int cuda, float *data, float *datas, segyhdr *hdrs,fftwf_plan mplanfw,fftwf_complex *dataw){
    fftwf_destroy_plan(mplanfw);

	free(data); data=NULL;
	free(hdrs); hdrs=NULL;
	fftwf_free(dataw); dataw=NULL;
	free(datas); datas=NULL;

	if(cuda){

	}
	else{
	}


	return;
}


