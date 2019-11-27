#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>


//external include files
#include "../headers/filter.h"
#include "../headers/errormessage.h"


//Set the blockSize for each block
struct blockSize
{
	int nthreadx;     //How many trace there is per gather?
	int nthready;     //How many thread per block in the time axis?
	int nthreadz;
	int nthreadxyz;
};

//Set the gridSize
struct gridSize
{
	int nblockx;     //How many shot gather there are in the data?
	int nblocky;     //How many time sample there are in the data?
	int nblockz;
};

typedef struct blockSize blockSize;
typedef struct gridSize gridSize;


__global__ void hello(){

    printf("FILTER2.CU world I am thread in blockx %d, threadIdx.x=%d blockDim.x=%d gridDim.x=%d  blocky %d, threadIdx.y=%d blockDim.y=%d gridDim.y=%d \n",blockIdx.x,threadIdx.x,blockDim.x,gridDim.x,blockIdx.y,threadIdx.y,blockDim.y,gridDim.y);
    return ;
}

__global__ void hello_printmessage(){

    printf("HELLO_PRINTMESSAGE world I am thread in blockx %d, threadIdx.x=%d blockDim.x=%d gridDim.x=%d  blocky %d, threadIdx.y=%d blockDim.y=%d gridDim.y=%d \n",blockIdx.x,threadIdx.x,blockDim.x,gridDim.x,blockIdx.y,threadIdx.y,blockDim.y,gridDim.y);
    return ;
}


extern "C" void __printmessage(){

	hello_printmessage<<<1,10>>>();
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());		//important for printing messages inside the kernel!

}

__device__ int getGlobalIndexPos(int x, int y, int width){
	return x + y*width;
//	return (gridDim.x * blockDim.x * y) + x;
}


//Median Filter with CUDA extension
__global__ void CudaMedianFilterFloat(float *d_data, float *d_filter, gathersInfo gInfo) {

	//extern __shared__ int s_data[];
	//local variable (independent to each thread)
	int globalIndexPos,localIndexPos;
	int2 globalIdx;

	//Assume input is row-major contiguous
	globalIdx.x = blockIdx.x * blockDim.x + threadIdx.x;
	globalIdx.y = blockIdx.y * blockDim.y + threadIdx.y;


	int ntr = gInfo.Ntr;
	ntr = 8;

	globalIndexPos = getGlobalIndexPos(globalIdx.x,globalIdx.y,ntr);
	printf("%i blockIdx( %i , %i ) threadIdx( %i , %i )  globalIdx.x= %i globalIdx.y= %i globalIndexPos= %i blockDim( %i , %i ) gridDim( %i ,%i ) \n",globalIndexPos,blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,globalIdx.x, globalIdx.y,globalIndexPos,blockDim.x,blockDim.y,gridDim.x,gridDim.y);
	__syncthreads();

	//d_filter[globalIndexPos]=d_data[globalIndexPos];

	//printf("globalIndexPos=%i globalIdx.x=%i globalIdx.y=%i \n",globalIndexPos,globalIdx.x,globalIdx.y);

	//if((globalIdx.x < gInfo->Nttr) && ( globalIdx.y < )){
	//globalIndexPos = getGlobalIndexPos(globalIdx.x,globalIdx.y,gInfo->Nt);

	//}

	/*int2 localIdx;
	localIdx.x = threadIdx.x;
	localIdx.y = threadIdx.y;

	localIndexPos = getLocalIndexPos(localIdx.x,localIdx.y,gInfo->Nt);

	//s_data[localIndexPos] = d_data[globalIndexPos];

	__syncthreads();*/

	//d_filter[globalIndexPos] =s_data[localIndexPos];

	//printf("%i h_in[%i]=%i threadIdx.x=%i blockIdx.x=%i blockDim.x=%i  \n",index,index, s_in[index],threadIdx.x,blockIdx.x,blockDim.x);

}

__global__ void transposeCoalesced(float *idata, float *odata, gathersInfo gInfo, int TILE_DIM, int TILE_DIMY, int nx)
{
  __shared__ float tile[3][3];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  //printf("gridDim.x= %i \n",gridDim.x);

  for (int j = 0; j < TILE_DIM; j += TILE_DIMY){
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];
     //printf("tile %f idata[%i] %f x=%i y=%i threadIdx ( %i %i )  blockIdx ( %i %i )\n",tile[threadIdx.y+j][threadIdx.x],(y+j)*width + x,idata[(y+j)*width + x],x,y,threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y);
  }


  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += TILE_DIMY){
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
     printf("tile %f odata[%i] %f x=%i y=%i threadIdx ( %i %i )  blockIdx ( %i %i )\n",tile[threadIdx.x][threadIdx.y+j],(y+j)*width + x,odata[(y+j)*width + x],x,y,threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y);
  }
}

__global__ void transposeNaive(float *idata, float *odata, gathersInfo gInfo, int TILE_DIM, int TILE_DIMY, int nx)
{
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;
	int globalIndexPos = x + y*nx;

	printf("%i blockIdx( %i , %i ) threadIdx( %i , %i )  globalIdx.x= %i globalIdx.y= %i blockDim( %i , %i ) gridDim( %i ,%i ) \n",globalIndexPos, blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,threadIdx.x, threadIdx.y,blockDim.x,blockDim.y,gridDim.x,gridDim.y);

	for (int j = 0; j < TILE_DIM; j+= TILE_DIMY)
		odata[x*width + (y+j)] = idata[(y+j)*width + x];

	return;
}

__global__ void copySharedMem(float *odata, float *idata, int nx, int ny, int TILE_DIM, int TILE_DIMY, int width, int height){
	// tile[ROW][COLUMN] = tile[icol+irow*ncol]  this is raw major alignement
	__shared__ float tile[3][3];
	//extern __shared__ float tile[];
	int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
	int index = xIndex + width*yIndex;

	for (int i=0; i<TILE_DIM; i+=TILE_DIMY) {
		tile[threadIdx.y+i][threadIdx.x] = idata[index+i*width];
		//printf("i=%i xIndex= %i yIndex= %i threadIdx=( %d , %d ) blockIdx =( %d , %d ) gridDim =( %d , %d) tile[%i][%i]=%f \n",i, xIndex,yIndex,threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,gridDim.x,gridDim.y,threadIdx.y+i,threadIdx.x,tile[threadIdx.y+i][threadIdx.x]);
	}
	__syncthreads();

	for (int i=0; i<TILE_DIM; i+=TILE_DIMY) {
		odata[index+i*width] = tile[threadIdx.y+i][threadIdx.x];
	}

	/*tile[threadIdx.y][threadIdx.x] = idata[index];
	__syncthreads();
	odata[index] = tile[threadIdx.y][threadIdx.x];*/

}

__global__ void transposeCoarseGrained(float *odata, float *idata, int nx, int ny, int TILE_DIM, int TILE_DIMY, int width, int height){
	__shared__ float block[3][3+1];
	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;
	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

	int index_out = xIndex + (yIndex)*height;
	for (int i=0; i<TILE_DIM; i += TILE_DIMY) {
		block[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
	}

	__syncthreads();
	for (int i=0; i<TILE_DIM; i += TILE_DIMY) {
		odata[index_out+i*height] = block[threadIdx.y+i][threadIdx.x];
	}
}

__global__ void transposeFineGrained(float *odata, float *idata, int nx, int ny, int TILE_DIM, int TILE_DIMY, int width, int height){
//fine-grained transpose: this kernel transposes the data within a tile, but writes the tile to the location

	__shared__ float tile[3][3+1];
	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	int index = xIndex + (yIndex)*nx;
	for (int i=0; i < TILE_DIM; i += TILE_DIMY){
		tile[threadIdx.y+i][threadIdx.x] = idata[index+i*nx];
		//tile[threadIdx.y+i][threadIdx.x] = idata[index+i*nx];
	}
	__syncthreads();
	for (int i=0; i < TILE_DIM; i += TILE_DIMY) {
		odata[index+i*ny] = tile[threadIdx.x][threadIdx.y+i];  //correct
	}
}

__global__ void transposeDiagonal(float *odata, float *idata, int nx, int ny, int TILE_DIM, int TILE_DIMY,int width, int height)
{
	//__shared__ float tile[TILE_DIM][TILE_DIM+1];
	extern __shared__ float tile[];
	int blockIdx_x, blockIdx_y;
	int2 globalIdx;

	// diagonal reordering
	if (nx == ny) {  //square matrix
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;   // change when the
		//printf("blockIdx_x= %i blockIdx_y= %i threadIdx=( %d , %d ) blockIdx =( %d , %d ) gridDim =( %d , %d) \n",blockIdx_x,blockIdx.y,threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,gridDim.x,gridDim.y);
	}
	else {
		int bid = blockIdx.x + gridDim.x*blockIdx.y;
		blockIdx_y = bid%gridDim.y;
		blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
	}

	globalIdx.x = blockIdx_x*TILE_DIM + threadIdx.x;   //index along the X axis
	globalIdx.y = blockIdx_y*TILE_DIM + threadIdx.y;   //index along the Y axis
	int index_in=getGlobalIndexPos(globalIdx.x,globalIdx.y,nx); //index of the input data

	globalIdx.x = blockIdx_y*TILE_DIM + threadIdx.x;  //swap index
	globalIdx.y = blockIdx_x*TILE_DIM + threadIdx.y;

	int index_out=getGlobalIndexPos(globalIdx.x,globalIdx.y,ny);

	for (int itrx=0; itrx<TILE_DIM; itrx+=TILE_DIMY) {
		//copy the data into the shared memory
		//tile[threadIdx.y+itrx][threadIdx.x] =	idata[index_in+itrx*nx];
		if(width > nx || height > ny){
			//padding zero
			tile[(threadIdx.y+itrx) + threadIdx.x*TILE_DIM] = 0;
		}
		else{
			tile[(threadIdx.y+itrx) + threadIdx.x*TILE_DIM] =	idata[index_in+itrx*nx];
		}
	}
	__syncthreads();

	for (int itrx=0; itrx<TILE_DIM; itrx+=TILE_DIMY) {
		//odata[index_out+itrx*height] = tile[threadIdx.x][threadIdx.y+itrx];
		odata[index_out+itrx*ny] = tile[threadIdx.x + (threadIdx.y+itrx)*TILE_DIM];
	}
    //printf("transposeDiagonal blockx %d, threadIdx.x=%d blockDim.x=%d gridDim.x=%d  blocky %d, threadIdx.y=%d blockDim.y=%d gridDim.y=%d \n",blockIdx.x,threadIdx.x,blockDim.x,gridDim.x,blockIdx.y,threadIdx.y,blockDim.y,gridDim.y);


}

__global__ void transposeCoalesced(float *odata, float *idata, int nx, int ny, int TILE_DIM, int TILE_DIMY,int width, int height)
{
    __shared__ float tile[3][3];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

	for (int i=0; i<TILE_DIM; i+=TILE_DIMY)
	{
		tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
	}

	/*__syncthreads();

	for (int i=0; i<TILE_DIM; i+=TILE_DIMY)
	{
		odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
	}*/

    //tile[threadIdx.y][threadIdx.x] = idata[index_in];
	//printf("xIndex= %i yIndex= %i threadIdx=( %d , %d ) blockIdx =( %d , %d ) gridDim =( %d , %d) tile[%i][%i]= %f \n",xIndex,yIndex,threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,gridDim.x,gridDim.y,threadIdx.y,threadIdx.x,tile[threadIdx.y][threadIdx.x]);

    __syncthreads();

//	for (int i=0; i<TILE_DIM; i+=TILE_DIMY)
//	{
		odata[index_in] = tile[threadIdx.x][threadIdx.y];
//	}

}

__global__ void transposeCoalescedRect(float *odata, float *idata, int nx, int ny, int TILE_DIM, int TILE_DIMY,int width, int height, bool Msquare)
{
    __shared__ float tile[3][3];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

	/*for (int i=0; i<TILE_DIM; i+=TILE_DIMY)
	{
		tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
	}*/

	/*__syncthreads();

	for (int i=0; i<TILE_DIM; i+=TILE_DIMY)
	{
		odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
	}*/

	if(ny<TILE_DIMY*blockIdx.y){
		//Find out the blockIdx.y where the shared memory must be set to zero
		printf("height=%i ny=%i TILE_DIM=%i Out of defined matrix blockIdx ( %i %i ) TILE_DIMY*blockIdx.y= %i \n",height, ny,TILE_DIMY,blockIdx.x,blockIdx.y,TILE_DIMY*blockIdx.y);
	}
	else{
		tile[threadIdx.y][threadIdx.x] = idata[index_in];
	}
	//printf("xIndex= %i yIndex= %i threadIdx=( %d , %d ) blockIdx =( %d , %d ) gridDim =( %d , %d) tile[%i][%i]= %f \n",xIndex,yIndex,threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,gridDim.x,gridDim.y,threadIdx.y,threadIdx.x,tile[threadIdx.y][threadIdx.x]);

    __syncthreads();

//	for (int i=0; i<TILE_DIM; i+=TILE_DIMY)
//	{
		odata[index_in] = tile[threadIdx.x][threadIdx.y];
//	}

}

bool __isSquareMatrix(int nx,int ny, int *width, int *height){
    if(nx > ny){
    	*width=nx;
    	*height = (nx-ny)+ny;
    	return false;
    }
    else if(nx < ny){
    	*height=ny;
    	*width=(ny-nx)+ny;
    	return false;
    }
    else{
    	*width=nx; *height=ny;
    	return true;
    }
}

extern "C" void __device__1DMedianReal( gathersInfo *gInfo,float *h_filter){

	int nblocks=1, nthreads=10;
	hello<<<nblocks,nthreads>>>();
	hello_printmessage<<<nblocks,nthreads>>>();
    cudaDeviceSynchronize();

    //free(h_in);free(h_out);   Maybe not !!

}

extern "C" void __device__2DMedianReal( gathersInfo *gInfo,float *h_filter){
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t error_id;
	float shared_mem;

	//data is row-major sorted. First axis (time)
	//second axis (trace number index)
	//square matrix parameters
	//int nx=9,ny=9,TILE_DIM=3,TILE_DIMY=3,TILE_DIM_EXT,width,height;
	//int nx=1024,ny=1024,TILE_DIM=32,TILE_DIMY=32,TILE_DIM_EXT,width,height;

	int nx=9,ny=9,TILE_DIM=3,TILE_DIMY=3,TILE_DIM_EXT,width,height;

	//Device property
	int devCount,dev=0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceCount(&devCount);
	cudaSetDevice(dev);

	CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&deviceProp, devCount-1));
	fprintf(stderr,"CUDA Device Query...\n");
	fprintf(stderr,"There are %d CUDA devices. Device name: %s\n", devCount,deviceProp.name);

	//Allocation memory
	size_t bytes = (nx)*(ny)*sizeof(float);

	//Host input
	float *input = (float*) malloc(bytes);
	// Fill the array with sequencial number
	int ind=0; float val;
	fprintf(stderr,"input:\n");
	for(int iny=0;iny<ny;iny++)
	{
		for(int inx=0;inx<nx;inx++){
			ind = inx+iny*nx;
			val = (float)ind;
			input[ind] = val;
			fprintf(stderr,"%5.0f",input[ind]);
		}
		fprintf(stderr,"\n");
	}
	float *h_data = input;

	//float *h_data = gInfo->Data;

	// Device input vectors
	float* d_data, *d_filter;

	//Is square matrix?
	bool Msquare = __isSquareMatrix(nx,ny,&width,&height);

	if(Msquare){
		TILE_DIM_EXT = (int)sqrt((float)width);
		//increase shared memory
		shared_mem=sizeof(float)*TILE_DIM_EXT*(TILE_DIM_EXT+1);
		fprintf(stderr,"The matrix is squared\n");
	}
	else{
		shared_mem=sizeof(float)*TILE_DIM*(TILE_DIM+1);
		fprintf(stderr,"The matrix is non-squared\n");
	}

	//Number of threads in each block
	blockSize blocksize;
	blocksize.nthreadx=nx/TILE_DIM;  	//maximum is 1024!
	blocksize.nthready=ny/TILE_DIM;		//gInfo->Ntr;
	blocksize.nthreadz=1;
	blocksize.nthreadxyz = blocksize.nthreadz*blocksize.nthready*blocksize.nthreadx;

	//Make sure the maximum #thread/block is not reached
	if(blocksize.nthreadxyz > deviceProp.maxThreadsPerBlock){
		fprintf(stderr,"The number of threads / block (%d) is not supported (maximum=%d)\n",blocksize.nthreadxyz,deviceProp.maxThreadsPerBlock);
		exit(0);
	}


    //Number of blocks in each grid dimension
    gridSize gridsize;
    gridsize.nblockx=TILE_DIM;
    gridsize.nblocky=TILE_DIMY; //gInfo->Nshot;
    gridsize.nblockz=1;

    dim3 nblocks(gridsize.nblockx,gridsize.nblocky,gridsize.nblockz);
    dim3 nthreads(blocksize.nthreadx,blocksize.nthready,blocksize.nthreadz);

    fprintf(stderr,"\n\nnblocks(%i,%i,%i) nthreads(%i,%i,%i)\n",nblocks.x,nblocks.y,nblocks.z,nthreads.x,nthreads.y,nthreads.z);

	//<<<**************kernel parameters**************
	//Shared memory
    shared_mem=sizeof(float)*gridsize.nblockx*gridsize.nblocky;


    //<<<************Allocate memory for each vector on GPU********************
    CHECK_CUDA_ERRORS(cudaMalloc(&d_data, bytes));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_filter, bytes));
    //>>>************Allocate memory for each vector on GPU********************

    CHECK_CUDA_ERRORS(cudaMemset(d_data, 0, bytes));
    CHECK_CUDA_ERRORS(cudaMemset(d_filter, 0, bytes));

    //<<<************Copy host vectors to device******************
    CHECK_CUDA_ERRORS(cudaMemcpy( d_data, h_data, bytes, cudaMemcpyHostToDevice));
    //>>>************Copy host vectors to device******************

    cudaEventRecord(start);
    transposeNaive<<<nblocks,nthreads>>>(d_data,d_filter,*gInfo,TILE_DIM,TILE_DIMY,nx);
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);

    error_id=cudaGetLastError();
	if (error_id != cudaSuccess) {
		fprintf(stderr,"GPU kernel assert: %s %s %d\n", cudaGetErrorString(error_id), __FILE__, __LINE__);
		assert(0);
	}

    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());		//important for printing messages inside the kernel!

	int nblock=1, nthread=10;
	hello<<<nblock,nthread>>>();
    cudaDeviceSynchronize();

}


