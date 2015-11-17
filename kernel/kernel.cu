#include <stdlib.h>
#include <helper_cuda.h>
#include "kernel.h"
#include <limits.h>
#include <math.h>


/**
 * The inner computational kernel (will be executed on the GPU).
 *
 * @param       lbDev   Left image (global memory)
 * @param       rbDev   Right image (global memory)
 * @param	zeilen	number of rows of the image
 * @param	spalten	number of columns of the image
 * @param       b       width of the correlation window
 * @param       h       height of the correlation window
 * @param       tauMax  Maximal dispersion
 * @param       useS    Use a threshold (Schwellwert)
 * @param       useF    Use weighted window (please ignore)
 * @param       s       threshold to apply (when useS==true)
 * @param       profileDev Dispersion profile in global memory (to be filled by the algorithm)
 * @param       validDev   Validation matrix in global memory (to be filled by the algorithm)
 */
__global__ void StereoKernel(char *lbDev,char *rbDev, int zeilen, int spalten,int tauMax, int *profileDev,
	      unsigned char *validDev, int b, int h, bool useS, bool useF, int s)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i >= spalten | j >= zeilen)
	{
		// out of bounds
		return;
	}

	int xu = -b / 2;
	int xo = b / 2;
	int yu = -h / 2;
	int yo = h / 2;
	
	int index = i + j * spalten;

	if ((i + xu - tauMax < 0) | (i + xo + tauMax >= spalten) | (j + yu < 0) | (j + yo >= zeilen))
	{
		// invalid
		profileDev[index] = 0;
		validDev[index] = 0;
	}
	else 
	{
		int optWert = INT_MAX;
		int optIndex = 0;
		int val = 0;
		int wert;
		int windowIndex;
		for (int tau = -tauMax; tau <= tauMax; tau++)
		{
			wert = 0;
			for (int k = xu; k <= xo; k++)
			{
				for (int l = yu; l <= yo; l++)
				{
					windowIndex = i+k + (j+l)*spalten;
					wert += abs( (unsigned char) lbDev[windowIndex] - (unsigned char) rbDev[windowIndex +tau] );
				}
			}
			if (wert < optWert)
			{
				optWert = wert;
				optIndex = tau;
				val = 1;
			}
			else if (wert == optWert)
			{
				val = 0;
			}
		}
		if ((useS) && (optWert > s))
		{
			// not in treshold
			val = 0;
		}
		profileDev[index] = optIndex;
		validDev[index] = val;
	}
}

/**
 * Perform the stereodiff algorithm in CUDA.
 *
 * @param       lb      Left image
 * @param       rb      Right image
 * @param	zeilen	number of rows of the image
 * @param	spalten	number of columns of the image
 * @param       b       width of the correlation window
 * @param       h       height of the correlation window
 * @param       tauMax  Maximal dispersion
 * @param       useS    Use a threshold (Schwellwert)
 * @param       useF    Use weighted window (please ignore)
 * @param       s       threshold to apply (when useS==true)
 * @param       profile Dispersion profile (to be filled by the algorithm)
 * @param       valid  Validation matrix (to be filled by the algorithm)
 */
void doCUDACalc(signed char *lb, signed char *rb, int zeilen, int spalten,int tauMax, int *profile,
 unsigned char *valid, int b, int h, bool useS, bool useF, int s)
{


	int num_devices, device;
	cudaGetDeviceCount(&num_devices);
	if (num_devices > 1) {
      		int max_multiprocessors = 0, max_device = 0;
      		for (device = 0; device < num_devices; device++) {
              		cudaDeviceProp properties;
              		cudaGetDeviceProperties(&properties, device);
              		if (max_multiprocessors < properties.multiProcessorCount) {
                      		max_multiprocessors = properties.multiProcessorCount;
                      		max_device = device;
              		}
      		}
      		cudaSetDevice(max_device);
	}

	char *lbDev; // left image in CUDA memory
	char *rbDev; // right image in CUDA memory
	int *profileDev; // profile matrix. to be filled by the algorithm
	unsigned char *validDev; // valid matrix. to be filled by the algorithm

	// allocate global memory for the GPU
	checkCudaErrors(cudaMalloc((void**)&lbDev, zeilen*spalten*sizeof(char)));
	checkCudaErrors(cudaMalloc((void**)&rbDev, zeilen*spalten*sizeof(char)));
	checkCudaErrors(cudaMalloc((void**)&profileDev, zeilen*spalten*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&validDev, zeilen*spalten*sizeof(unsigned char)));

	printf("Copying data to CUDA memory\n");
	checkCudaErrors(cudaMemcpy(lbDev, lb, spalten*zeilen, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rbDev, rb, spalten*zeilen, cudaMemcpyHostToDevice));


	// call the kernel
	dim3 dimBlock (16, 16);
	dim3 dimGrid( (spalten + dimBlock.x - 1) / dimBlock.x, (zeilen + dimBlock.y -1) / dimBlock.y );
	StereoKernel<<< dimGrid , dimBlock >>>(lbDev, rbDev, zeilen, spalten, tauMax, profileDev, validDev, b, h, useS, useF, s) ;


	printf("Copying results back from CUDA memory\n");
	checkCudaErrors(cudaMemcpy(valid, validDev, spalten*zeilen, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(profile, profileDev, spalten*zeilen*sizeof(int), cudaMemcpyDeviceToHost));

	// free memory
	checkCudaErrors(cudaFree(lbDev));
	checkCudaErrors(cudaFree(rbDev));
	checkCudaErrors(cudaFree(profileDev));
	checkCudaErrors(cudaFree(validDev));

        cudaThreadExit();

}


