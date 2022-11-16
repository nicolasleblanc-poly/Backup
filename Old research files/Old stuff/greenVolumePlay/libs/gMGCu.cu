// Definitions of sysInitO, volInitO, blcMemId, blcMemCpy, sysVecCpy, fftExeO have changed
// sysVecCpy has become sysVecCpyO
// fftExeO no longer synchronizes
// joinCurrO has been replaced by joinCurrO
// variables required by sysInitO have been modified.
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cufft.h>
#include "gMGCu.h"
// blsNum: number of BLAS streams created for a given block.  
#define blsNum 3
// blcWorkAreas: number of memory locations created for embedding and projecting on a block.
#define blcWorkAreas 2
// fftNum: number of fft streams created for a block. 
#define fftNum 3
// The three levels of memory referenced in the definitions given below are 
// 1. System---multiple distinct systems setups.
// 2. Volume---the division existing within a system. 
// 3. Blocks---the resulting blocks (matrix blocks) of the Green functions for a collection volumes.  

// Given that a system may be freely divided, each volume is supposed to exist on a single device. 
// All blocks in the corresponding to a volume row should be stored on the same device to reduce communication overhead. 

// *** Block level memory allocations. 
// fftPlans: Storage for FFT plans.
// fftStream: CUDA control variable for FFT computations. 
// blcStream: CUDA control variable for a Green function block. 
// blcWork: Workspace for embedding current into circulant forms and performing Fourier transforms.
// blcSum: Memory for summing outputs within a common volume. 
// blcCurrSrc: Memory for input to a unique block of the Green function.
// blcCurrTrg: Memory for output from a unique block of the Green function.
// blcGreenSlf: ``Self'' orientation elements of a Green function block.  
// blcGreenXrs: ``Cross'' orientation elements of a Green function block. 
// blcOnesVec: Vector of ones, used for summing as part of the cuBLAS cublasZgemv operation. 
// blcCells: Cells in a given Green function (block). The three positions hold the number of source, target, and circulant cells.
cufftHandle ***fftPlans;
cudaStream_t ***fftStream;
cudaStream_t **blcStream;
cufftDoubleComplex ****blcWork;
cufftDoubleComplex ***blcSum;
cufftDoubleComplex ***blcCurrSrc;
cufftDoubleComplex ***blcCurrTrg;
cufftDoubleComplex ***blcGreenSlf;
cufftDoubleComplex ***blcGreenXrs;
cufftDoubleComplex ***blcOnesVec;
int ****blcCells;
// *** Volume level memory allocations (row of the Green function matrix).
// volWork: Work space for summing results from associated Fourier streams.
// volCurrSrc: Portion of source system current in a given volume.
// volCurrTrg: Portion of target system current in a given volume.
// volCells: Number of cells in a given volume. The two positions hold the number of source cells and target cells.
// volOnesVec: Vector of ones, used for summing as part of the cuBLAS cublasZgemv operation. 
cudaStream_t **volStream;
cufftDoubleComplex ****volWork;
cufftDoubleComplex ***volCurrSrc;
cufftDoubleComplex ***volCurrTrg;
cufftDoubleComplex ***volOnesVec;
int ****volCells;
// *** System level memory allocations.
// sysCurrSrc: full input current of a system. 
// sysCurrTrg: full output field of a system.
// numDevs[sysId]: Number of devices associated with system.
// devList: List of numbered identifiers for each system device.
// domains[sysId]: Number of domains in the system.
// blocks[sysId]: Number of Green function block in the system.
// sysCells: Number of cells on the source and target sides of a system---covering multiple volumes.
// blocksCUDA: CUDA computation setting, see CUDA toolkit documentation.
// threadsPerBlc: CUDA computation setting, see CUDA toolkit documentation.
// volStreamSplit: Holds number of volume streams on a given device.
// blcStreamSplit: Holds number of block streams on a given device.
// volStreamDev: Device location of each volume stream.
// blcStreamDev: Device location of each Fourier stream.
// adjSetting: Switch between system and its adjoint.
// slfSetting: Setting to allow code to be used for partial Green function interactions. 
cuDoubleComplex **sysCurrSrc;
cuDoubleComplex **sysCurrTrg;
int *numDevs;
int **devList;
int *domains;
int *blocks;
int **sysCells;
int blocksCUDA;
int threadsPerBlc;
int **volStreamSplit;
int **blcStreamSplit;
int **volStreamDev;
int **blcStreamDev;
int *adjSetting;
int *slfSetting;
// Asynchronous controls for volume computations. 
cublasHandle_t ***volBlasHandle;
cudaStream_t ***volBlasStream;
// Asynchronous controls for block computations. 
cublasHandle_t ***blcBlasHandle;
cudaStream_t ***blcBlasStream;
// *** Other conventions
// In reference to the Green function vectors of a particular block, the Slf and Xrs modifiers are used to indicate interaction along a repeated Cartesian direction and distinct Cartesian directions respectively.

// *** Support device kernels. 
// A terminating O is used to indicate that the function overwrites one the memory locations it is given. 
// Conjugate a complex array. 
__global__
void conjArrO(cufftDoubleComplex *memLoc, int numElements)
{
	int localId = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for (int i = localId; i < numElements; i += stride)
	{
		memLoc[i].y = -1.0 * memLoc[i].y;
	}
	return;		
}
// Set all elements of an array to zero. 
__global__ 
void zeroArrO(cufftDoubleComplex *memLoc, int numCells)
{ 
	int localId = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for (int i = localId; i < 3 * numCells; i += stride)
	{
		memLoc[i].x = 0.0;
		memLoc[i].y = 0.0;
	}
	return;	
}
// Set all elements of an array to one. 
__global__ 
void oneArrO(cufftDoubleComplex *memLoc, int numElements)
{ 
	int localId = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for (int i = localId; i < 3 * numElements; i += stride)
	{
		memLoc[i].x = 1.0;
		memLoc[i].y = 0.0;
	}
	return;	
}
// Elementwise multiplication of two arrays. The target memory location can not overlap with either of the source memory locations. If this functionality is ever needed, the kernel must be modified.
__global__
void elewiseMultO(int numElements, const cufftDoubleComplex *vec1, const cufftDoubleComplex *vec2, cufftDoubleComplex *prodVec)
{
	int localId = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for (int i = localId; i < numElements; i += stride)
	{
		prodVec[i].x = vec1[i].x * vec2[i].x - vec1[i].y * vec2[i].y;
		prodVec[i].y = vec1[i].y * vec2[i].x + vec1[i].x * vec2[i].y;
	}
	return;		
}
// Overwrite target memory location with the difference of the source and the target. 
__global__
void arrDiffO(int mode, cufftDoubleComplex *currSrc, cufftDoubleComplex *currTrg, int numElements)
{
	int localId = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	if(mode == -1)
	{
		for (int i = localId; i < numElements; i += stride)
		{
			currTrg[i].x = currSrc[i].x - currTrg[i].x;
			currTrg[i].y = currSrc[i].y - currTrg[i].y;
		}	
	}
	else
	{
		for (int i = localId; i < numElements; i += stride)
		{
			currTrg[i].x = currTrg[i].x - currSrc[i].x;
			currTrg[i].y = currTrg[i].y - currSrc[i].y;
		}	
	}
	return;
}
// Embed a source vector into a circulant form.
__global__
void embedVecO(cufftDoubleComplex *sourceMemLoc, int numEleCrc, cufftDoubleComplex *circMemLoc, int sourceCellsX, int sourceCellsY, int sourceCellsZ, int circCellsX, int circCellsY, int circCellsZ)
{
	int cellIndSrc, cellX, cellY, cellZ;
	int stride = gridDim.x * blockDim.x;
	int localId = threadIdx.x + blockIdx.x * blockDim.x;
	
	for (int i = localId; i < numEleCrc; i += stride)
	{	
		cellX = i % circCellsX;
		cellY = ((i - cellX) % (circCellsX * circCellsY)) / circCellsX;
		cellZ = (i - cellX - cellY * circCellsX) / (circCellsX * circCellsY);

		if ((cellX < sourceCellsX) && (cellY < sourceCellsY) && (cellZ < sourceCellsZ))
		{	
			cellIndSrc = cellX + (cellY * sourceCellsX) + (cellZ * sourceCellsX * sourceCellsY);
			circMemLoc[i].x = sourceMemLoc[cellIndSrc].x;
			circMemLoc[i].y = sourceMemLoc[cellIndSrc].y;
		}
		else
		{
			circMemLoc[i].x = 0.0;
			circMemLoc[i].y = 0.0;
		}
	}
	return;
}
// Project a vector out of a circulant form.
__global__
void projVecO(cufftDoubleComplex *circMemLoc, cufftDoubleComplex *targetMemLoc, int circCellsX,int circCellsY, int circCellsZ, int targetCellsX, int targetCellsY, int targetCellsZ)
{
	int cellIndTrg, cellX, cellY, cellZ;
	int stride = gridDim.x * blockDim.x;
	int localId = threadIdx.x + blockIdx.x * blockDim.x;
	int numCellsCrc = circCellsX * circCellsY * circCellsZ;

	for (int i = localId; i < numCellsCrc; i += stride)
	{
		cellX = i % circCellsX;
		cellY = ((i - cellX) % (circCellsX * circCellsY)) / circCellsX;
		cellZ = (i - cellX - cellY * circCellsX) / (circCellsX * circCellsY);
		
		if((cellX < targetCellsX) && (cellY < targetCellsY) && (cellZ < targetCellsZ))
		{
			cellIndTrg = cellX + cellY * targetCellsX + cellZ * targetCellsX * targetCellsY; 
			targetMemLoc[cellIndTrg].x = circMemLoc[i].x;
			targetMemLoc[cellIndTrg].y = circMemLoc[i].y;
		}
	}
	return;
}
// *** General support functions. 
// Function for examining device data. Otherwise unessential.
extern "C"{__host__
	void gPrint(int offset, int rows, int cols, cuDoubleComplex *gArray, int devNum)
	{
		int rowNum;
		int colNum;
		cuDoubleComplex *hostMem;
		hostMem = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * rows * cols);

		if(cudaSetDevice(devNum) != cudaSuccess)
		{
			fprintf(stderr, "gPrint CUDA Error: Failed to switch to device %d.\n", devNum);
			return;
		}
		if(cudaDeviceSynchronize() != cudaSuccess)
		{
			fprintf(stderr, "gPrint CUDA Error: Failed to synchronize device %d.\n", devNum);
			return;	
		}
		if(cudaMemcpy(hostMem, &(gArray[offset]), sizeof(cuDoubleComplex) * rows * cols, cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			fprintf(stderr, "gPrint CUDA Error: Failed to copy data to host.\n");
			return;	
		}	

		fprintf(stdout, "\n");

		for(int itr = 0; itr < rows * cols; itr++)
		{
			colNum = itr % cols;
			rowNum = itr / cols;

			if((itr + 1) % cols == 0)
			{
				fprintf(stdout, "%8.7f+%8.7fi\n", hostMem[rowNum + colNum * rows].x, 
					hostMem[ rowNum + colNum * rows].y);
			}
			else
			{
				fprintf(stdout, "%8.7f+%8.7fi ", hostMem[rowNum + colNum * rows].x, 
					hostMem[rowNum + colNum * rows].y);
			}
		}
		fprintf(stdout, "\n");
		free(hostMem);
		return;
	}
}
// Reset all devices in a given system. 
extern "C"{__host__
	void fullReset(int sysId)
	{
		for(int devId = 0; devId < numDevs[sysId]; devId++)
		{
			if(cudaSetDevice(devList[sysId][devId]) != cudaSuccess)
			{
				fprintf(stderr, "fullReset CUDA Error: Failed to switch to device %d.\n", devId);
				return;
			}	

			if(cudaDeviceReset() != cudaSuccess)
			{
				fprintf(stderr, "fullReset CUDA Error: Failed to reset device %d.\n", devId);
				return;
			}
		}		
		return;
	}
}
// Scale an array by a real constant.  
extern "C"{__host__
	int scaleArrO(int sysId, int blcId, int blsId, cufftDoubleComplex *memLoc, double rScale, int numElements)
	{ 
		cuDoubleComplex scale;
		scale.x = rScale;
		scale.y = 0.0;

		if(cublasZscal(blcBlasHandle[sysId][blcId][blsId], numElements,
			&scale, (cuDoubleComplex*)memLoc, 1) != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "scaleArrO CUBLAS Error: Failed to scale array BLAS stream %d block %d.\n", blsId, blcId);
			return 1;
		}
		return 0;	
	}
}
// Check concurrent memory access: CPU and GPU paging ability.
extern "C"{__host__
	int checkMemAccess(int sysId, int devId)
	{
		int concurMem;
		cudaDeviceAttr attr = cudaDevAttrConcurrentManagedAccess;
		cudaDeviceGetAttribute(&concurMem, attr, devList[sysId][devId]);
		return concurMem;
	}
}
// Determine number of GPUs present.
extern "C"{__host__
	int devCount(void)
	{	
		int devCount;

		if(cudaGetDeviceCount(&devCount) != cudaSuccess)
		{
			fprintf(stderr, "devCount CUDA error: Failed to get device count.\n");
			return 0;
		}
		return devCount;
	}
}
// Switch between devices in a multi-GPU setting.
extern "C"{__host__
	void setDevice(int sysId, int devId)
	{
		if(cudaSetDevice(devList[sysId][devId]) != cudaSuccess)
		{
			fprintf(stderr, "setDevice CUDA error: Failed to switch to device %d, system %d.\n", devList[sysId][devId], sysId);
			return;
		}
		return;
	}
}
// Synchronize a specific device. 
extern "C"{__host__
	int devSync(int sysId, int devId)
	{
		setDevice(sysId, devId);

		if(cudaDeviceSynchronize() != cudaSuccess)
		{
			fprintf(stderr, "devSync CUDA error: Failed to synchronize device %d.", devList[sysId][devId]);
			return 1;
		}
		return 0;
	}
}
// Return total memory (in bytes) across all devices included in devList.
extern "C"{__host__
	size_t devMem(int gDevs, int* devList)
	{
		size_t freeMem, totalMem;
		size_t totFreeMem = sizeof(double) * 0;

		for(int devId = 0; devId < gDevs; devId++)
		{
			if(cudaSetDevice(devList[devId]) != cudaSuccess)
			{
				fprintf(stderr, "devMem CUDA error: Failed to switch to device %d.\n", devList[devId]);
				return 1;
			}

			if(cudaMemGetInfo(&freeMem, &totalMem) != cudaSuccess)
			{
				fprintf(stderr, "devMem CUDA error: Failed to fetch free memory information for device %d.\n", devList[devId]);
				return 1;	
			}

			if(cudaDeviceSynchronize() != cudaSuccess)
			{
				fprintf(stderr, "devMem CUDA error: Failed to synchronize device %d.", devList[devId]);
				return 1;
			}
			
			totFreeMem += freeMem;
		}
		return totFreeMem;
	}
}
// Return number of block streams for the system device devId.
extern "C"{__host__
	int blcNumStreams(int sysId, int devId)
	{	
		if(devId == 0)
		{
			return blcStreamSplit[sysId][0];
		}
		else
		{
			return blcStreamSplit[sysId][devId] - blcStreamSplit[sysId][devId - 1];
		}
	}
}
// Return number of volume streams for the system device devId.
extern "C"{__host__
	int volNumStreams(int sysId, int devId)
	{	
		if(devId == 0)
		{
			return volStreamSplit[sysId][0];
		}
		else
		{
			return volStreamSplit[sysId][devId] - volStreamSplit[sysId][devId - 1];
		}
	}
}

// *** maxG operation functions. 

// Perform green function multiplication of circulant. The input current comes is assumed to reside on blcWork[sysId][blcId][0]. The output products are written to blcSum[sysId][blcId]. The required sums are then computed from this memory and written to blcWork[sysId][blcId][1] by sumWorkVecsO.
extern "C"{__host__
	void greenMultO(int sysId, int blcId)
	{
		int devId;
		int curItr;
		int grnItr;
		int posItr;
		int crcCells = blcCells[sysId][blcId][2][0] * blcCells[sysId][blcId][2][1] * blcCells[sysId][blcId][2][2];
		// Determine device responsible for the requested block. 
		devId = blcStreamDev[sysId][blcId];
		setDevice(sysId, devId);
		// Carry out multiplication of matrix direction elements using three streams. 
		// ``Self'' orientation interactions. 
		elewiseMultO<<<blocksCUDA, threadsPerBlc, 0, blcBlasStream[sysId][blcId][0]>>>(3 * crcCells, blcWork[sysId][blcId][0], blcGreenSlf[sysId][blcId], blcSum[sysId][blcId]);
		// First set of cross orientation interactions. 
		posItr = 3;
		
		for(int itr = 0; itr < 3; itr++)
		{
			grnItr = (3 - itr) % 3;
			curItr = (1 + itr) % 3;

			elewiseMultO<<<blocksCUDA, threadsPerBlc, 0, blcBlasStream[sysId][blcId][1]>>>(crcCells, &(blcWork[sysId][blcId][0][curItr * crcCells]), &(blcGreenXrs[sysId][blcId][grnItr * crcCells]), &(blcSum[sysId][blcId][(posItr + itr) * crcCells]));
		}
		// Second set of cross orientation interactions. 
		posItr = 6;

		for(int itr = 0; itr < 3; itr++)
		{
			grnItr = 1 - itr + (3 * (itr / 2)); 
			curItr = (2 + itr) % 3;

			elewiseMultO<<<blocksCUDA, threadsPerBlc, 0, blcBlasStream[sysId][blcId][2]>>>(crcCells, &(blcWork[sysId][blcId][0][curItr * crcCells]), &(blcGreenXrs[sysId][blcId][grnItr * crcCells]), &(blcSum[sysId][blcId][(posItr + itr) * crcCells]));
		}
		return;
	}
}
// Sum over a collection components saved to a work area, depending on the mode setting which switches between volume and block level operations. See cuBLAS CUDA toolkit API for additional information. 
extern "C"{__host__
	int sumWorkVecsO(int sysId, int mode, int itr, int arrayDim, int numCells, const cufftDoubleComplex *workArr, cufftDoubleComplex *sumVec)
	{
		int devId;
		cuDoubleComplex a, b;
		a.x = 1.0;
		a.y = 0.0;
		b.x = 0.0;
		b.y = 0.0;
		
		if(mode == 0)
		{
			devId = volStreamDev[sysId][itr];
		}
		else
		{
			devId = blcStreamDev[sysId][itr];
		}
		// Switch to requested device.
		setDevice(sysId, devId);
		// Perform either a volume sum over distinct source contributions, or a orientation sum for currents generated in a specific block.  
		if(mode == 0)
		{
			if(cublasZgemv(volBlasHandle[sysId][itr][0], CUBLAS_OP_N, 3 * numCells, arrayDim, &a, (cuDoubleComplex*)workArr, 3 * numCells, volOnesVec[sysId][itr], 1, &b, (cuDoubleComplex*)sumVec, 1) != CUBLAS_STATUS_SUCCESS)
			{
				fprintf(stderr, "sumWorkVecsO volume CUBLAS error: Operation failure.");
				return 1;
			}
		}
		else
		{
			if(cublasZgemv(blcBlasHandle[sysId][itr][0], CUBLAS_OP_N, 3 * numCells, arrayDim, &a, (cuDoubleComplex*)workArr, 3 * numCells, blcOnesVec[sysId][itr], 1, &b, (cuDoubleComplex*)sumVec, 1) != CUBLAS_STATUS_SUCCESS)
			{
				fprintf(stderr, "sumWorkVecsO block CUBLAS error: Operation failure.");
				return 1;
			}
		}
		return 0;
	}
}

// *** Memory assignment functions.

// Generate stream assignment information for the system sysId. These MUST be consistent with assignments given in the controlling program or else errors will result.  
extern "C"{__host__
	void assignStreamsO(int sysId)
	{
		int devId = 0;
		int remStreams;
		// Container for determining the split of streams between system devices.
		volStreamSplit[sysId] = (int*)malloc(sizeof(int) * numDevs[sysId]);
		blcStreamSplit[sysId] = (int*)malloc(sizeof(int) * numDevs[sysId]);
		// List connecting each stream to a system device. 
		volStreamDev[sysId] = (int*)malloc(sizeof(int) * domains[sysId]);
		blcStreamDev[sysId] = (int*)malloc(sizeof(int) * blocks[sysId]);
		// Determine stream splits. 
		for(int devId = 0; devId < numDevs[sysId]; devId++)
		{	
			volStreamSplit[sysId][devId] = (int)roundf(domains[sysId] * (devId + 1) / numDevs[sysId]);
			blcStreamSplit[sysId][devId] = (int)roundf(blocks[sysId] * (devId + 1) / numDevs[sysId]);
		}
		// Volume stream allocation. 
		remStreams = volNumStreams(sysId, 0);

		for(int volId = 0; volId < domains[sysId]; volId++)
		{
			if(remStreams > 0)
			{
				remStreams--;
			}
			else
			{
				devId++;
				remStreams = volNumStreams(sysId, devId) - 1;
			}
			volStreamDev[sysId][volId] = devId;
		}
		// Block stream allocation. 
		devId = 0;
		remStreams = blcNumStreams(sysId, 0);

		for(int blcId = 0; blcId < blocks[sysId]; blcId++)
		{
			if(remStreams > 0)
			{
				remStreams--;
			}
			else
			{
				devId++;
				remStreams = blcNumStreams(sysId, devId) - 1;
			}
			blcStreamDev[sysId][blcId] = devId;
		}
		return;
	}
}	
// Allocation of super system level memory.
extern "C"{__host__
	void glbInitO(int sysNum, int devMax)
	{
		// *** CPU side control data. 
		// Adjoint setting. 
		adjSetting = (int*)malloc(sizeof(int) * sysNum);
		// ''Self'' vs. inter-domain interaction setting.
		slfSetting = (int*)malloc(sizeof(int) * sysNum);
 		// Domains, blocks and the number of cells in a system. 
		domains = (int*)malloc(sizeof(int) * sysNum);
		blocks = (int*)malloc(sizeof(int) * sysNum);
		sysCells = (int**)malloc(sizeof(int) * sysNum);
		// Number of devices, and list of devices, used by a system. 
		numDevs = (int*)malloc(sizeof(int) * sysNum);
		devList = (int**)malloc(sizeof(int*) * sysNum);
		// Container for splitting the volumes and blocks of a system across multiple devices. 
		volStreamSplit = (int**)malloc(sizeof(int*) * sysNum);
		blcStreamSplit = (int**)malloc(sizeof(int*) * sysNum);
		// Device locations for each volume and block control stream. 
		volStreamDev = (int**)malloc(sizeof(int*) * sysNum);
		blcStreamDev = (int**)malloc(sizeof(int*) * sysNum);
		// *** GPU side memory.
		// Control streams for non-BLAS operations. 
		blcStream = (cudaStream_t**)malloc(sizeof(cudaStream_t*) * sysNum);
		volStream = (cudaStream_t**)malloc(sizeof(cudaStream_t*) * sysNum);
		fftStream = (cudaStream_t***)malloc(sizeof(cudaStream_t**) * sysNum);
		// Control streams for BLAS operations on the volume and block level. 
		volBlasHandle = (cublasHandle_t***)malloc(sizeof(cublasHandle_t**) * sysNum);
		volBlasStream = (cudaStream_t***)malloc(sizeof(cudaStream_t**) * sysNum); 
		blcBlasHandle = (cublasHandle_t***)malloc(sizeof(cublasHandle_t**) * sysNum);
		blcBlasStream = (cudaStream_t***)malloc(sizeof(cudaStream_t**) * sysNum); 
		// Storage for precompiled Fourier transform plans. 
		fftPlans = (cufftHandle***)malloc(sizeof(cufftHandle**) * sysNum);
		// System current memory. 
		sysCurrSrc = (cuDoubleComplex**)malloc(sizeof(cuDoubleComplex*) * sysNum);
		sysCurrTrg = (cuDoubleComplex**)malloc(sizeof(cuDoubleComplex*) * sysNum);
		// Memory associated with a volume.
		volCells = (int****)malloc(sizeof(int***) * sysNum);
		volWork = (cufftDoubleComplex****)malloc(sizeof(cufftDoubleComplex***) * sysNum);
		volCurrSrc = (cufftDoubleComplex***)malloc(sizeof(cufftDoubleComplex**) * sysNum);
		volCurrTrg = (cufftDoubleComplex***)malloc(sizeof(cufftDoubleComplex**) * sysNum);
		volOnesVec = (cufftDoubleComplex***)malloc(sizeof(cufftDoubleComplex**) * sysNum);
		// Memory associated with a block.
		blcCells = (int****)malloc(sizeof(int***) * sysNum);
		blcWork = (cufftDoubleComplex****)malloc(sizeof(cufftDoubleComplex***) * sysNum);
		blcSum = (cufftDoubleComplex***)malloc(sizeof(cufftDoubleComplex**) * sysNum);
		blcCurrSrc = (cufftDoubleComplex***)malloc(sizeof(cufftDoubleComplex**) * sysNum);
		blcCurrTrg = (cufftDoubleComplex***)malloc(sizeof(cufftDoubleComplex**) * sysNum);
		blcGreenSlf = (cufftDoubleComplex***)malloc(sizeof(cufftDoubleComplex**) * sysNum);
		blcGreenXrs = (cufftDoubleComplex***)malloc(sizeof(cufftDoubleComplex**) * sysNum);
		blcOnesVec = (cufftDoubleComplex***)malloc(sizeof(cufftDoubleComplex**) * sysNum);
		return;
	}
}
// Allocation of system level memory. 
extern "C"{__host__
	int sysInitO(int sysId, int blocksGPU, int threadsPerBlock, int vols, int tSrcCells, int tTrgCells, int gDevs, int slfVal, int* inDevList)
	{	
		// GPU computation settings. 
		blocksCUDA = blocksGPU;
		threadsPerBlc = threadsPerBlock;
		// Store system identification information for later use.  
		domains[sysId] = vols; 
		blocks[sysId] = vols * vols;
		numDevs[sysId] = gDevs;
		// Number of cells in system. 
		sysCells[sysId] = (int*)malloc(sizeof(int) * 2);
		sysCells[sysId][0] = tSrcCells;
		sysCells[sysId][1] = tTrgCells;
		// All systems start as a non-adjoint. The adjoint setting can be flipped by calling the sysAdj function. 
		adjSetting[sysId] = 0;
		// Setting based on whether the system corresponds to a ``complete'' or partial calculation.
		slfSetting[sysId] = slfVal;
		// Association of devices with system. 
		devList[sysId] = (int*)malloc(sizeof(int) * gDevs);

		for(int devId = 0; devId < numDevs[sysId]; devId++)
		{
			devList[sysId][devId] = inDevList[devId];
		}
		// Streams for controlling asynchronous computation of non-BLAS operations.
		volStream[sysId] = (cudaStream_t*)malloc(sizeof(cudaStream_t) * domains[sysId]);
		blcStream[sysId] = (cudaStream_t*)malloc(sizeof(cudaStream_t) * blocks[sysId]);
		fftStream[sysId] = (cudaStream_t**)malloc(sizeof(cudaStream_t*) * blocks[sysId]);
		// Precompiled Fourier transform planes. 
		fftPlans[sysId] = (cufftHandle**)malloc(sizeof(cufftHandle*) * blocks[sysId]);
		// Streams and handles for controlling BLAS operations on the block and volume level. 
		volBlasHandle[sysId] = (cublasHandle_t**)malloc(sizeof(cublasHandle_t*) * domains[sysId]);
		volBlasStream[sysId] = (cudaStream_t**)malloc(sizeof(cudaStream_t*) * domains[sysId]); 
		blcBlasHandle[sysId] = (cublasHandle_t**)malloc(sizeof(cublasHandle_t*) * blocks[sysId]);
		blcBlasStream[sysId] = (cudaStream_t**)malloc(sizeof(cudaStream_t*) * blocks[sysId]); 
		// Information about cells in a given block.
		blcCells[sysId] = (int***)malloc(sizeof(int**) * blocks[sysId]);
		// Work space for embedding / projecting currents into / out of circulant form.  
		blcWork[sysId] = (cufftDoubleComplex***)malloc(sizeof(cufftDoubleComplex**) * blocks[sysId]);
		// Storage for source and target currents of a block. 
		blcCurrSrc[sysId] = (cufftDoubleComplex**)malloc(sizeof(cufftDoubleComplex*) * blocks[sysId]);
		blcCurrTrg[sysId] = (cufftDoubleComplex**)malloc(sizeof(cufftDoubleComplex*) * blocks[sysId]);
		// Vectors used by sumWorkVecsO on a block level. 
		blcOnesVec[sysId] = (cufftDoubleComplex**)malloc(sizeof(cufftDoubleComplex*) * blocks[sysId]);
		// Storage for Green function interaction elements between alike Cartesian directions (Slf) and distinct Cartesian directions (Xrs). 
		blcGreenSlf[sysId] = (cufftDoubleComplex**)malloc(sizeof(cufftDoubleComplex*) * blocks[sysId]);
		blcGreenXrs[sysId] = (cufftDoubleComplex**)malloc(sizeof(cufftDoubleComplex*) * blocks[sysId]);
		// Work space for summing contributions across cartesian directions.
		blcSum[sysId] = (cufftDoubleComplex**)malloc(sizeof(cufftDoubleComplex*) * blocks[sysId]);
		// Number of cells in each volume of a system. 
		volCells[sysId] = (int***)malloc(sizeof(int**) * domains[sysId]);
		// Vectors used by sumWorkVecsO on a volume level. 
		volOnesVec[sysId] = (cufftDoubleComplex**)malloc(sizeof(cufftDoubleComplex*) * domains[sysId]);
		// Work space for summing contributions from multiple source volumes into a single target volume. 
		volWork[sysId] = (cufftDoubleComplex***)malloc(sizeof(cufftDoubleComplex**) * domains[sysId]);
		// Storage for the source and target currents of a volume, here used in the sense of a row of the Green function matrix. 
		volCurrSrc[sysId] = (cufftDoubleComplex**)malloc(sizeof(cufftDoubleComplex*) * domains[sysId]);
		volCurrTrg[sysId] = (cufftDoubleComplex**)malloc(sizeof(cufftDoubleComplex*) * domains[sysId]);
		// Create 
		// The set device function is defined so that 0 always corresponds to the ``head'' device of a system. 
		setDevice(sysId, 0);

		if(cudaMallocManaged((void**) &(sysCurrSrc[sysId]), sizeof(cuDoubleComplex) * 3 * tCells) != cudaSuccess)
		{
			fprintf(stderr, "sysInitO CUDA Error: Failed to create device space for system source current.\n");
			return 1;
		}
		if(cudaMallocManaged((void**) &(sysCurrTrg[sysId]), sizeof(cuDoubleComplex) * 3 * tCells) != cudaSuccess)
		{
			fprintf(stderr, "sysInitO CUDA Error: Failed to create device space for system image current.\n");
			return 1;
		}
		// Generate stream assignment data---number of streams to assign per device and location.
		assignStreamsO(sysId);
		return 0;
	}
}
// Allocation of volume level memory---memory for operations on a row of the Green function matrix. 
extern "C"{__host__
	int volInitO(int sysId, int volId, int *cellsS, int *cellsT)
	{	
		int devId = volStreamDev[sysId][volId];
		setDevice(sysId, devId);
		volCells[sysId][volId] = (int**)malloc(sizeof(int*) * 2);

		for(int typeItr = 0; typeItr < 2; typeItr++)
		{
			volCells[sysId][blcId][typeItr] = (int*)malloc(sizeof(int) * 3);
		}
		for(int cartItr = 0; cartItr < 3; cartItr++)
		{
			volCells[sysId][volId][0][cartItr] = cellsS[cartItr];
			volCells[sysId][volId][1][cartItr] = cellsT[cartItr];
		}
		// Amount of memory needed to give a source current in a volume. 
		size_t volSizeS = sizeof(cufftDoubleComplex) * 3 * volCells[sysId][volId][0][0] * volCells[sysId][volId][0][1] * volCells[sysId][volId][0][2];
		// Amount of memory needed to give a target current in a volume. 
		size_t volSizeT = sizeof(cufftDoubleComplex) * 3 * volCells[sysId][volId][1][0] * volCells[sysId][volId][1][1] * volCells[sysId][volId][1][2];
		// Allocate device memory. 
		if(cudaStreamCreate(&(volStream[sysId][volId])) != cudaSuccess)
		{
			fprintf(stderr, "volInitO CUDA Error: Failed to initialize volume stream %d on device %d.\n", 
				volId, devList[sysId][devId]);
			return 1;
		}
		if(cudaMallocManaged((void**) &(volCurrSrc[sysId][volId]), volSizeS) != cudaSuccess)
		{
			fprintf(stderr, "Stream CUDA Error: Failed to create device space for input volume current on volume stream %d device %d.\n", 
				volId, devList[sysId][devId]);
			return 1;
		}
		if(cudaMallocManaged((void**) &(volCurrTrg[sysId][volId]), volSizeT) != cudaSuccess)
		{
			fprintf(stderr, "volInitO CUDA Error: Failed to create space for output volume current on volume stream %d device %d.\n", 
				volId, devList[sysId][devId]);
			return 1;
		}
		// Volume work areas 
		volWork[sysId][volId] = (cufftDoubleComplex**)malloc(sizeof(cufftDoubleComplex*) * 2);
		// Source work area
		if(cudaMallocManaged((void**) &(volWork[sysId][volId][0]), volSizeS) 
		!= cudaSuccess)
		{
			fprintf(stderr, "volInitO CUDA Error: Failed to create source work space on volume stream %d device %d.\n", volId, devId);
			return 1;
		}
		// Target work area
		if(cudaMallocManaged((void**) &(volWork[sysId][volId][1]), volSizeT * domains[sysId]) 
		!= cudaSuccess)
		{
			fprintf(stderr, "volInitO CUDA Error: Failed to create source work space on volume stream %d device %d.\n", volId, devId);
			return 1;
		}
		// CUBLAS Handles
		volBlasHandle[sysId][volId] = (cublasHandle_t*)malloc(sizeof(cublasHandle_t) * blsNum);
		volBlasStream[sysId][volId] = (cudaStream_t*)malloc(sizeof(cudaStream_t) * blsNum);

		for(int blsId = 0; blsId < blsNum; blsId++)
		{
			if(cudaStreamCreate(&(volBlasStream[sysId][volId][blsId])) != cudaSuccess)
			{
				fprintf(stderr, "volInitO CUDA Error: Failed to initialize cuBLAS stream %d on volume %d.\n", blsId, volId);
				return 1;
			}
			if(cublasCreate(&(volBlasHandle[sysId][volId][blsId])) != CUBLAS_STATUS_SUCCESS) 
			{
				fprintf(stderr, "volInitO CUBLAS Error: Failed to initialize cuBLAS handle %d on volume %d.\n", blsId, volId);
				return 1;
			}
			if(cublasSetStream(volBlasHandle[sysId][volId][blsId], 
				volBlasStream[sysId][volId][blsId]) != CUBLAS_STATUS_SUCCESS) 
			{
				fprintf(stderr, "volInitO CUBLAS Error: Failed to set cuBLAS handle %d to cuBLAS stream on volume %d.\n", blsId, volId);
				return 1;	
			}
		}
		if(cudaMallocManaged((void**) &(volOnesVec[sysId][volId]), sizeof(cufftDoubleComplex) * domains[sysId]) != cudaSuccess)
		{
			fprintf(stderr, "volInitO CUDA Error: Failed to create ones vector for performing sums on volume stream %d, device %d.\n", volId, devList[sysId][devId]);
			return 1;
		}	
		oneArrO<<<blocksCUDA, threadsPerBlc, 0, volStream[sysId][volId]>>>(volOnesVec[sysId][volId], domains[sysId]);

		if(cudaStreamSynchronize(volStream[sysId][volId]) != cudaSuccess)
		{
			fprintf(stderr, "volInitO CUDA Error: Failed to synchronize volume stream %d on device %d.\n", 
				volId, devList[sysId][devId]);
			return 1;
		}
		return 0;
	}
}
// Initialize memory for a Fourier transform stream.
extern "C"{__host__
	int blcInitO(int sysId, int *cellS, int *cellT, int blcId)
	{	
		int devId = blcStreamDev[sysId][blcId];
		setDevice(sysId, devId);
		// Block cell information and streams
		blcCells[sysId][blcId] = (int**)malloc(sizeof(int*) * 3);
		blcWork[sysId][blcId] = (cufftDoubleComplex**)malloc(sizeof(cufftDoubleComplex*) * blcWorkAreas);
		// FFT plans and streams
		fftPlans[sysId][blcId] = (cufftHandle*)malloc(sizeof(cufftHandle) * fftNum);
		fftStream[sysId][blcId] = (cudaStream_t*)malloc(sizeof(cudaStream_t) * fftNum);

		for(int typeItr = 0; typeItr < 3; typeItr++)
		{
			blcCells[sysId][blcId][typeItr] = (int*)malloc(sizeof(int) * 3);
		}
		// Note that this is the cudaFFFT storage convention. Cell numbers are reported from 
		// outermost to innermost (contiguous) storage dimension.
		int fftCells[] = {cellS[2] + cellT[2], cellS[1] + cellT[1], cellS[0] + cellT[0]};

		for(int cartItr = 0; cartItr < 3; cartItr++)
		{
			blcCells[sysId][blcId][0][cartItr] = cellS[cartItr];
			blcCells[sysId][blcId][1][cartItr] = cellT[cartItr];
			blcCells[sysId][blcId][2][cartItr] = cellS[cartItr] + cellT[cartItr];
		}
		size_t srcSize = sizeof(cufftDoubleComplex) * 3 * blcCells[sysId][blcId][0][0] * blcCells[sysId][blcId][0][1] * blcCells[sysId][blcId][0][2];

		size_t trgSize = sizeof(cufftDoubleComplex) * 3 * blcCells[sysId][blcId][1][0] * blcCells[sysId][blcId][1][1] * blcCells[sysId][blcId][1][2];

		size_t circSize = sizeof(cufftDoubleComplex) * 3 * blcCells[sysId][blcId][2][0] * blcCells[sysId][blcId][2][1] * blcCells[sysId][blcId][2][2];

		if(cudaStreamCreate(&(blcStream[sysId][blcId])) != cudaSuccess)
		{
			fprintf(stderr, "blcInitO CUDA Error: Failed to initialize block stream %d device %d.\n", blcId, devList[sysId][devId]);
			return 1;
		}
		for(int fftId = 0; fftId < fftNum; fftId++)
		{
			if(cudaStreamCreate(&(fftStream[sysId][blcId][fftId])) != cudaSuccess)
			{
				fprintf(stderr, "blcInitO CUDA Error: Failed to initialize FFT stream %d on block %d device %d.\n", fftId, blcId, devList[sysId][devId]);
				return 1;
			}
			if(cufftPlan3d(&(fftPlans[sysId][blcId][fftId]), fftCells[0], fftCells[1], fftCells[2], CUFFT_Z2Z) != CUFFT_SUCCESS)
			{
				fprintf(stderr, "blcInitO CUFFFT Error: Failed to create FFT %d plan for stream %d device %d.\n", fftId, blcId, devList[sysId][devId]);
				return 1;	
			}	
			if(cufftSetStream(fftPlans[sysId][blcId][fftId], fftStream[sysId][blcId][fftId]) != CUFFT_SUCCESS)
			{
				fprintf(stderr, "blcInitO CUFFT Error: Failed to associate FFT plan %d on block stream %d device %d.\n", fftId, blcId, devList[sysId][devId]);
				return 1;
			}
		}
		if(cudaMallocManaged((void**) &(blcGreenSlf[sysId][blcId]), circSize) != cudaSuccess)
		{
			fprintf(stderr, "blcInitO CUDA Error: Failed to create device space for self Green block on block stream %d device %d.\n", blcId, devList[sysId][devId]);
			return 1;
		}
		if(cudaMallocManaged((void**) &(blcGreenXrs[sysId][blcId]), circSize) != cudaSuccess)
		{
			fprintf(stderr, "blcInitO CUDA Error: Failed to create device space for exchange Green block on block stream %d, device %d.\n", blcId, devList[sysId][devId]);
			return 1;
		}
		if(cudaMallocManaged((void**) &(blcCurrSrc[sysId][blcId]), srcSize) != cudaSuccess)
		{
			fprintf(stderr, "blcInitO CUDA Error: Failed to create device space for source current on block stream %d, device %d.\n", blcId, devList[sysId][devId]);
			return 1;
		}
		if(cudaMallocManaged((void**) &(blcSum[sysId][blcId]), 3 * circSize) != cudaSuccess)
		{
			fprintf(stderr, "blcInitO CUDA Error: Failed to create device space for summing currents on block stream %d, device %d.\n", blcId, devList[sysId][devId]);
			return 1;
		}
		if(cudaMallocManaged((void**) &(blcCurrTrg[sysId][blcId]), trgSize) != cudaSuccess)
		{
			fprintf(stderr, "blcInitO CUDA Error: Failed to create device space for target current on block stream %d, device %d.\n", blcId, devList[sysId][devId]);
			return 1;
		}
		if(cudaMallocManaged((void**) &(blcOnesVec[sysId][blcId]), 
			sizeof(cufftDoubleComplex) * 3 ) != cudaSuccess)
		{
			fprintf(stderr, "blcInitO CUDA Error: Failed to one vector for performing sums on block stream %d, device %d.\n", blcId, devList[sysId][devId]);
			return 1;
		}

		oneArrO<<<blocksCUDA, threadsPerBlc, 0, blcStream[sysId][blcId]>>>(blcOnesVec[sysId][blcId], 3);

		if(cudaStreamSynchronize(blcStream[sysId][blcId]) != cudaSuccess)
		{
			fprintf(stderr, "blcInitO CUDA Error: Failed to synchronize block stream %d on device %d.\n", blcId, devList[sysId][devId]);
			return 1;
		}
		// Block work areas
		blcWork[sysId][blcId] = (cufftDoubleComplex**)malloc(sizeof(cufftDoubleComplex*) * blcWorkAreas);

		for(int typeItr = 0; typeItr < blcWorkAreas; typeItr++)
		{
			if(cudaMallocManaged((void**) &(blcWork[sysId][blcId][typeItr]), circSize) != cudaSuccess)
			{
				fprintf(stderr, "blcInitO CUDA Error: Failed to create device FFT work space %d on stream %d device %d.\n", typeItr, blcId, devList[sysId][devId]);
				return 1;
			}
		}
		// CUBLAS Handles
		blcBlasHandle[sysId][blcId] = (cublasHandle_t*)malloc(sizeof(cublasHandle_t) * blsNum);
		blcBlasStream[sysId][blcId] = (cudaStream_t*)malloc(sizeof(cudaStream_t) * blsNum); 

		for(int blsId = 0; blsId < blsNum; blsId++)
		{
			if(cudaStreamCreate(&(blcBlasStream[sysId][blcId][blsId])) != cudaSuccess)
			{
				fprintf(stderr, "blcInitO CUDA Error: Failed to initialize cuBLAS stream %d for block %d.\n", blsId, blcId);
				return 1;
			}
			if(cublasCreate(&(blcBlasHandle[sysId][blcId][blsId])) != CUBLAS_STATUS_SUCCESS) 
			{
				fprintf(stderr, "blcInitO CUBLAS Error: Failed to initialize cuBLAS handle %d for block %d.\n", blsId, blcId);
				return 1;
			}
			if(cublasSetStream(blcBlasHandle[sysId][blcId][blsId], 
				blcBlasStream[sysId][blcId][blsId]) != CUBLAS_STATUS_SUCCESS) 
			{
				fprintf(stderr, "blcInitO CUBLAS Error: Failed to set cuBLAS handle %d to cuBLAS stream %d for block %d.\n", blsId, blsId, blcId);
				return 1;	
			}
		}

		if(cudaStreamSynchronize(blcStream[sysId][blcId]) != cudaSuccess)
		{
			fprintf(stderr, "blcInitO CUDA Error: Failed to synchronize block stream %d on device %d.\n", 
				blcId, devList[sysId][devId]);
			return 1;
		}
		return 0;
	}
}
// Free device memory associated with a Fourier transform stream.
extern "C"{__host__
	void blcFinlStreamO(int sysId, int blcId)
	{
		int devId = blcStreamDev[sysId][blcId];
		setDevice(sysId, devId);
		// Synchronize all block streams. 
		if(cudaStreamSynchronize(blcStream[sysId][blcId]) != cudaSuccess)
		{
			fprintf(stderr, "blcFinlStreamO CUDA Error: Failed to synchronize block stream %d on device %d.\n", blcId, devList[sysId][devId]);
			return;
		}
		for(int blsId = 0; blsId < blsNum; blsId++)
		{
			if(cudaStreamSynchronize(blcBlasStream[sysId][blcId][blsId]) != cudaSuccess)
			{
				fprintf(stderr, "blcFinlStreamO CUDA Error: Failed to synchronize block BLAS stream %d on block in preparation for finalization.\n", blsId, blcId);
				return;
			}
		}
		for(int fftId = 0; fftId < fftNum; fftId++)
		{
			if(cudaStreamSynchronize(fftStream[sysId][blcId][fftId]) != cudaSuccess)
			{
				fprintf(stderr, "blcFinlStreamO CUDA Error: Failed to synchronize block FFT stream %d on block %d in preparation for finalization.\n", fftId, blcId);
				return;
			}
		}
		// CUBLAS handles
		for(int blsId = 0; blsId < blsNum; blsId++)
		{
			if(cublasDestroy(blcBlasHandle[sysId][blcId][blsId]) != CUBLAS_STATUS_SUCCESS) 
			{
				fprintf(stderr, "blcFinlStreamO CUBLAS Error: Failed to free cuBLAS handle %d on block %d.\n", blsId, blcId);
				return;
			}
			if(cudaStreamDestroy(blcBlasStream[sysId][blcId][blsId]) != cudaSuccess)
			{
				fprintf(stderr, "blcFinlStreamO CUDA Error: Failed to free cuBLAS stream %d on block %d.\n", blsId, blcId);
				return;
			}
		}
		free(blcBlasHandle[sysId][blcId]);
		free(blcBlasStream[sysId][blcId]);		
		// Block work areas
		for(int typeItr = 0; typeItr < blcWorkAreas; typeItr++)
		{
			if(cudaFree((void*)blcWork[sysId][blcId][typeItr]) != cudaSuccess)
			{
				fprintf(stderr, "blcFinlStreamO CUDA Error: Failed to clear FFT work space %d on stream %d device %d.\n", typeItr, blcId, devList[sysId][devId]);
				return;
			}
		}
		free(blcWork[sysId][blcId]);
		// Sum vector
		if(cudaFree((void*)blcOnesVec[sysId][blcId]) != cudaSuccess)
		{
			fprintf(stderr, "blcFinlStreamO CUDA Error: Failed to clear ones vector on stream %d device %d.\n", blcId, devList[sysId][devId]);
			return;
		}
		if(cudaFree((void*)blcCurrSrc[sysId][blcId]) != cudaSuccess)
		{
			fprintf(stderr, "blcFinlStreamO CUDA Error: Failed to clear source current on block stream %d device %d.\n", blcId, devList[sysId][devId]);
			return;
		}
		if(cudaFree((void*)blcSum[sysId][blcId]) != cudaSuccess)
		{
			fprintf(stderr, "blcFinlStreamO CUDA Error: Failed to clear sum current area on block stream %d device %d.\n", blcId, devList[sysId][devId]);
			return;
		}
		if(cudaFree((void*)blcCurrTrg[sysId][blcId]) != cudaSuccess)
		{
			fprintf(stderr, "blcFinlStreamO CUDA Error: Failed to clear target current on block stream %d device %d.\n", blcId, devList[sysId][devId]);
			return;
		}
		if(cudaFree((void*)blcGreenXrs[sysId][blcId]) != cudaSuccess)
		{
			fprintf(stderr, "blcFinlStreamO CUDA Error: Failed to clear exchange Green block on block stream %d device %d.\n", blcId, devList[sysId][devId]);
			return;
		}
		if(cudaFree((void*)blcGreenSlf[sysId][blcId]) != cudaSuccess)
		{
			fprintf(stderr, "blcFinlStreamO CUDA Error: Failed to clear self Green block on block stream %d device %d.\n", blcId, devList[sysId][devId]);
			return;
		}
		for(int fftId = 0; fftId < fftNum; fftId++)
		{
			if(cufftDestroy(fftPlans[sysId][blcId][fftId]) != CUFFT_SUCCESS)
			{
				fprintf(stderr, "blcFinlStreamO CUDA Error: Failed to clear Fourier plan %d on block stream %d device %d.\n", fftId, blcId, devList[sysId][devId]);
				return;	
			}
			
			if(cudaStreamDestroy(fftStream[sysId][blcId][fftId]) != cudaSuccess)
			{
				fprintf(stderr, "blcFinlStreamO CUDA Error: Failed to destroy FFT stream %d on block %d device %d.\n", fftId, blcId, devList[sysId][devId]);
				return;
			}			
		}
		if(cudaStreamDestroy(blcStream[sysId][blcId]) != cudaSuccess)
		{
			fprintf(stderr, "blcFinlStreamO CUDA Error: Failed to destroy block stream %d on device %d.\n", 
				blcId, devList[sysId][devId]);
			return;
		}
		free(blcCells[sysId][blcId][0]);
		free(blcCells[sysId][blcId][1]);
		free(blcCells[sysId][blcId][2]);
		free(blcCells[sysId][blcId]);
		free(fftPlans[sysId][blcId]);
		free(fftStream[sysId][blcId]);
		return;
	}
}
// Free device memory associated with a volume stream.
extern "C"{__host__
	void volFinlStreamO(int sysId, int volId)
	{
		int devId = volStreamDev[sysId][volId];
		setDevice(sysId, devId);
		// Synchronize all volume streams.
		if(cudaStreamSynchronize(volStream[sysId][volId]) != cudaSuccess)
		{
			fprintf(stderr, "volFinlStreamO CUDA Error: Failed to synchronize volume stream %d on device %d.\n", volId, devList[sysId][devId]);
			return;
		}
		for(int blsId = 0; blsId < blsNum; blsId++)
		{
			if(cudaStreamSynchronize(volBlasStream[sysId][volId][blsId]) != cudaSuccess)
			{
				fprintf(stderr, "volFinlStreamO CUDA Error: Failed to synchronize BLAS stream %d for volume %d on device %d.\n", blsId, volId, devList[sysId][devId]);
				return;
			}
		}
		// Free volume memory. 
		if(cudaFree((void*)volOnesVec[sysId][volId]) != cudaSuccess)
		{
			fprintf(stderr, "volFinlStreamO CUDA Error: Failed to clear ones vector on volume stream %d device %d.\n", volId, devList[sysId][devId]);
			return;
		}
		for(int blsId = 0; blsId < blsNum; blsId++)
		{
			if(cublasDestroy(volBlasHandle[sysId][volId][blsId]) != CUBLAS_STATUS_SUCCESS) 
			{
				fprintf(stderr, "volFinlStreamO CUBLAS Error: Failed to free cuBLAS %d handle on body %d.\n", blsId, volId);
				return;
			}

			if(cudaStreamDestroy(volBlasStream[sysId][volId][blsId]) != cudaSuccess)
			{
				fprintf(stderr, "volFinlStreamO CUDA Error: Failed to free cuBLAS stream %d on body %d.\n", blsId, volId);
				return;
			}
		}
		free(volBlasStream[sysId][volId]);
		free(volBlasHandle[sysId][volId]);
		// Work areas.
		// Source
		if(cudaFree((void*)volWork[sysId][volId][0]) != cudaSuccess)
		{
			fprintf(stderr, "volFinlStreamO CUDA Error: Failed to clear source work area on volume stream %d device %d.\n", volId, devList[sysId][devId]);
			return;
		}
		// Target
		if(cudaFree((void*)volWork[sysId][volId][1]) != cudaSuccess)
		{
			fprintf(stderr, "volFinlStreamO CUDA Error: Failed to clear target work area on volume stream %d device %d.\n", volId, devList[sysId][devId]);
			return;
		}
		free(volWork[sysId][volId]);
		// Current vectors
		if(cudaFree((void*)volCurrTrg[sysId][volId]) != cudaSuccess)
		{
			fprintf(stderr, "volFinlStreamO CUDA Error: Failed to clear output body current on volume stream %d device %d.\n", volId, devList[sysId][devId]);
			return;
		}
		if(cudaFree((void*)volCurrSrc[sysId][volId]) != cudaSuccess)
		{
			fprintf(stderr, "volFinlStreamO CUDA Error: Failed to clear input body current on volume stream %d device %d.\n", volId, devList[sysId][devId]);
			return;
		}
		if(cudaStreamDestroy(volStream[sysId][volId]) != cudaSuccess)
		{
			fprintf(stderr, "volFinlStreamO CUDA Error: Failed to destroy volume stream %d on device %d.", volId, devId);
			return;
		}
		free(volCells[sysId][volId][0]);
		free(volCells[sysId][volId][1]);
		free(volCells[sysId][volId])
		return;
	}
}
// Free system CPU memory
extern "C"{__host__
	void sysFinlO(int sysId)
	{
		// Free the DMR solver.
		// finlDMR(sysId);
		// Free stream assignment information
		free(blcStreamDev[sysId]);
		free(blcStreamSplit[sysId]);
		free(volStreamDev[sysId]);
		free(volStreamSplit[sysId]);
		// Free head device memory
		setDevice(sysId, 0);

		if(cudaFree((void*) sysCurrSrc[sysId]) != cudaSuccess)
		{
			fprintf(stderr, "sysFinlO CUDA Error: Failed to clear global device source current memory.\n");
			return;
		}
		if(cudaFree((void*) sysCurrTrg[sysId]) != cudaSuccess)
		{
			fprintf(stderr, "sysFinlO CUDA Error: Failed to clear global device image current memory.\n");
			return;
		}
		// Free CPU device memory pointers. Note that stream freeing is done by the blcFinlStreamO and volFinlStreamO functions/
		// Body memory
		free(volCurrSrc[sysId]);
		free(volCurrTrg[sysId]);
		free(volOnesVec[sysId]);
		free(volWork[sysId]);
		free(volCells[sysId]);
		// Block memory
		free(blcSum[sysId]);
		free(blcGreenSlf[sysId]);
		free(blcGreenXrs[sysId]);
		free(blcOnesVec[sysId]);
		free(blcWork[sysId]);
		free(blcCells[sysId]);
		free(blcCurrSrc[sysId]);
		free(blcCurrTrg[sysId]);
		// BLAS streams
		free(blcBlasHandle[sysId]);
		free(blcBlasStream[sysId]);
		free(volBlasHandle[sysId]);
		free(volBlasStream[sysId]);
		// Standard streams
		free(volStream[sysId]);
		free(blcStream[sysId]);
		free(fftPlans[sysId]);
		free(fftStream[sysId]);
		// Device list
		free(devList[sysId]);
		// System cells
		free(sysCells[sysId]);
		return;
	}
}
// Free all super system memory. 
extern "C"{__host__
	void glbFinlO(int sysNum)
	{
		// Global settings
		free(adjSetting);
		free(slfSetting);
 		// System size identifiers
		free(domains);
		free(blocks);
		free(numDevs);
		free(sysCells);
		free(devList);
		// Device splits
		free(volStreamDev);
		free(blcStreamDev);
		free(volStreamSplit);
		free(blcStreamSplit);
		// Streams and cuda plans
		free(fftStream);
		free(fftPlans);
		free(blcStream);
		free(volStream);
		// CUBLAS handles
		free(volBlasHandle);
		free(volBlasStream); 
		free(blcBlasHandle);
		free(blcBlasStream); 
		// Per system memory.
		free(sysCurrSrc);
		free(sysCurrTrg);
		// Per block memory.
		free(blcSum);
		free(blcWork);
		free(blcCells);
		free(blcOnesVec);
		free(blcCurrSrc);
		free(blcCurrTrg);
		free(blcGreenSlf);
		free(blcGreenXrs);
		// Per volume memory.
		free(volWork);
		free(volCells);
		free(volCurrSrc);
		free(volCurrTrg);
		free(volOnesVec);
		return;
	}
}
// Returns pointer to block stream memory location.
extern "C"{__host__
	cuDoubleComplex* blcMemId(int sysId, int blcId, int memId)
	{
		switch (memId)
		{
			case 0:
			return blcGreenSlf[sysId][blcId];
			case 1:
			return blcGreenXrs[sysId][blcId];
			case 2:
			return blcWork[sysId][blcId][memId-2];
			case 3:
			return blcWork[sysId][blcId][memId-2];
			default:
			fprintf(stderr, "blcMemId Error: Unrecognized memory selection of %d for block stream %d.\n", memId, blcId);
			return NULL;
		}
	}
}
// Copy memory from host to device allocated block stream location.
extern "C"{__host__
	int blcMemCpyO(double _Complex *hostPtr, int sysId, int blcId, int memId)
	{
		int devId = blcStreamDev[sysId][blcId];
		size_t cpySize = sizeof(cuDoubleComplex) * 3 * blcCells[sysId][blcId][2][2] * blcCells[sysId][blcId][2][1] * blcCells[sysId][blcId][2][0];
		
		cuDoubleComplex* devPtr = blcMemId(sysId, blcId, memId);

		if(cudaSetDevice(devList[sysId][devId]) != cudaSuccess)
		{
			fprintf(stderr, "blcMemCpyO CUDA Error: Failed to set device to %d.\n", devList[sysId][devId]);
			return 1;
		}
		if(cudaMemcpy(devPtr, hostPtr, cpySize, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			fprintf(stderr, "blcMemCpyO CUDA Error: Failed to copy memory from host to device.\n");
			return 1;
		}
		return 0;
	}
}
// Copy memory from host to device allocated volume location. 
// dir == 0 host to device, dir == 1 device to host. 
extern "C"{__host__
	int sysVecCpyO(double _Complex *hostPtr, int dir, int sysId, int memId)
	{
		cuDoubleComplex *sysMemLoc;
		size_t cpySize;
		
		if(memId == 0)
		{
			sysMemLoc = sysCurrSrc[sysId];
			cpySize = sizeof(cuDoubleComplex) * 3 * sysCells[sysId][0];
		}
		else if(memId == 1)
		{
			sysMemLoc = sysCurrTrg[sysId];
			cpySize = sizeof(cuDoubleComplex) * 3 * sysCells[sysId][1];
		}
		else
		{
			fprintf(stderr, "sysVecCpyO Error: Unrecognized memory identifier.\n");	
		}
		if(cudaSetDevice(devList[sysId][0]) != cudaSuccess)
		{
			fprintf(stderr, "sysVecCpyO CUDA Error: Failed to set device to %d.\n", devList[sysId][0]);
			return 1;
		}
		if(dir == 0)
		{
			if(cudaMemcpy(sysMemLoc, hostPtr, cpySize, cudaMemcpyHostToDevice) != cudaSuccess)
			{
				fprintf(stderr, "sysVecCpyO CUDA Error: Failed to copy memory from host to device.\n");
				return 1;
			}
		}
		else if(dir == 1)
		{
			if(cudaMemcpy(hostPtr, sysMemLoc, cpySize, cudaMemcpyDeviceToHost) != cudaSuccess)
			{
				fprintf(stderr, "sysVecCpyO CUDA Error: Failed to copy memory from device to host.\n");
				return 1;
			}
		}
		else
		{
			fprintf(stderr, "sysVecCpyO Error: Unrecognized memory transfer direction.\n");
			return 1;
		}
		if(devSync(sysId, 0) != 0)
		{
			fprintf(stderr, "sysVecCpyO Error: Failed to synchronize device.\n");
			return 1;	
		}
		return 0;
	}
}
// Perform forward or reverse Fourier transform on an block stream memory location.
// No normalization is carried out, in agreement with the requirements of the circulant form calculation.
extern "C"{__host__
	int fftExeO(int fftDir, int sysId, int blcId, int memId)
	{
		int devId;
		int circCells;
		int fftDirection;
		cufftDoubleComplex *memLoc;
		
		if(fftDir == 0)
		{		
			fftDirection = 	CUFFT_FORWARD;
		}
		else if(fftDir == 1)
		{
			fftDirection = 	CUFFT_INVERSE;
		}
		else
		{
			fprintf(stderr, "fftExeO Error: Unrecognized transform direction %d", fftDir);
			return 1;
		}
		devId = blcStreamDev[sysId][blcId];
		setDevice(sysId, devId);
		// Get device memory location
		memLoc = blcMemId(sysId, blcId, memId);
		circCells = blcCells[sysId][blcId][2][0] * blcCells[sysId][blcId][2][1] * blcCells[sysId][blcId][2][2];
		
		for(int fftId = 0; fftId < fftNum; fftId++)
		{
			if(cufftExecZ2Z(fftPlans[sysId][blcId][fftId], &(memLoc[fftId * circCells]), &(memLoc[fftId * circCells]), fftDirection) != CUFFT_SUCCESS)
			{
				fprintf(stderr, "fftExeO CUFFT Error: Failed to execute forward Fourier transform %d on block %d device %d.\n", fftId, blcId, devList[sysId][devId]);
				return 1;
			}
		}
		return 0;
	}
}
// Perform initial Fourier transform of all system Green blocks.
extern "C"{__host__
	int fftInitHostO(int *sysIds, int sysNum)
	{
		int sysId;
		int devId;
		int circCells;
		int fftDirection;
		int greenMemBound = 2;
		cufftDoubleComplex *memLoc;
		fftDirection = 	CUFFT_FORWARD;
		
		for(int sysInd = 0; sysInd < sysNum; sysInd++)
		{
			sysId = sysIds[sysInd];
			
			for(int blcId = 0 ; blcId < blocks[sysId]; blcId++)
			{
				devId = blcStreamDev[sysId][blcId];
				setDevice(sysId, devId);

				for(int memId = 0; memId < greenMemBound; memId++)
				{
					memLoc = blcMemId(sysId, blcId, memId);
					// Cell sizes
					circCells = blcCells[sysId][blcId][2][0] * blcCells[sysId][blcId][2][1] * blcCells[sysId][blcId][2][2];
					
					for(int fftId = 0; fftId < fftNum; fftId++)
					{
						if(cufftExecZ2Z(fftPlans[sysId][blcId][fftId], &(memLoc[fftId * circCells]), &(memLoc[fftId * circCells]), fftDirection) != CUFFT_SUCCESS)
						{
							fprintf(stderr, "fftInitHostO CUFFT Error: Failed to execute forward Fourier transform %d on block %d device %d.\n", fftId, blcId, devList[sysId][devId]);
							return 1;
						}
					}
				}
			}
		}
		for(int sysInd = 0; sysInd < sysNum; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int blcId = 0 ; blcId < blocks[sysId]; blcId++)
			{
				devId = blcStreamDev[sysId][blcId];
				setDevice(sysId, devId);

				for(int fftId = 0; fftId < fftNum; fftId++)
				{
					if(cudaStreamSynchronize(fftStream[sysId][blcId][fftId]) != cudaSuccess)
					{
						fprintf(stderr, "fftInitHostO CUDA Error: Failed to synchronize FFT stream %d on block %d after host execution.\n", fftId, blcId);
						return 1;
					}
				}
			}
		}
		return 0;
	}
}
// Splits input current into domains and passes results to appropriate devices 
extern "C"{__host__
	int sysCurrSowO(int *sysIds, int sysNum)
	{
		int sysId;
		int cellsS;
		int offset;
		int devTrgItr;
		int headDevItr = 0;

		for(int sysInd = 0; sysInd < sysNum; sysInd++)
		{
			offset = 0;
			sysId = sysIds[sysInd];

			for(int volId = 0; volId < domains[sysId]; volId++)
			{	
				cellsS = volCells[sysId][volId][0][0] * volCells[sysId][volId][0][1] * volCells[sysId][volId][0][2];
				
				if(volId > 0)
				{
					offset += 3 * volCells[sysId][volId - 1][0][0] * volCells[sysId][volId - 1][0][1] * volCells[sysId][volId - 1][0][2];
				}
				devTrgItr = volStreamDev[sysId][volId];

				if(cudaMemcpyPeerAsync(volCurrSrc[sysId][volId], devList[sysId][devTrgItr], &(sysCurrSrc[sysId][offset]), devList[sysId][headDevItr], sizeof(cuDoubleComplex) * 3 * cellsS, volStream[sysId][volId]) != cudaSuccess)
				{
					fprintf(stderr, "sysCurrSowO CUDA Error: Failed to copy volume current to volume stream %d device %d, system %d.\n", volId, devList[sysId][devTrgItr], sysId);
					return 1;
				}
				if(cudaMemcpyPeerAsync(volWork[sysId][volId][0], devList[sysId][devTrgItr], volCurrSrc[sysId][volId], devList[sysId][devTrgItr], sizeof(cuDoubleComplex) * 3 * cellsS, volStream[sysId][volId]) != cudaSuccess)
				{
					fprintf(stderr, "sysCurrSowO CUDA Error: Failed to copy volume current to volume source work area on device %d, system %d.\n", volId, devList[sysId][devTrgItr], sysId);
					return 1;
				}
			}
		}
		for(int sysInd = 0; sysInd < sysNum; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int volId = 0; volId < domains[sysId]; volId++)
			{
				devId = volStreamDev[sysId][volId];
				setDevice(sysId, devId);
				
				if(cudaStreamSynchronize(volStream[sysId][volId]) != cudaSuccess)
				{
					fprintf(stderr, "sysCurrSow CUDA Error: Failed to synchronize the block stream %d after current copy from global source.\n", volId);
					return 1;
				}
			}
		}
		return 0;
	}
}
// Joins output body currents from all active devices into system current. 
extern "C"{__host__
	int sysCurrReapO(int *sysIds, int sysSize)
	{
		int sysId;
		int cellsT;
		int offset;
		int devSrcItr = 0;
		int headDevItr = 0;

		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			offset = 0;
			sysId = sysIds[sysInd];

			for(int volId = 0; volId < domains[sysId]; volId++)
			{
				cellsT = volCells[sysId][volId][1][0] * volCells[sysId][volId][1][1] * volCells[sysId][volId][1][2];

				if(volId > 0)
				{
					offset += 3 * volCells[sysId][volId - 1][1][0] * volCells[sysId][volId - 1][1][1] * volCells[sysId][volId - 1][1][2];
				}
				devSrcItr = volStreamDev[sysId][volId];

				if(cudaMemcpyPeerAsync(&(sysCurrSrc[sysId][offset]), devList[sysId][headDevItr], volCurrTrg[sysId][volId], devList[sysId][devSrcItr], sizeof(cuDoubleComplex) * 3 * cellsT, volStream[sysId][volId]) != cudaSuccess)
				{
					fprintf(stderr, "sysCurrReapO CUDA Error: Failed to copy body current from volume stream %d device %d to global current, system %d.\n", volId, devSrcItr, sysId);
					return 1;
				}
			}
		}
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];
			
			for(int volId = 0; volId < domains[sysId]; volId++)
			{	
				devId = volStreamDev[sysId][volId];
				setDevice(sysId, devId);

				if(cudaStreamSynchronize(volStream[sysId][volId]) != cudaSuccess)
				{
					fprintf(stderr, "sysCurrReapO CUDA Error: Failed to synchronize volume stream %d on device %d.\n", volId, devSrcItr);
					return 1;
				}
			}
		}
		return 0;
	}
}
// Write volume current to all associated block streams. 
// The linear numbering of the blocks of the Green matrix is assumed to follow column major order.
extern "C"{__host__
	int volCurrSowO(int sysId, int volId)
	{
		int devTrgItr;
		int devSrcItr = volStreamDev[sysId][volId];
		size_t cpySize = sizeof(cuDoubleComplex) * 3 * volCells[sysId][volId][0][0] * volCells[sysId][volId][0][1] * volCells[sysId][volId][0][2];

		for(int blcId = volId * domains[sysId]; blcId < (volId + 1) * domains[sysId]; blcId++)
		{
			devTrgItr = blcStreamDev[sysId][blcId];

			if(cudaMemcpyPeerAsync(blcCurrSrc[sysId][blcId], devList[sysId][devTrgItr], volWork[sysId][volId][0], devList[sysId][devSrcItr], cpySize, blcStream[sysId][blcId]) != cudaSuccess)
			{
				fprintf(stderr, "volCurrSow CUDA Error: Failed to copy volume current from volume stream %d device %d to block source location %d on device %d.\n", volId, devSrcItr, blcId, devTrgItr);
				return 1;
			}
		}
		return 0;
	}
}
// Write all output block streams into the volume stream work areas. 
// The linear numbering of the blocks of the Green matrix is assumed to follow column major order.
extern "C"{__host__
	int volCurrReapO(int sysId, int volId)
	{
		int loopCount = 0;
		int devSrcId = 0;
		int devTrgId = volStreamDev[sysId][volId];
		int cpyCells = volCells[sysId][volId][1][0] * volCells[sysId][volId][1][1] * volCells[sysId][volId][1][2];
		size_t cpySize = sizeof(cuDoubleComplex) * 3 * cpyCells; 

		for(int blcId = volId; blcId < blocks[sysId]; blcId += domains[sysId])
		{
			devSrcId = blcStreamDev[sysId][blcId];

			if(cudaMemcpyPeerAsync(&(volWork[sysId][volId][1][loopCount * 3 * cpyCells]), devList[sysId][devTrgId], blcCurrTrg[sysId][blcId], devList[sysId][devSrcId], cpySize, blcStream[sysId][blcId]) != cudaSuccess)
			{
				fprintf(stderr, "volCurrReapO CUDA Error: Failed to copy body current from block stream %d on device %d to volume stream %d on device %d.\n", blcId, devSrcId, blcId, devTrgId);
				return 1;
			}
			loopCount++;
		}
		return 0;
	}
}
extern "C"{__host__
	int sysAdjO(int *sysIds, int sysNum)
	{	
		int sysId;
		int devId;
		int tCells; 
		int blcCircCells;
		
		for(int sysInd = 0; sysInd < sysNum; sysInd++)
		{
			sysId = sysIds[sysInd];
			adjSetting[sysId] = (adjSetting[sysId] + 1) % 2;
			
			for(int devId = 0; devId < numDevs[sysId]; devId++)
			{
				if(devSync(sysId, devId) != 0)
				{
					fprintf(stderr, "sysAdj Error: Failed to synchronize device %d in preparation for taking system adjoint.", devList[sysId][devId]);
					return 1;
				}
			}
		}
		for(int sysInd = 0; sysInd < sysNum; sysInd++)
		{
			sysId = sysIds[sysInd];
			
			for(int blcId = 0; blcId < blocks[sysId]; blcId++)
			{			
				blcCircCells = blcCells[sysId][blcId][2][0] * blcCells[sysId][blcId][2][1] * blcCells[sysId][blcId][2][2]; 
				devId = blcStreamDev[sysId][blcId];
				setDevice(sysId, devId);

				for(int blsId = 0; blsId < blsNum; blsId++)
				{
					conjArrO<<<blocksCUDA, threadsPerBlc, 0, fftStream[sysId][blcId][blsId]>>>(&(blcGreenXrs[sysId][blcId][blsId * blcCircCells]), blcCircCells);

					conjArrO<<<blocksCUDA, threadsPerBlc, 0, blcBlasStream[sysId][blcId][blsId]>>>(&(blcGreenSlf[sysId][blcId][blsId * blcCircCells]), blcCircCells);
				}
			}
		}
		for(int sysInd = 0; sysInd < sysNum; sysInd++)
		{	
			sysId = sysIds[sysInd];

			for(devId = 0; devId < numDevs[sysId]; devId++)
			{
				if(devSync(sysId, devId) != 0)
				{
					fprintf(stderr, "sysAdj Error: Failed to synchronize device %d after taking system adjoint.", devList[sysId][devId]);
					return 1;
				}
			}
		}
		return 0;
	}
}
// Perform first half of application of the Green function operator. 
extern "C"{__host__
	int greenOprO(int *sysIds, int sysSize)
	{	
		int devId; 
		int sysId;
		int crcCells;
		int srcCells;
		int trgCells;
		int fftDirection;
		// block work positions, setting both positions to 0 skips Green function action.
		int srcPos = 0;
		int trgPos = 1;
		// Copy system current to all volume streams.
		if(sysCurrSowO(sysIds, sysSize) != 0)
		{
			fprintf(stderr, "greenOprO CUDA Error: Failed to split system current to volume streams.\n");
			return 1;
		}
		// Copy body currents to appropriate block streams
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];
			
			for(int volId = 0; volId < domains[sysId]; volId++)
			{
				if(volCurrSowO(sysId, volId) != 0)
				{
					fprintf(stderr, "greenOprO CUDA Error: Failed to copy body current %d to associated block streams.\n", volId);
					return 1;
				}
			}
		}
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int blcId = 0; blcId < blocks[sysId]; blcId++)
			{	
				devId = blcStreamDev[sysId][blcId];
				setDevice(sysId, devId);
				
				if(cudaStreamSynchronize(blcStream[sysId][blcId]) != cudaSuccess)
				{
					fprintf(stderr, "greenOprO CUDA Error: Failed to synchronize block stream %d after current copy from volume stream.\n", blcId);
					return 1;
				}
			}
		}
		// Perform embeddings
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int blcId = 0; blcId < blocks[sysId]; blcId++)
			{
				devId = blcStreamDev[sysId][blcId];
				setDevice(sysId, devId);
				
				srcCells = blcCells[sysId][blcId][0][0] * blcCells[sysId][blcId][0][1] * blcCells[sysId][blcId][0][2];
				
				crcCells = blcCells[sysId][blcId][2][0] * blcCells[sysId][blcId][2][1] * blcCells[sysId][blcId][2][2];
				
				for(int blsId = 0; blsId < blsNum; blsId++)
				{
					embedVecO<<<blocksCUDA, threadsPerBlc, 0, blcBlasStream[sysId][blcId][blsId]>>>(&(blcCurrSrc[sysId][blcId][srcCells * blsId]), crcCells, &(blcWork[sysId][blcId][srcPos][crcCells * blsId]), blcCells[sysId][blcId][0][0], blcCells[sysId][blcId][0][1],blcCells[sysId][blcId][0][2], blcCells[sysId][blcId][2][0], blcCells[sysId][blcId][2][1], blcCells[sysId][blcId][2][2]);
				}
			}
		}
		// Wait for completion of embeddings
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int blcId = 0; blcId < blocks[sysId]; blcId++)
			{
				devId = blcStreamDev[sysId][blcId];
				setDevice(sysId, devId);

				for(int blsId = 0; blsId < blsNum; blsId++)
				{	
					if(cudaStreamSynchronize(blcBlasStream[sysId][blcId][blsId]) != cudaSuccess)
					{
						fprintf(stderr, "greenOprO CUDA Error: Failed to synchronize BLAS stream %d  on block %d after embedding.\n",
							blsId, blcId);
						return 1;
					}
				}
			}
		}
		// First FFT step
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int blcId = 0; blcId < blocks[sysId]; blcId++)
			{
				devId = blcStreamDev[sysId][blcId];
				setDevice(sysId, devId);
				crcCells = blcCells[sysId][blcId][2][0] * blcCells[sysId][blcId][2][1] * blcCells[sysId][blcId][2][2];

				if(adjSetting[sysId] == 0)
				{
					fftDirection = 	CUFFT_FORWARD;
				}
				else
				{
					fftDirection = 	CUFFT_INVERSE;
				}
				for(int fftId = 0; fftId < fftNum; fftId++)
				{
					if(cufftExecZ2Z(fftPlans[sysId][blcId][fftId], &(blcWork[sysId][blcId][srcPos][fftId * crcCells]), &(blcWork[sysId][blcId][srcPos][fftId * crcCells]), fftDirection) != CUFFT_SUCCESS)
					{
						fprintf(stderr, "greenOprO CUFFT Error: Failed to execute forward Fourier transform %d on block stream %d device %d.\n", 
							fftId, blcId, devList[sysId][devId]);
						return 1;
					}
				}
			}
		}
		// Synchronize
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];
			
			for(int blcId = 0; blcId < blocks[sysId]; blcId++)
			{
				devId = blcStreamDev[sysId][blcId];
				setDevice(sysId, devId);

				for(int fftId = 0; fftId < fftNum; fftId++)
				{
					if(cudaStreamSynchronize(fftStream[sysId][blcId][fftId]) != cudaSuccess)
					{
						fprintf(stderr, "greenOprO CUDA Error: Failed to synchronize FFT stream %d on block %d after first FFT.\n",
							fftId, blcId);
						return 1;
					}
				}
			}
		}
		// Perform Green function multiplication.
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];
			
			for(int blcId = 0; blcId < blocks[sysId]; blcId++)
			{
				greenMultO(sysId, blcId);
			}
		}
		// Synchronized branched currents in preparation for summing results
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];
			
			for(int blcId = 0; blcId < blocks[sysId]; blcId++)
			{
				devId = blcStreamDev[sysId][blcId];
				setDevice(sysId, devId);

				for(int blsId = 0; blsId < blsNum; blsId++)
				{
					if(cudaStreamSynchronize(blcBlasStream[sysId][blcId][blsId]) != cudaSuccess)
					{
						fprintf(stderr, "greenOprO CUDA Error: Failed to synchronize BLAS stream %d on block %d, device %d.\n", blsId, blcId, devList[sysId][devId]);
						return 1;
					}
				}
			}
		}
		// Sum results, interaction between different Cartesian directions.
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int blcId = 0; blcId < blocks[sysId]; blcId++)
			{
				crcCells = blcCells[sysId][blcId][2][0] * blcCells[sysId][blcId][2][1] * blcCells[sysId][blcId][2][2];

				if(sumWorkVecsO(sysId, 1, blcId, 3, crcCells, blcSum[sysId][blcId], blcWork[sysId][blcId][trgPos]) != 0)
				{
					fprintf(stderr, "greenOprO CUDA Error: Failed sum green function multiplication results for block %d.\n", blcId);
					return 1;	
				}	
			}
		}
		// Synchronize 
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];
			
			for(int blcId = 0; blcId < blocks[sysId]; blcId++)
			{
				devId = blcStreamDev[sysId][blcId];
				setDevice(sysId, devId);

				for(int blsId = 0; blsId < blsNum; blsId++)
				{
					if(cudaStreamSynchronize(blcBlasStream[sysId][blcId][blsId]) != cudaSuccess)
					{
						fprintf(stderr, "greenOprO CUDA Error: Failed to synchronize BLAS stream %d on block %d after Green function sum.\n", blsId, blcId);
						return 1;
					}
				}
			}
		}
		// Second FFT Step
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int blcId = 0; blcId < blocks[sysId]; blcId++)
			{
				devId = blcStreamDev[sysId][blcId];
				setDevice(sysId, devId);
				crcCells = blcCells[sysId][blcId][2][0] * blcCells[sysId][blcId][2][1] * blcCells[sysId][blcId][2][2];

				if(adjSetting[sysId] == 0)
				{
					fftDirection = 	CUFFT_INVERSE;
				}
				else
				{
					fftDirection = 	CUFFT_FORWARD;
				}

				for(int fftId = 0; fftId < fftNum; fftId++)
				{
					if(cufftExecZ2Z(fftPlans[sysId][blcId][fftId], &(blcWork[sysId][blcId][trgPos][fftId * crcCells]), &(blcWork[sysId][blcId][trgPos][fftId * crcCells]), fftDirection) != CUFFT_SUCCESS)
					{
						fprintf(stderr, "greenOprO CUFFT Error: Failed to execute second FFT step, stream %d on block %d device %d.\n",
							fftId, blcId, devList[sysId][devId]);
						return 1;
					}
				}
			}
		}
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int blcId = 0; blcId < blocks[sysId]; blcId++)
			{
				devId = blcStreamDev[sysId][blcId];
				setDevice(sysId, devId);

				for(int fftId = 0; fftId < fftNum; fftId++)
				{
					if(cudaStreamSynchronize(fftStream[sysId][blcId][fftId]) != cudaSuccess)
					{
						fprintf(stderr, "greenOprO CUDA Error: Failed to synchronize FFT stream %d on block %d after second FFT.\n", fftId, blcId);
						return 1;
					}
				}
			}
		}
		// Scaling step
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int blcId = 0; blcId < blocks[sysId]; blcId++)
			{
				devId = blcStreamDev[sysId][blcId];
				setDevice(sysId, devId);
				
				crcCells = blcCells[sysId][blcId][2][0] * blcCells[sysId][blcId][2][1] * blcCells[sysId][blcId][2][2];

				for(int blsId = 0; blsId < blsNum; blsId++)
				{
					if(scaleArrO(sysId, blcId, blsId, &(blcWork[sysId][blcId][trgPos][blsId * crcCells]), 1.0 / crcCells, crcCells) != 0)
					{
						fprintf(stderr, "greenOprO CUBLAS Error: Failed scale array after FFT steps, BLAS stream %d on block %d.\n", blsId, blcId);
						return 1;
					}
				}
			}
		}
		// Perform projections
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int blcId = 0; blcId < blocks[sysId]; blcId++)
			{
				devId = blcStreamDev[sysId][blcId];
				setDevice(sysId, devId);

				crcCells = blcCells[sysId][blcId][2][0] * 
				blcCells[sysId][blcId][2][1] * blcCells[sysId][blcId][2][2];
				trgCells = blcCells[sysId][blcId][1][0] * 
				blcCells[sysId][blcId][1][1] * blcCells[sysId][blcId][1][2];

				for(int blsId = 0; blsId < blsNum; blsId++)
				{	
					projVecO<<<blocksCUDA, threadsPerBlc, 0, blcBlasStream[sysId][blcId][blsId]>>>(&(blcWork[sysId][blcId][trgPos][blsId * crcCells]), &(blcCurrTrg[sysId][blcId][blsId * trgCells]), blcCells[sysId][blcId][2][0], blcCells[sysId][blcId][2][1], blcCells[sysId][blcId][2][2], blcCells[sysId][blcId][1][0], blcCells[sysId][blcId][1][1], blcCells[sysId][blcId][1][2]);
				}
			}
		}
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int blcId = 0; blcId < blocks[sysId]; blcId++)
			{
				devId = blcStreamDev[sysId][blcId];
				setDevice(sysId, devId);

				for(int blsId = 0; blsId < blsNum; blsId++)
				{
					if(cudaStreamSynchronize(blcBlasStream[sysId][blcId][blsId]) 
						!= cudaSuccess)
					{
						fprintf(stderr, "greenOprO CUDA Error: Failed to synchronize BLAS stream %d on block %d after projections.\n",
							blsId, blcId);
						return 1;
					}
				}
			}
		}
		return 0;
	}
}
// Join block results to form a system output. slfSetting is a two parameter option specifying whether the system corresponds to a collections of domains that could be considered as complete, or a simply some inter-domain interaction. 
// slfSetting == 0 -> ``self'' domain interactions; slfSetting == 0 -> inter-domain interaction. 
extern "C"{__host__
	int joinCurrO(int* sysIds, int sysSize)
	{	
		int sysId;
		int devId;
		int tCells;
		// Collect all block stream results
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int volId = 0; volId < domains[sysId]; volId++)
			{
				if(volCurrReapO(sysId, volId) != 0)
				{
					fprintf(stderr,"joinCurrO Error: Failed to collect block stream %d, system %d.\n", volId, sysId);
					return 1;
				}
			}
		}
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int blcId = 0; blcId < blocks[sysId]; blcId++)
			{
				devId = blcStreamDev[sysId][blcId];
				setDevice(sysId, devId);

				if(cudaStreamSynchronize(blcStream[sysId][blcId]) != cudaSuccess)
				{
					fprintf(stderr, "joinCurrO CUDA Error: Failed to synchronize block stream %d.\n", blcId);
					return 1;
				}
			}
		}
		// Sum results into volCurrTrg
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int volId = 0; volId < domains[sysId]; volId++)
			{
				tCells = volCells[sysId][volId][1][0] * volCells[sysId][volId][1][1] * volCells[sysId][volId][1][2]; 
				// Sums columns of volWork into volCurrTrg.
				if(sumWorkVecsO(sysId, 0, volId, domains[sysId], tCells, volWork[sysId][volId][1], volCurrTrg[sysId][volId]) != 0)
				{
					fprintf(stderr, "joinCurrO Error: Failed to sum work array for volume stream %d.\n", volId);
					return 1;	
				}
			}
		}
		// Synchronize in preparation to return to main streams.
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int volId = 0; volId < domains[sysId]; volId++)
			{
				devId = volStreamDev[sysId][volId];
				setDevice(sysId, devId);

				if(cudaStreamSynchronize(volBlasStream[sysId][volId][0]) != cudaSuccess)
				{
					fprintf(stderr, "joinCurrO CUDA Error: Failed to synchronize BLAS stream for body %d.\n", volId);
					return 1;
				}
			}
		}
		// Account for differing behaviour of ``self'' and inter-domain Green function interactions.
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];
			
			if(slfSetting[sysId] == 0)
			{
				for(int volId = 0; volId < domains[sysId]; volId++)
				{
					devId = volStreamDev[sysId][volId];
					setDevice(sysId, devId);
					tCells = volCells[sysId][volId][1][0] * volCells[sysId][volId][1][1] * volCells[sysId][volId][1][2]; 

					for(int blsId = 0; blsId < blsNum; blsId++)
					{
						arrDiffO<<<blocksCUDA, threadsPerBlc, 0, volBlasStream[sysId][volId][blsId]>>>(1, &(volCurrSrc[sysId][volId][blsId * tCells]), &(volCurrTrg[sysId][volId][blsId * tCells]), tCells);	
					}
				}
			}
		}
		// Synchronize in preparation for exporting volume results to system. 
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int volId = 0; volId < domains[sysId]; volId++)
			{
				devId = volStreamDev[sysId][volId];
				setDevice(sysId, devId);

				for(int blsId = 0; blsId < blsNum; blsId++)
				{
					if(cudaStreamSynchronize(volBlasStream[sysId][volId][blsId]) != cudaSuccess)
					{
						fprintf(stderr, "joinCurrO CUDA Error: Failed to synchronize BLAS stream %d on body %d.\n", blsId, volId);
						return 1;
					}
				}
			}
		}
		// Load results into system currents
		if(sysCurrReapO(sysIds, sysSize) != 0)
		{
			fprintf(stderr, "joinCurrO Error: Failed to join body currents into system current.\n");
			return 1;
		}
		// Wait for completion of asynchronous kernel launches
		for(int sysInd = 0; sysInd < sysSize; sysInd++)
		{
			sysId = sysIds[sysInd];

			for(int volId = 0; volId < domains[sysId]; volId++)
			{
				devId = volStreamDev[sysId][volId];
				setDevice(sysId, devId);

				if(cudaStreamSynchronize(volStream[sysId][volId]) != cudaSuccess)
				{
					fprintf(stderr, "joinCurrO CUDA Error: Failed to synchronize volume stream %d, system %d.\n", volId, sysId);
					return 1;
				}
			}
		}
		return 0;
	}
}
// Perform an iteration of the operator. joinCurrO ensures synchronization.
extern "C"{__host__
	int maxGO(int *sysIds, int sysSize)
	{	
		if(greenOpr(sysIds, sysSize) != 0)
		{
			fprintf(stderr, "maxGO Error: Application of circulant Green function has failed.\n");
			return 1;
		}

		if(joinCurrO(sysIds, sysSize, fieldMode) != 0)
		{
			fprintf(stderr, "maxGO Error: Summing has failed.\n");
			return 1;
		}
		return 0;
	}
}