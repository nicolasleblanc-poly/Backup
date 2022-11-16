#ifdef __cplusplus
extern "C" {
#endif
// Returns number of currently visible devices.
int devCount(void);
// Sets global number of systems that will be considered based on sysNum.
// In all following functions the system is specified by sysItr.
void glbInit(int sysNum, int devMax);
// Initialize global variables based number of bodies and available devices. 
// Call before any following functions. Returns 0 on successful initialization, 1 otherwise.
// blockGPU and threadsPerBlock: GPU computation settings (see cudaToolkit documentation).
// sysItr: select a system to initialize, number must be smaller than sysNum.
// bodies: number of bodies in the system.
// totCells: number of cells in the total system.
// basisDim: dimensionality of total solution basis (Krylov + deflation), basisDim + 1 must be
// divisible by the number of DMR solver devices numDMRDevs.
// deflatDim: dimensionality of the deflation basis, must be divisible by the number of DMR solver
// devices numDMRDevs.
// prefactorMode : determines how material response is applied to currents.
// numDevs: number of devices performing  operator actions.
// viDevList: integer list of device locations for  operator application. Referred to by devItr.
// numDMRDevs: number of devices performing DMR inverse solves.
// dmrDevList: integer list of device locations for DMR solver.
int sysInit(int sysItr, int blocksGPU, int threadsPerBlock, int bodies, int totCells, int basisDim, int deflateDim, int numDevs, int* viDevList, int numDMRDevs, int*dmrDevList);
// Sets device to interpret commands. Done internally in all functions with devItr variable.
void setDevice(int sysItr, int devItr);
// Waits for completion of all scheduled tasks on a device.
int devSync(int sysItr, int devItr);
// Initialize memory for a block device stream. Returns 0 on successful initialization, 1 otherwise.
int blcInitStream(int sysItr, int *cellS, int *cellT, int blcItr);
// Initialize memory for a body device stream. Returns 0 on successful initialization, 1 otherwise.
int volInitStream(int sysItr, int *cellsBdy, int volItr);
// Free memory allocated for an block stream.
void blcFinlStream(int sysItr, int blcItr);
// Free memory allocated for a body stream.
void volFinlStream(int sysItr, int volItr);
// Free system memory positions allocated by sysInit.
void sysFinl(int sysItr);
// Free global memory.
void glbFinl(int sysNum);
// Copy memory from host to device allocated block stream location: 
// memInd == 0, blcGreenSlf
// memInd == 1, blcGreenExc
// Returns 0 on successful copy, 1 otherwise.
int blcMemCpy(int sysItr, double _Complex *hostPtr, int memInd, int blcItr);
// Copy material response memory from host to device allocated body stream location.
// Returns 0 on successful copy, 1 otherwise.
int volMemCpy(int sysItr, double _Complex *hostPtr, int volItr);
// Copy memory from host to the head device system stream: 
// memInd == 0, sysCurrSrc
// memInd == 1, sysCurrTrg
// dir == 0 host to device, dir == 1 device to host.
// Returns 0 on successful copy, 1 otherwise.
int sysMemCpy(int sysItr, double _Complex *hostPtr, int dir, int memInd);
// Perform forward or reverse Fourier transform of memory on a Fourier stream:
// fftDir == 0 => FORWARD, fftDir == 1 => REVERSE
// memInd == 0, blcGreenSlf
// memInd == 1, blcGreenExc
// Returns 0 on successful copy, 1 otherwise.
int fftExeHost(int *sysIds, int sysNum, int fftDir, int memItr, int blcItr);
// Perform system wide forward Fourier transform of all green function memory
int fftInitHost(int *sysIds, int sysNum);
// Conjugate all material responses in order to solve adjoint system.
int sysAdj(int *sysIds, int sysNum);
// Perform inverse solver for current sysCurrTrg, saved to sysCurrSrc. 
// sysIds: Array of system identifier to perform inverse solves on.
// sysSize: Size of sysIds array.
// deflateMode: A value of !0 indicates that the solver has been previously called 
// on a similar linear system and that this existing deflation space should 
// be used in the first Arnoldi iteration.
// solTol: Relative allowable magnitude of the residual, 
// difference between true and approximate images.
// numIts: Number of iterations used by the solver.
// relRes: Resulting relative solution residuals.
// Returns 0 on successful exit, 1 on failure. 
int invSolve(int* sysIds, int sysSize, int *deflateMode, double *solTol, int *numIts, double *relRes);
// Convert the total current density found by the solver into green function field.
int fieldConvert(int *sysIds, int sysSize);
// System function, do not call directly, do not remove from header.
int opr(int *sysIds, int sysSize);
// Returns free device memory in bytes
size_t devMem(int numDevs, int* devList);
#ifdef __cplusplus
}
#endif