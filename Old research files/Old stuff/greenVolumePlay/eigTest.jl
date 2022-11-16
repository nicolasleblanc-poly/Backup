using Random, MaxGOpr
### Prepare for positive semi-definiteness test.
## Pre-allocate memory 
# Fourier transform of the Green function, making use of real space symmetry 
# under transposition. Entries are xx, yy, zz, xy, xz, yz
grnCF = Array{ComplexF64}(undef, 2 * cellsA[1], 2 * cellsA[2], 2 * cellsA[3], 6)
# Conjugate of the Fourier transform of the Green function
grnCFConj = Array{ComplexF64}(undef, 2 * cellsA[1], 2 * cellsA[2], 
	2 * cellsA[3], 9)
# Target embedding working memory
currTrgEmbd = Array{ComplexF64}(undef, 8 * cellsA[1] * cellsA[2] * cellsA[3], 9)
# Source embedding sum memory
currSumEmbd = Array{ComplexF64}(undef, 8 * cellsA[1] * cellsA[2] * cellsA[3], 3)
# Source current memory
currSrc = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
# Target current memory
currTrg = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
# Target current memory for adjoint operation
currAdjTrg = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
# Working memory for target current
currWrkA = Array{ComplexF64}(undef, cellsA[1] * cellsA[2] * cellsA[3] * 3)
currWrkB = Array{ComplexF64}(undef, cellsA[1] * cellsA[2] * cellsA[3] * 3) 
# shouldn't this be with the cells of B?

## Size settings for self Green function
srcSize = (cellsA[1], cellsA[2], cellsA[3])
crcSize = (2 * cellsA[1], 2 * cellsA[2], 2 * cellsA[3])
trgSize = (cellsA[1], cellsA[2], cellsA[3])
# total cells 
totCells = prod(srcSize)

## Plan in-place 3D Fast-Fourier transform
fftPlanFwd = plan_fft!(greenCircAA[1,1,:,:,:],(1,2,3))
fftPlanInv = plan_ifft!(greenCircAA[1,1,:,:,:],(1,2,3))
fftPlanFwdOut = plan_fft(greenCircAA[1,1,:,:,:],(1,2,3))
fftPlanInvOut = plan_ifft(greenCircAA[1,1,:,:,:],(1,2,3))

## Preform Fast-Fourier transforms of circulant Green functions
greenItr = 0
blockItr = 0

for colInd in 1 : 3, rowInd in 1 : colInd

	global blockItr = 3 * (colInd - 1) + rowInd
	global greenItr = blockGreenItr(blockItr)

	grnCF[:,:,:,greenItr] =  fftPlanFwdOut * greenCircAA[rowInd,colInd,:,:,:]
end

### Declare operators
greenAct! = () -> grnOpr!(fftPlanFwd, fftPlanInv, grnCF, trgSize, crcSize,
	srcSize, currTrg, currSumEmbd, currTrgEmbd, currSrc)

greenAdjAct! = () -> grnAdjOpr!(fftPlanFwd, fftPlanInv, grnCF, trgSize, crcSize,
	srcSize, currAdjTrg, currSumEmbd, currTrgEmbd, currSrc)


# # Zero source current in preparation for use
# for dirItr in 1 : 3
	
# 	for itrZ in 1 : srcSize[3], itrY in 1 : srcSize[2], itrX in 1 : srcSize[1]

# 		currSrc[itrX,itrY,itrZ,dirItr] = 0.0 + 0.0im
# 	end
# end
# # function for checking current
# function currRearrange(currOut::Array{ComplexF64,1},currIn::Array{ComplexF64,4})::Nothing

# 	sizeX = size(currIn)[1]
# 	sizeY = size(currIn)[2]
# 	sizeZ = size(currIn)[3]

# 	linCell = 0


# 	for itrZ in 1 : sizeZ, itrY in 1 : sizeY, itrX in 1 : sizeX, dirItr in 1 : 3

# 		global linCell = dirItr + (itrX - 1) * 3 + (itrY - 1) * sizeX * 3 + (itrZ - 1) * sizeX * sizeY * 3

# 		currOut[linCell] = currIn[itrX, itrY, itrZ, dirItr]

# 	end
# end
# ## asym test valid for cellsA = [2,2,1] only

# gMat = Array{ComplexF64}(undef, 12, 12)

# gMat[1 : 3, 1 : 3] = greenCircAA[:,:,1,1,1]
# gMat[4 : 6, 4 : 6] = greenCircAA[:,:,1,1,1]
# gMat[7 : 9, 7 : 9] = greenCircAA[:,:,1,1,1]
# gMat[10 : 12, 10 : 12] = greenCircAA[:,:,1,1,1]

# gMat[4 : 6, 1 : 3] = greenCircAA[:,:,2,1,1]
# gMat[1 : 3, 4 : 6] = greenCircAA[:,:,2,1,1]

# gMat[7 : 9, 1 : 3] = greenCircAA[:,:,1,2,1]
# gMat[1 : 3, 7 : 9] = greenCircAA[:,:,1,2,1]

# gMat[10 : 12, 1 : 3] = greenCircAA[:,:,2,2,1]
# gMat[1 : 3, 10 : 12] = greenCircAA[:,:,2,2,1]

# gMat[7 : 9, 4 : 6] = greenCircAA[:,:,2,2,1]
# gMat[4 : 6, 7 : 9] = greenCircAA[:,:,2,2,1]

# gMat[10 : 12, 4 : 6] = greenCircAA[:,:,1,2,1]
# gMat[4 : 6, 10 : 12] = greenCircAA[:,:,1,2,1]

# gMat[10 : 12, 7 : 9] = greenCircAA[:,:,2,1,1]
# gMat[7 : 9, 10 : 12] = greenCircAA[:,:,2,1,1]

# gMatAsym = (gMat .- adjoint(gMat)) ./ (2.0im)
# ## Begin adjoint test
# testSize = 1
# adjNorms = Array{ComplexF64}(undef, testSize) 

# # for itr = 1 : testSize

# # 	rand!(currSrc)	
# # end
# currNrm = Array{ComplexF64}(undef, testSize)
# asymValsA = Array{ComplexF64}(undef, testSize)
# asymValsB = Array{ComplexF64}(undef, testSize)

# # for itr = 1 : testSize
# currSrc[1,2,1,1] = 1.0 + 1.0im
# # currSrc[1,1,1,1] = 2.0 + 2.0im

# 	# rand!(currSrc)
# 	greenAct!()
# 	# greenAdjAct!()
# 	currNrm[1] = real(dotc(totCells, currSrc, 1, currSrc, 1))
# 	asymValsA[1] = imag(dotc(totCells, currSrc, 1, currTrg, 1)) / currNrm[1]
# 	# asymValsB[1] = dotc(totCells, currSrc, 1, (currTrg - currAdjTrg) / 2.0im, 1)
# # end

# # currRearrange(currWrkA,currTrg)

# # currRearrange(currWrkB,currSrc)

# # (gMat * currWrkB) - currWrkA
