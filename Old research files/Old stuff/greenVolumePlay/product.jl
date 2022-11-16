module product
export G_v_prod, GAdj_v_prod
using Random, MaxGOpr, FFTW
### Prepare for positive semi-definiteness test.
## Pre-allocate memory 
# # Fourier transform of the Green function, making use of real space symmetry 
# # under transposition. Entries are xx, yy, zz, xy, xz, yz
# grnCF = Array{ComplexF64}(undef, 2 * cellsA[1], 2 * cellsA[2], 2 * cellsA[3], 6)
# # Conjugate of the Fourier transform of the Green function
# grnCFConj = Array{ComplexF64}(undef, 2 * cellsA[1], 2 * cellsA[2], 
# 	2 * cellsA[3], 9)
# # Target embedding working memory
# currTrgEmbd = Array{ComplexF64}(undef, 8 * cellsA[1] * cellsA[2] * cellsA[3], 9)
# # Source embedding sum memory
# currSumEmbd = Array{ComplexF64}(undef, 8 * cellsA[1] * cellsA[2] * cellsA[3], 3)
# # Source current memory
# currSrc = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
# # Target current memory
# currTrg = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
# # Target current memory for adjoint operation
# currAdjTrg = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
# # Working memory for target current
# currWrkA = Array{ComplexF64}(undef, cellsA[1] * cellsA[2] * cellsA[3] * 3)
# currWrkB = Array{ComplexF64}(undef, cellsA[1] * cellsA[2] * cellsA[3] * 3) 
# # shouldn't this be with the cells of B?

# ## Size settings for self Green function
# srcSize = (cellsA[1], cellsA[2], cellsA[3])
# crcSize = (2 * cellsA[1], 2 * cellsA[2], 2 * cellsA[3])
# trgSize = (cellsA[1], cellsA[2], cellsA[3])
# # total cells 
# totCells = prod(srcSize)

# ## Plan in-place 3D Fast-Fourier transform
# fftPlanFwd = plan_fft!(greenCircAA[1,1,:,:,:],(1,2,3))
# fftPlanInv = plan_ifft!(greenCircAA[1,1,:,:,:],(1,2,3))
# fftPlanFwdOut = plan_fft(greenCircAA[1,1,:,:,:],(1,2,3))
# fftPlanInvOut = plan_ifft(greenCircAA[1,1,:,:,:],(1,2,3))

# ## Preform Fast-Fourier transforms of circulant Green functions
# greenItr = 0
# blockItr = 0

# for colInd in 1 : 3, rowInd in 1 : colInd

# 	global blockItr = 3 * (colInd - 1) + rowInd
# 	global greenItr = blockGreenItr(blockItr)

# 	grnCF[:,:,:,greenItr] =  fftPlanFwdOut * greenCircAA[rowInd,colInd,:,:,:]
# end

# ### Declare operators
# greenAct! = () -> grnOpr!(fftPlanFwd, fftPlanInv, grnCF, trgSize, crcSize,
# 	srcSize, currTrg, currSumEmbd, currTrgEmbd, currSrc)

# greenAdjAct! = () -> grnAdjOpr!(fftPlanFwd, fftPlanInv, grnCF, trgSize, crcSize,
# 	srcSize, currAdjTrg, currSumEmbd, currTrgEmbd, currSrc)

function G_v_prod(greenCircAA, cellsA, currSrc) 
    currSrc = currSrc
	## Pre-allocate memory 
	# Fourier transform of the Green function, making use of real space symmetry 
	# under transposition. Entries are xx, yy, zz, xy, xz, yz
	grnCF = Array{ComplexF64}(undef, 2 * cellsA[1], 2 * cellsA[2], 2 * cellsA[3], 6)
	# Target embedding working memory
	currTrgEmbd = Array{ComplexF64}(undef, 8 * cellsA[1] * cellsA[2] * cellsA[3], 9)
	# Source embedding sum memory
	currSumEmbd = Array{ComplexF64}(undef, 8 * cellsA[1] * cellsA[2] * cellsA[3], 3)
	# Source current memory
	currSrc = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
	# Target current memory
	currTrg = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)

	## Size settings for self Green function
	srcSize = (cellsA[1], cellsA[2], cellsA[3])
	crcSize = (2 * cellsA[1], 2 * cellsA[2], 2 * cellsA[3])
	trgSize = (cellsA[1], cellsA[2], cellsA[3])

	## Plan in-place 3D Fast-Fourier transform
	fftPlanFwd = plan_fft!(greenCircAA[1,1,:,:,:],(1,2,3))
	fftPlanInv = plan_ifft!(greenCircAA[1,1,:,:,:],(1,2,3))
	fftPlanFwdOut = plan_fft(greenCircAA[1,1,:,:,:],(1,2,3))

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
    return greenAct!
end 

function GAdj_v_prod(greenCircAA, cellsA, currSrc)
    currSrc = currSrc
		## Pre-allocate memory 
	# Fourier transform of the Green function, making use of real space symmetry 
	# under transposition. Entries are xx, yy, zz, xy, xz, yz
	grnCF = Array{ComplexF64}(undef, 2 * cellsA[1], 2 * cellsA[2], 2 * cellsA[3], 6)
	# Target embedding working memory
	currTrgEmbd = Array{ComplexF64}(undef, 8 * cellsA[1] * cellsA[2] * cellsA[3], 9)
	# Source embedding sum memory
	currSumEmbd = Array{ComplexF64}(undef, 8 * cellsA[1] * cellsA[2] * cellsA[3], 3)
	# Source current memory
	currSrc = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
	# Target current memory for adjoint operation
	currAdjTrg = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)

	## Size settings for self Green function
	srcSize = (cellsA[1], cellsA[2], cellsA[3])
	crcSize = (2 * cellsA[1], 2 * cellsA[2], 2 * cellsA[3])
	trgSize = (cellsA[1], cellsA[2], cellsA[3])


	## Plan in-place 3D Fast-Fourier transform
	fftPlanFwd = plan_fft!(greenCircAA[1,1,:,:,:],(1,2,3))
	fftPlanInv = plan_ifft!(greenCircAA[1,1,:,:,:],(1,2,3))
	fftPlanFwdOut = plan_fft(greenCircAA[1,1,:,:,:],(1,2,3))

	## Preform Fast-Fourier transforms of circulant Green functions
	greenItr = 0
	blockItr = 0

	for colInd in 1 : 3, rowInd in 1 : colInd
		global blockItr = 3 * (colInd - 1) + rowInd
		global greenItr = blockGreenItr(blockItr)
		grnCF[:,:,:,greenItr] =  fftPlanFwdOut * greenCircAA[rowInd,colInd,:,:,:]
	end
	### Declare operators
	greenAdjAct! = () -> grnAdjOpr!(fftPlanFwd, fftPlanInv, grnCF, trgSize, crcSize,
	srcSize, currAdjTrg, currSumEmbd, currTrgEmbd, currSrc)
    return greenAdjAct!
end
end