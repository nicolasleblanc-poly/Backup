module product 
# using Random, MaxGOpr, FFTW, Distributed, MaxGParallelUtilities, MaxGStructs, 
# MaxGBasisIntegrals, MaxGCirc, LinearAlgebra, MaxGCUDA,  MaxGOpr

using MaxGOpr
using Distributed
@everywhere using Printf, Base.Threads, LinearAlgebra.BLAS, MaxGParallelUtilities, MaxGStructs, 
MaxGBasisIntegrals, MaxGCirc, LinearAlgebra, MaxGCUDA,  MaxGOpr, product, Random, FFTW
export Gv_AA, GAdjv_AA, Gv_AB #, A, asym, sym, asym_vect

# Function of the (G_{AA})v product 
function Gv_AA(greenCircAA, cellsA, vect) # offset
    ### Prepare for positive semi-definiteness test.
	## Pre-allocate memory 
	# Fourier transform of the Green function, making use of real space symmetry 
	# under transposition. Entries are xx, yy, zz, xy, xz, yz
	# Self Green function 
	grnCFAA = Array{ComplexF64}(undef, 2 * cellsA[1], 2 * cellsA[2], 2 * cellsA[3], 6)	
	# Embedded target 
	currTrgEmbdAA = Array{ComplexF64}(undef, 8 * cellsA[1] * cellsA[2] * cellsA[3], 3)
	# Embedded source memory
	currSrcEmbdAA = Array{ComplexF64}(undef, 8 * cellsA[1] * cellsA[2] * cellsA[3], 3)
	# Source current memory -> This is replaced by vect 
	# currSrcAA = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
	# Target current memory
	currTrgAA = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)

	## Size settings for self Green function
	srcSizeAA = (cellsA[1], cellsA[2], cellsA[3])
	crcSizeAA = (2 * cellsA[1], 2 * cellsA[2], 2 * cellsA[3])
	trgSizeAA = (cellsA[1], cellsA[2], cellsA[3])
	
	# total cells 
	totCellsAA = prod(srcSizeAA)

	## Plan in-place 3D Fast-Fourier transform
	fftPlanFwdAA = plan_fft!(greenCircAA[1,1,:,:,:],(1,2,3))
	fftPlanInvAA = plan_ifft!(greenCircAA[1,1,:,:,:],(1,2,3))
	fftPlanFwdOutAA = plan_fft(greenCircAA[1,1,:,:,:],(1,2,3))

	## Preform Fast-Fourier transforms of circulant Green functions
	greenItr = 0
	blockItr = 0

	for colInd in 1 : 3, rowInd in 1 : colInd

		global blockItr = 3 * (colInd - 1) + rowInd
		global greenItr = blockGreenItr(blockItr)

		grnCFAA[:,:,:,greenItr] =  fftPlanFwdOutAA * greenCircAA[rowInd,colInd,:,:,:]
	end

	# Apply the Green operator 
	grnOpr!(fftPlanFwdAA, fftPlanInvAA, grnCFAA, trgSizeAA, 
	crcSizeAA, srcSizeAA, currTrgAA, currTrgEmbdAA, currWrkEmbdAA, 
	currSrcEmbdAA, vect) # replaced the last output (currSrcAA) by vect
	
	mode = 1
    return currTrgAA + G_offset(vect, mode) #offset 
end 

function G_offset(v, mode) # offset, 
	offset = 1e-4im
    if mode == offset # Double check this part of the code with Sean because I feel like the condition
		# mode == offset is never met seeing that mode is either 1 or 2.  
		return (offset).*v
	else 
		return conj(offset).*v
	end
end

# Function of the (G_{AB}^A)v product 
function GAdjv_AA(greenCircAA, cellsA, vect) # , offset
    ### Prepare for positive semi-definiteness test.
	## Pre-allocate memory 
	# Fourier transform of the Green function, making use of real space symmetry 
	# under transposition. Entries are xx, yy, zz, xy, xz, yz
	# Conjugate of the Fourier transform of the Green function
	grnCFConjAA = Array{ComplexF64}(undef, 2 * cellsA[1], 2 * cellsA[2], 2 * cellsA[3], 6)
	#  Work area on target side
	cAdjWrkEmbdAA = Array{ComplexF64}(undef, 8 * cellsA[1] * cellsA[2] * cellsA[3], 9)
	# Embedded target 
	cAdjTrgEmbdAA = Array{ComplexF64}(undef, 8 * cellsA[1] * cellsA[2] * cellsA[3], 3)
	# Embedded source memory
	cAdjSrcEmbdAA = Array{ComplexF64}(undef, 8 * cellsA[1] * cellsA[2] * cellsA[3], 3)
	# Source current memory -> This is replaced by vect 
	# cAdjSrcAA = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
	# Target current memory
	cAdjTrgAA = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)

	## Size settings for self Green function
	srcSizeAA = (cellsA[1], cellsA[2], cellsA[3])
	crcSizeAA = (2 * cellsA[1], 2 * cellsA[2], 2 * cellsA[3])
	trgSizeAA = (cellsA[1], cellsA[2], cellsA[3])	
	# total cells 
	totCellsAA = prod(srcSizeAA)

	## Plan in-place 3D Fast-Fourier transform
	fftPlanFwdAA = plan_fft!(greenCircAA[1,1,:,:,:],(1,2,3))
	fftPlanInvAA = plan_ifft!(greenCircAA[1,1,:,:,:],(1,2,3))
	fftPlanFwdOutAA = plan_fft(greenCircAA[1,1,:,:,:],(1,2,3))

	## Preform Fast-Fourier transforms of circulant Green functions
	greenItr = 0
	blockItr = 0

	for colInd in 1 : 3, rowInd in 1 : colInd

		global blockItr = 3 * (colInd - 1) + rowInd
		global greenItr = blockGreenItr(blockItr)

		grnCFConjAA[:,:,:,greenItr] =  fftPlanFwdOutAA * greenCircAA[rowInd,colInd,:,:,:]
	end

	# Old code 
	# grnAdjOpr!(fftPlanFwd, fftPlanInv, grnCF, trgSize, crcSize,
	# srcSize, currAdjTrg, currSumEmbd, currTrgEmbd, vect)

	# New code 
	grnAdjOpr!(fftPlanFwdAA, fftPlanInvAA, grnCFAA, 
	trgSizeAA, crcSizeAA, srcSizeAA, cAdjTrgAA, cAdjTrgEmbdAA, cAdjWrkEmbdAA, 
	cAdjSrcEmbdAA, vect) # replaced the last output (cAdjSrcAA) by vect
	
	mode = 2 
    return cAdjTrgAA + G_offset(vect, mode) # offset, 
end

# Function of the (G_{AB})v product 
function Gv_AB(greenCircAB, cellsA, cellsB, vect) # offset
    ### Prepare for positive semi-definiteness test.
	## Pre-allocate memory 
	# Fourier transform of the Green function, making use of real space symmetry 
	# under transposition. Entries are xx, yy, zz, xy, xz, yz
	# External Green function
	grnCFAB = Array{ComplexF64}(undef, cellsA[1] + cellsB[1], cellsA[2] + cellsB[2], 
	cellsA[3] + cellsB[3], 6)

	## External Green
	#  Work area on target side
	currWrkEmbdAB = Array{ComplexF64}(undef, 8 * cellsA[1] * cellsA[2] * cellsA[3], 9)
	# Embedded target 
	currTrgEmbdAB = Array{ComplexF64}(undef, 8 * cellsA[1] * cellsA[2] * cellsA[3], 3)
	# Embedded source memory
	currSrcEmbdAB = Array{ComplexF64}(undef, 8 * cellsB[1] * cellsB[2] * cellsB[3], 3)
	# Source current memory
	# currSrcAB = Array{ComplexF64}(undef, cellsB[1], cellsB[2], cellsB[3], 3)
	# Target current memory
	currTrgAB = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)

	## Size settings for external Green function
	srcSizeAB = (cellsB[1], cellsB[2], cellsB[3])
	crcSizeAB = (cellsA[1] + cellsB[1], cellsA[2] + cellsB[2], cellsA[3] + cellsB[3])
	trgSizeAB = (cellsA[1], cellsA[2], cellsA[3])
	
	# total cells 
	totCellsBB = prod(srcSizeAB)

	## Plan in-place 3D Fast-Fourier transform
	fftPlanFwdAB = plan_fft!(greenCircAB[1,1,:,:,:],(1,2,3))
	fftPlanInvAB = plan_ifft!(greenCircAB[1,1,:,:,:],(1,2,3))
	fftPlanFwdOutAB = plan_fft(greenCircAB[1,1,:,:,:],(1,2,3))

	## Preform Fast-Fourier transforms of circulant Green functions
	greenItr = 0
	blockItr = 0

	for colInd in 1 : 3, rowInd in 1 : colInd

		global blockItr = 3 * (colInd - 1) + rowInd
		global greenItr = blockGreenItr(blockItr)

		grnCFAB[:,:,:,greenItr] =  fftPlanFwdOutAB * greenCircAB[rowInd,colInd,:,:,:]
	end

	# Apply the Green operator 
	grnOpr!(fftPlanFwdAB, fftPlanInvAB, grnCFAB, trgSizeAB, 
	crcSizeAB, srcSizeAB, currTrgAB, currTrgEmbdAB, currWrkEmbdAB, 
	currSrcEmbdAB, vect) # replaced the last output (currSrcAB) by vect
	
	# This will do
	mode = 1
    return currTrgAB + G_offset(vect, mode) # I assume I have to add the offset like for the AA G func 
end 


# function A(greenCircAA, cellsA, chi_inv_coeff, l, P, vect)
# 	# For when we will also consider the symmetric part + l[2]sym(greenCircAA, cellsA, chi_inv_coeff, P))
# 	return l[1]*(asym(greenCircAA, cellsA, chi_inv_coeff, P))
# end

# function asym(greenCircAA, cellsA, chi_inv_coeff, P)
# 	chi_inv_coeff_dag = conj(chi_inv_coeff)
# 	term_1 = chi_inv_coeff_dag*P 
# 	term_2 = GAdjv_AA(greenCircAA, cellsA, P)
# 	term_3 = chi_inv_coeff*conj.(transpose(P))
# 	term_4 = Gv_AA(greenCircAA, cellsA, P)
# 	return (term_1-term_2-term_3+term_4)/2im
# end

# function sym(greenCircAA, cellsA, chi_inv_coeff, P)
# 	chi_inv_coeff_dag = conj(chi_inv_coeff)
# 	term_1 = chi_inv_coeff_dag*P 
# 	term_2 = GAdjv_AA(greenCircAA, cellsA, P)
# 	term_3 = chi_inv_coeff*conj.(transpose(P))
# 	term_4 = Gv_AA(greenCircAA, cellsA, P)
# 	return (term_1-term_2-term_3+term_4)/2
# end

# function asym_vect(greenCircAA, cellsA, chi_inv_coeff, l, P, vect)
# 	chi_inv_coeff_dag = conj(chi_inv_coeff)
# 	term_1 = chi_inv_coeff_dag*P 
# 	term_2 = GAdjv_AA(greenCircAA, cellsA, vect) # P*
# 	term_3 = chi_inv_coeff*P # supposed to be *conj.(transpose(P)) but gives error, so let's use *P for now
# 	term_4 = Gv_AA(greenCircAA, cellsA, vect) # P*
# 	return (.-term_2.+ term_4)./2im # term_1  .-term_3
# end


end



# function Av(greenCircAA, cellsA, chi_inv_coeff, l, P, vect)
# 	chi_inv_coeff_dag = conj(chi_inv_coeff)
# 	term_1 = chi_inv_coeff_dag*P 
# 	term_2 = GAdjv(greenCircAA, cellsA, P)
# 	term_3 = chi_inv_coeff*conj.(transpose(P))
# 	term_4 = Gv(greenCircAA, cellsA, P)
# end 

# # Define test volume, all lengths are defined relative to the wavelength. 
# # Number of cells in the volume. 
# cellsA = [16, 16, 16]
# cellsB = [1, 1, 1]
# currSrc = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)

# # Edge lengths of a cell relative to the wavelength. 
# scaleA = (0.02, 0.02, 0.02)
# scaleB = (0.02, 0.02, 0.02)
# # Center position of the volume. 
# coordA = (0.0, 0.0, 0.0)
# coordB = (-0.3, 0.3, 0.0)
# # Create MaxG volumes.
# volA = genMaxGVol(MaxGDom(cellsA, scaleA, coordA))
# volB = genMaxGVol(MaxGDom(cellsB, scaleB, coordB))
# # Information for Green function construction. 
# # Complex frequency ratio. 
# freqPhase = 1.0 + im * 0.0
# # Gauss-Legendre approximation orders. 
# ordGLIntFar = 2
# ordGLIntMed = 8
# ordGLIntNear = 16
# # Cross over points for Gauss-Legendre approximation.
# crossMedFar = 16
# crossNearMed = 8
# assemblyInfo = MaxGAssemblyOpts(freqPhase, ordGLIntFar, ordGLIntMed, 
# 	ordGLIntNear, crossMedFar, crossNearMed)
# # Pre-allocate memory for circulant green function vector. 
# # Let's say we are only considering the AA case for simplicity
# greenCircAA = Array{ComplexF64}(undef, 3, 3, 2 * cellsA[1], 2 * cellsA[2], 
# 	2 * cellsA[3])
# greenCircBA = Array{ComplexF64}(undef, 3, 3, cellsB[1] + cellsA[1], cellsB[2] +
# 	cellsA[2], cellsB[3] + cellsA[3])

# Nic Test:
# CPU computation of Green function
# genGreenSlf!(greenCircAA, volA, assemblyInfo)

# rand!(currSrc)

# print(greenAct!())

# print(Gv(greenCircAA, cellsA, currSrc))