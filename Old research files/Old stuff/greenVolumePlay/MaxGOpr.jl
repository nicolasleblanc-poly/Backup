"""
The MaxGOpr module provides support functions for purely CPU implementation of 
the embedded circulant form Green function calculated by MaxGCirc. Presently, 
this module exists solely for testing purposes and no documentation is provided.
The code is distributed under GNU LGPL.

Author: Sean Molesky 
"""
module MaxGOpr
using Base.Threads, MaxGStructs, AbstractFFTs
export blockGreenItr, grnOpr!, grnAdjOpr!
# Write full green function matrix from greenCirc. 
function grnFull!()

	cellsX = cellsA[1]
	cellsY = cellsA[2]
	cellsZ = cellsA[3]
	totCells = prod(cellsA)

	overCountX = 0
	overCountY = 0
	overCountZ = 0



	### Create first three columns.
	# x-source
	greenFull[1 : totCells, 1] = reshape(greenCirc[1, 1, 1 : cellsX, 
		1 : cellsY, 1 : cellsZ], (totCells, 1))

	greenFull[(totCells + 1) : (2 * totCells), 1] = reshape(greenCirc[2, 1, 
		1 : cellsX, 1 : cellsY, 1 : cellsZ], (totCells, 1))

	greenFull[(2 * totCells + 1) : (3 * totCells), 1] = reshape(greenCirc[3, 1, 
		1 : cellsX, 1 : cellsY, 1 : cellsZ], (totCells, 1))
	# y-source
	greenFull[1 : totCells, 2] = reshape(greenCirc[1, 2, 1 : cellsX, 
		1 : cellsY, 1 : cellsZ], (totCells, 1))

	greenFull[(totCells + 1) : (2 * totCells), 2] = reshape(greenCirc[2, 2, 
		1 : cellsX, 1 : cellsY, 1 : cellsZ], (totCells, 1))

	greenFull[(2 * totCells + 1) : (3 * totCells), 2] = reshape(greenCirc[3, 2, 
		1 : cellsX, 1 : cellsY, 1 : cellsZ], (totCells, 1))
	# z-source
	greenFull[1 : totCells, 3] = reshape(greenCirc[1, 3, 1 : cellsX, 
		1 : cellsY, 1 : cellsZ], (totCells, 1))

	greenFull[(totCells + 1) : (2 * totCells), 3] = reshape(greenCirc[2, 3, 
		1 : cellsX, 1 : cellsY, 1 : cellsZ], (totCells, 1))

	greenFull[(2 * totCells + 1) : (3 * totCells), 3] = reshape(greenCirc[3, 3, 
		1 : cellsX, 1 : cellsY, 1 : cellsZ], (totCells, 1))

	### Create first three rows.
	greenFull[1 : totCells, 1]


end



# Effective Green function operator. 
function grnOpr!(fftPlanFwd::AbstractFFTs.Plan{ComplexF64}, 
	fftPlanInv::AbstractFFTs.Plan{ComplexF64},
	grnCF::Union{Array{ComplexF64},SubArray{ComplexF64}}, 
	trgSize::NTuple{3,Int64}, crcSize::NTuple{3,Int64}, 
	srcSize::NTuple{3,Int64}, 
	currTrg::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	currSumEmbd::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	currTrgEmbd::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	currSrc::Union{Array{ComplexF64}, SubArray{ComplexF64}})::Nothing
	# Prepare embedded vectors, performing forward FFTs. 
	embdSow!(fftPlanFwd, crcSize, srcSize, currSumEmbd, currSrc)
	# Green function multiplications
	blockItr = 0
	greenItr = 0

	for colItr in 1 : 3, rowItr in 1 : 3

		blockItr = rowItr + (colItr - 1) * 3
		greenItr = blockGreenItr(blockItr)

		thrdMult!(0, prod(crcSize), view(currTrgEmbd, :, blockItr), 
			view(grnCF, :, :, :, greenItr), view(currSumEmbd, :, colItr))
	end
	# Collect results, performing inverse FFTs.
	embdReap!(fftPlanInv, trgSize, crcSize, currTrg, currSumEmbd, currTrgEmbd)
	return nothing
end
# Effective adjoint of the Green function operator. 
function grnAdjOpr!(fftPlanFwd::AbstractFFTs.Plan{ComplexF64}, 
	fftPlanInv::AbstractFFTs.Plan{ComplexF64},
	grnCF::Union{Array{ComplexF64},SubArray{ComplexF64}}, 
	trgSize::NTuple{3,Int64}, crcSize::NTuple{3,Int64}, 
	srcSize::NTuple{3,Int64}, 
	currTrg::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	currSumEmbd::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	currTrgEmbd::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	currSrc::Union{Array{ComplexF64}, SubArray{ComplexF64}})::Nothing
	# Prepare embedded vectors, performing forward FFTs. 
	embdSow!(fftPlanFwd, crcSize, srcSize, currSumEmbd, currSrc)
	# Green function multiplications
	blockItr = 0
	greenItr = 0

	for colItr in 1 : 3, rowItr in 1 : 3

		blockItr = rowItr + (colItr - 1) * 3
		greenItr = blockGreenItr(blockItr)
		# Here making use of the fact that the Green function is symmetric 
		# under the transpose in real space. 
		thrdMult!(1, prod(crcSize), view(currTrgEmbd, :, blockItr), 
			view(grnCF, :, :, :, greenItr), view(currSumEmbd, :, colItr))
	end
	# Collect results, performing inverse FFTs.
	embdReap!(fftPlanInv, trgSize, crcSize, currTrg, currSumEmbd, currTrgEmbd)
	return nothing
end
# Collect circulant currents and calculate the resulting projection. 
function embdReap!(fftPlanInv::AbstractFFTs.Plan{ComplexF64}, 
	trgSize::NTuple{3,Int64}, crcSize::NTuple{3,Int64}, 
	trgMem::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	currSumEmbd::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	currTrgEmbd::Union{Array{ComplexF64}, SubArray{ComplexF64}})::Nothing
	# Sum components in preparation for inverse Fourier transform
	for dirItr in 1 : 3

		@threads for posItr in 1 : prod(crcSize) 

			currSumEmbd[posItr, dirItr] =  currTrgEmbd[posItr, dirItr] + 
			currTrgEmbd[posItr, dirItr + 3] + currTrgEmbd[posItr, dirItr + 6]
		end
	end
	# Preform inverse Fourier transforms
	for dirItr in 1 : 3

		currSumEmbd[:, dirItr] = reshape((fftPlanInv * 
			reshape(currSumEmbd[:, dirItr], crcSize)), prod(crcSize))
	end
	# Project out of circulant form
	for dirItr in 1 : 3
		
		projVec!(trgSize, crcSize, view(trgMem, :, :, :, dirItr), 
			view(currSumEmbd, :, dirItr))
	end
end
# Create embedded circulant currents in preparation for Green operation. 
function embdSow!(fftPlanFwd::AbstractFFTs.Plan{ComplexF64},
	crcSize::NTuple{3,Int64}, srcSize::NTuple{3,Int64}, 
	currEmbd::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	currSrc::Union{Array{ComplexF64}, SubArray{ComplexF64}})::Nothing
	
	for dirItr in 1 : 3

		embdVec!(crcSize, srcSize, view(currEmbd, :, dirItr), 
			view(currSrc, :, :, :, dirItr))
	end
	
	for dirItr in 1 : 3

		currEmbd[:,dirItr] = reshape(fftPlanFwd * reshape(currEmbd[:,dirItr], 
			crcSize), prod(crcSize))
	end

end
# Circulant projection for a single Cartesian direction. 
function projVec!(trgSize::NTuple{3,Int64}, crcSize::NTuple{3,Int64}, 
	trgMem::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	embMem::Union{Array{ComplexF64}, SubArray{ComplexF64}})::Nothing
	# Get number of active threads to assign memory. 
	numThreads = nthreads()
	# Local cell counters
	cellX = Array{Int64}(undef, numThreads)	
	cellY = Array{Int64}(undef, numThreads)	
	cellZ = Array{Int64}(undef, numThreads)	
	# Project vector
	@threads for itr in 0 : (prod(crcSize) - 1)
		# Linear index to Cartesian index
		cellX[threadid()] = mod(itr, crcSize[1])
		cellY[threadid()] = div(mod(itr - cellX[threadid()], 
			crcSize[1] * crcSize[2]), crcSize[1])
		cellZ[threadid()] = div(itr - cellX[threadid()] - 
			(cellY[threadid()] * crcSize[1]), crcSize[1] * crcSize[2])

		if ((cellX[threadid()] < trgSize[1]) && 
			(cellY[threadid()] < trgSize[2]) && 
			(cellZ[threadid()] < trgSize[3]))

			trgMem[cellX[threadid()] + 1, cellY[threadid()] + 1, 
			cellZ[threadid()] + 1] = embMem[itr + 1]
		end
	end
end
# Threaded vector multiplication 
@inline function thrdMult!(mode::Int64, size::Int64, 
	trgMem::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	srcAMem::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	srcBMem::Union{Array{ComplexF64}, SubArray{ComplexF64}})::Nothing
	
	if mode == 0
		
		@threads for itr in 1 : size 

			trgMem[itr] = srcAMem[itr] * srcBMem[itr]
		end
	
	elseif mode == 1

		@threads for itr in 1 : size 

			trgMem[itr] = conj(srcAMem[itr]) * srcBMem[itr]
		end
	
	elseif mode == 2

		@threads for itr in 1 : size 

			trgMem[itr] = conj(srcAMem[itr]) * conj(srcBMem[itr])
		end

	else

		error("Improper use case.")
	end
end
# Circulant embedding for a single Cartesian direction. 
function embdVec!(crcSize::NTuple{3,Int64}, srcSize::NTuple{3,Int64}, 
	embMem::Union{Array{ComplexF64}, SubArray{ComplexF64}}, 
	srcMem::Union{Array{ComplexF64}, SubArray{ComplexF64}})::Nothing
	# Get number of active threads to assign memory. 
	numThreads = nthreads()
	# Local cell counters
	cellX = Array{Int64}(undef, numThreads)	
	cellY = Array{Int64}(undef, numThreads)	
	cellZ = Array{Int64}(undef, numThreads)	

	@threads for itr in 0 : (prod(crcSize) - 1)
		# Linear index to Cartesian index
		cellX[threadid()] = mod(itr, crcSize[1])
		cellY[threadid()] = div(mod(itr - cellX[threadid()], 
			crcSize[1] * crcSize[2]), crcSize[1])
		cellZ[threadid()] = div(itr - cellX[threadid()] - 
			(cellY[threadid()] * crcSize[1]), crcSize[1] * crcSize[2])

		if ((cellX[threadid()] < srcSize[1]) && 
			(cellY[threadid()] < srcSize[2]) && 
			(cellZ[threadid()] < srcSize[3]))

			embMem[itr + 1] = srcMem[cellX[threadid()] + 1, 
			cellY[threadid()] + 1, cellZ[threadid()] + 1] 
		
		else

			embMem[itr + 1] = 0.0 + 0.0im
		end
	end
end
# Get Green block index for a given Cartesian index.
@inline function blockGreenItr(cartInd::Int64)::Int64

	if cartInd == 1

		return 1

	elseif cartInd == 2 || cartInd == 4

		return 4

	elseif cartInd == 5 

		return 2
	
	elseif cartInd == 7 || cartInd == 3

		return 5

	elseif cartInd == 8 || cartInd == 6

		return 6

	elseif cartInd == 9 

		return 3

	else

		error("Improper use case.")
		return 0
	end
end
end