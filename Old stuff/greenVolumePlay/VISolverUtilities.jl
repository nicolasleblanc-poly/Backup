"""
Utilities for implementing electromagnetic VI solvers
"""
module VISolverUtilities
using Printf, Base.Threads, LinearAlgebra, ParallelUtilities, VIStructs, VICu
export viSVD!, genHeatKer, bdyMask!, prfMask!, projCurr, embedCurr, svdSolver!, rPrint

"""
rPrint(offset::Int, rows::Int, cols::Int, array::AbstractArray{T} where {T <: Number})::Nothing

Simple array printing function assuming column ordering of array.
"""
function rPrintCol(cols::Int, rows::Int, colOffset::Int, rowOffset::Int, colSize::Int, rowSize::Int, array::AbstractArray{T} where {T <: Number})::Nothing

	@printf(stdout, "\n");

	offset = colOffset + rowOffset * col
	cellNum = 1

	for rowItr in 1:rows
		for colItr in 1:cols

			cellNum = (colItr - 1) * rows + rowItr 

			@printf(stdout, "%5.4f+%5.4fi ", 
				convert(Float64, real(array[cellNum])), 
				convert(Float64, imag(array[cellNum])))
		end
		@printf(stdout, "\n");
	end
	@printf(stdout, "\n");
	return;
end
"""

function embedCurr(smallCurr::AbstractArray{ComplexF64}, largeCurr::AbstractArray{ComplexF64}, bdyIds::Array{Int, 1}, sysInfo::VISysInfo)::Nothing

Embeds a smaller current vector into a larger you based on a bdyIds passed.
"""
function embedCurr(smallCurr::AbstractArray{ComplexF64}, largeCurr::AbstractArray{ComplexF64}, bdyIds::Array{Int, 1}, sysInfo::VISysInfo)::Nothing

	totalCellIndex = 4
	startPos = 0
	bdyElms = 0
	smallCurrOffset = 0
	
	for bdyItr in bdyIds
		startPos = sysInfo.cellList[bdyItr, 5]
		bdyElms = 3 * sysInfo.cellList[bdyItr, totalCellIndex]
		
		@threads for posItr in 1:bdyElms
			largeCurr[3 * startPos + posItr] = smallCurr[smallCurrOffset + posItr]
		end
		smallCurrOffset += bdyElms
	end
	return nothing
end
"""

    function projCurr(largeCurr::AbstractArray{ComplexF64}, smallCurr::AbstractArray{ComplexF64}, bdyIds::Array{Int, 1}, sysInfo::VISysInfo)::Nothing

Projects a larger current into a smaller collection of bodies based on the bdyIds passed to the 
function.
"""
function projCurr(largeCurr::AbstractArray{ComplexF64}, smallCurr::AbstractArray{ComplexF64}, bdyIds::Array{Int, 1}, sysInfo::VISysInfo)::Nothing

	totalCellIndex = 4
	startPos = 0
	bdyElms = 0
	smallCurrOffset = 0
	
	for bdyItr in bdyIds
		startPos = sysInfo.cellList[bdyItr, 5]
		bdyElms = 3 * sysInfo.cellList[bdyItr, totalCellIndex]
		
		@threads for posItr in 1:bdyElms
			smallCurr[smallCurrOffset + posItr] = largeCurr[3 * startPos + posItr]
		end
		smallCurrOffset += bdyElms
	end
	return nothing
end
"""

    function bdyMask!(viOptMode::VIOptMode, sysInfo::VISysInfo, bdyIds::Array{Int, 1}, viSystem::Array{VIObject, 1}, fieldArray::AbstractArray{ComplexF64, 1})

Sets current values inside the bounding body cell to zero if they fall outside the body, 
as specified by the mask in the body object.
"""
function bdyMask!(viOptMode::VIOptMode, sysInfo::VISysInfo, bdyIds::Array{Int, 1}, viSystem::Array{VIObject, 1}, fieldArray::AbstractArray{ComplexF64, 1})

	offset = 0
	bdyElms = 0
	bdyCells = 0
	totalCellIndex = 4
	
	if(viOptMode == VIStructs.rHeatTrnf::VIOptMode)

		for bdyItr in bdyIds
			bdyCells = sysInfo.cellList[bdyItr, totalCellIndex]
			bdyElms = 3 * bdyCells
			
			@threads for posItr in 1:bdyElms
				fieldArray[offset + posItr] = sqrt(imag(viSystem[bdyItr].elecSus[posItr])) * 
				fieldArray[offset + posItr]
			end

			offset += bdyElms
		end

	elseif(viOptMode == VIStructs.greenFunc::VIOptMode)

		for bdyItr in bdyIds
			bdyCells = sysInfo.cellList[bdyItr, totalCellIndex]
			bdyElms = 3 * bdyCells

			@threads for posItr in 1:bdyElms
				fieldArray[offset + posItr] = viSystem[bdyItr].mask[posItr] * fieldArray[offset + posItr]
			end
			offset += bdyElms
		end

	else
		error("Unrecognized VIOptMode for SVD solver.")	
	end
	return nothing
end
"""

    function prfMask!(sysInfo::VISysInfo, bdyIds::Array{Int, 1}, viSystem::Array{VIObject, 1}, fieldArray::AbstractArray{ComplexF64, 1})

Multiplies current values inside the bounding body cell by the inverse of the permittivity if they
fall inside the body, as specified by the mask in the body object.
"""
function prfMask!(sysInfo::VISysInfo, bdyIds::Array{Int, 1}, viSystem::Array{VIObject, 1}, fieldArray::AbstractArray{ComplexF64, 1})

	offset = 0
	bdyElms = 0
	bdyCells = 0
	totalCellIndex = 4
	

	for bdyItr in bdyIds
		bdyCells = sysInfo.cellList[bdyItr, totalCellIndex]
		bdyElms = 3 * bdyCells

		@threads for posItr in 1:bdyElms
			fieldArray[offset + posItr] = viSystem[bdyItr].mask[posItr] * fieldArray[offset + posItr] / (1.0 + viSystem[bdyItr].elecSus[posItr])
		end
		offset += bdyElms
	end

	return nothing
end
"""

    svdSolver!(sysIds::Array{Int,1}, sysInfo::VISysInfo, svdItr::Int, 
	solverOpts::VISolverOpts, deflateMode::Int, sysCharac::VISysCharac, sysCharacInd::Int,
	initCurrSys::AbstractArray{ComplexF64, 1}, totlCurrSys::AbstractArray{ComplexF64, 1})

Low level function for solving singular value decomposition of the electromagnetic Green function.
Calls GPU solver, called by svdController! and viSVD! functions.
"""
function svdSolver!(sysIds::Array{Int,1}, sysInfo::VISysInfo, svdItr::Int, 
	solverOpts::VISolverOpts, deflateMode::Int, sysCharac::VISysCharac, sysCharacInd::Int,
	initCurrSys::AbstractArray{ComplexF64, 2}, totlCurrSys::AbstractArray{ComplexF64, 2})::Int

	sysNum = length(sysIds)
	defMode = Array{Int, 1}(undef, sysNum)
	solTol = Array{Float64, 1}(undef, sysNum)
	# Use previous deflation space if previous iteration exists and was successful
	if((svdItr > sysNum) || (deflateMode != 0))
		deflateMode = 1
	else
		deflateMode = 0
	end

	for sysInd in 1:sysNum
		defMode[sysInd] = deflateMode
		solTol[sysInd] = sysInfo.solverInfo.relativeSolutionTolearance
	end
	# Solve system
	if(svdItr <= sysNum)

		@printf(stdout, "\n")		
		@time solverOut = solveVI!(sysInfo.viCuLibHandle, sysIds, defMode, solTol, 
			initCurrSys, totlCurrSys)

		if(solverOut[2][1] == -1)
			error("solveVI! failure.")
			return 1
		end
		@printf(stdout, "%d random sources solved.\n", sysNum)
		@printf(stdout, "Iterations:   Relative residuals:\n")

		for sysInd in 1:sysNum
			@printf(stdout, "%d             %f\n", solverOut[2][sysInd], solverOut[1][sysInd])
		end
		@printf(stdout, "\n")
	else
		solverOut = solveVI!(sysInfo.viCuLibHandle, sysIds, defMode, solTol, initCurrSys,
			totlCurrSys)

		if(solverOut[2][1] == -1)
			error("solveVI! failure.")
			return 1
		end
	end

	for sysInd in 1:sysNum
		
		if(isnan(solverOut[1][sysInd]) || (solverOut[1][sysInd] > solTol[sysInd]))
			error("svdSolver! has failed compute a solution system ", sysIds[sysInfo], ".")
			return 1
		end
		# Save total current pair 
		if(svdItr + (sysInd - 1) <= sysCharac.maxTrials)
			threadCpy!(view(totlCurrSys, :, sysInd), 
			view(sysCharac.totlCurrs[sysCharacInd], :, svdItr + (sysInd - 1)))
		end
	end
	# Compute electric field (totlCurrSys is now the electric field)
	fieldConvertVI!(sysInfo.viCuLibHandle, sysIds, totlCurrSys)
	
	if(svdItr % (10 * sysNum) == 0)
		@printf(stdout, "%d iterations completed.\n", svdItr + sysNum)
	end
	return 0
end
"""

    svdController!(sysIds::Array{Int, 1}, coordChn::Channel, termnChn::Channel, svdFlg::Int, deflateMode::Int, prevMaxItr::Int, sysInfo::VISysInfo, sysCharac::VISysCharac, sysCharacInd::Int, viSystem::Array{VIObject, 1}, srcBdyIds::Array{Int, 1}, srcElms::Int, trgBdyIds::Array{Int, 1}, trgElms::Int, solverOpts::VISolverOpts, totlFieldsTrg::AbstractArray{ComplexF64, 2})

Low level function for solving singular value decomposition of the electromagnetic Green function.
Calls GPU solver, called by svdMemPrep! and viSVD! functions.
"""
function svdController!(sysIds::Array{Int, 1}, coordChn::Channel, termnChn::Channel, svdFlg::Int, deflateMode::Int, prevMaxItr::Int, sysInfo::VISysInfo, sysCharac::VISysCharac, sysCharacInd::Int, viSystem::Array{VIObject, 1}, srcBdyIds::Array{Int, 1}, srcElms::Int, trgBdyIds::Array{Int, 1}, trgElms::Int, solverOpts::VISolverOpts, totlFieldsTrg::AbstractArray{ComplexF64, 2})

	svdItr = 1
	sysNum = length(sysIds)
	
	# Initialize additional memory
	initCurrSrc = zeros(ComplexF64, 3 * srcElms)
	initCurrSys = zeros(ComplexF64, 3 * sysInfo.totalCells, sysNum)
	totlCurrSys = zeros(ComplexF64, 3 * sysInfo.totalCells, sysNum)
	zeroCurrSys = zeros(ComplexF64, 3 * sysInfo.totalCells)

	while true
		state = fetch(termnChn)::Int
		(state == svdFlg) && break
		
		if(svdItr > sysCharac.maxTrials)
			error("svdController! Failure: Randomized SVD solver has exhausted allocated memory without reaching prescribed accuracy.")
			break
		end

		for sysItr in 1:sysNum
			
			if(svdItr + (sysItr - 1) <= prevMaxItr)
				# Get random current.
				threadCpy!(view(sysCharac.initCurrs[sysCharacInd], :, svdItr + sysItr - 1),
				initCurrSrc)
				# Use previous solution.
				threadCpy!(view(sysCharac.totlCurrs[sysCharacInd], :, svdItr + sysItr - 1), 
				view(totlCurrSys, :, sysItr))
			else
				# Get random currents
				threadCpy!(view(sysCharac.initCurrs[sysCharacInd], :, svdItr + sysItr - 1), 
				initCurrSrc)
				# Set initial guess to zero.
				threadCpy!(zeroCurrSys, view(totlCurrSys, :, sysItr)) 
			end

			if(solverOpts.prefacMode == 0)
				bdyMask!(VIStructs.greenFunc::VIOptMode, sysInfo, srcBdyIds, viSystem, 
				initCurrSrc)
			elseif(solverOpts.prefacMode == 1)
				prfMask!(sysInfo, srcBdyIds, viSystem, initCurrSrc)
			else
				error("svdController!: Unrecognized prefactor mode.")
			end
			# Load source
			embedCurr(initCurrSrc, view(initCurrSys, :, sysItr), srcBdyIds, sysInfo)
			# Perform inverse solve
		end

		state = fetch(termnChn)::Int
		(state == svdFlg) && break

		@sync begin
			@async if(svdSolver!(sysIds, sysInfo, svdItr, solverOpts, deflateMode, sysCharac, 
			sysCharacInd, initCurrSys, totlCurrSys) != 0)
				return nothing
			end

			if(svdItr > sysNum)

				for sysItr in 1:sysNum
					put!(coordChn, svdItr - (sysNum + 1) + sysItr)
				end
			end
		end

		state = fetch(termnChn)::Int
		(state == svdFlg) && break

		for sysItr in 1:sysNum
			projCurr(view(totlCurrSys, :, sysItr), view(totlFieldsTrg, :, svdItr), 
			trgBdyIds, sysInfo)
			# Modify body field values base on VI operational viOptMode setting.
			# Multiplies by mask if viOptMode is Green function.
			# Multiplies by sqrt of imaginary permittivity if viOptMode is heat transfer.
			bdyMask!(VIStructs.greenFunc::VIOptMode, sysInfo, trgBdyIds, viSystem, 
			view(totlFieldsTrg, :, svdItr))
			svdItr += 1
		end
	end
	@printf(stdout, "svdController completed, %d inverses calculated.\n", svdItr - 1)
	close(coordChn)
	close(termnChn)
	return nothing
end
"""

    basisOrthog!(coordChn::Channel, termnChn::Channel, svdFlg::Int, sysNum::Int, srcElms::Int, trgElms::Int, sysCharac::VISysCharac, sysCharacInd::Int, solverOpts::VISolverOpts, orthBasis::AbstractArray{ComplexF64, 2}, totlFieldsTrg::AbstractArray{ComplexF64, 2})::Nothing

Low level orthogonalization function, building a basis from solutions passed by the svdController!.
Called by svdMemPrep!, and in turn viSVD!.
"""
function basisOrthog!(coordChn::Channel, termnChn::Channel, svdFlg::Int, sysNum::Int, srcElms::Int, trgElms::Int, sysCharac::VISysCharac, sysCharacInd::Int, solverOpts::VISolverOpts, orthBasis::AbstractArray{ComplexF64, 2}, totlFieldsTrg::AbstractArray{ComplexF64, 2})::Nothing

	initNrm = 1.0
	projNrm = 1.0
	successCount = 0
	exitCount = solverOpts.svdExitCount
	maxItr = sysCharac.maxTrials
	successCond = solverOpts.svdAccuracy
	# Orthogonalization memory
	workFieldTrg = Array{ComplexF64, 1}(undef, trgElms)
	projCoeffs = Array{ComplexF64, 1}(undef, sysCharac.maxTrials)

	while true
		# Give termination signal if successful
		if(successCount == exitCount)
			svdItr = take!(termnChn)::Int
			put!(termnChn, svdFlg)

			for freeSVD in 1:sysNum

				if(isready(coordChn))
					svdItr = take!(coordChn)::Int
				else
					break
				end
			end
			break
		end

		svdItr = take!(coordChn)::Int
		(svdItr == maxItr) && break

		if(svdItr > 1)
			BLAS.gemv!('C', 1.0+0.0im, view(orthBasis, :, 1:(svdItr - 1)), view(totlFieldsTrg, :, svdItr), 0.0+0.0im, view(projCoeffs, 1:(svdItr - 1)))
			# Remove projection components
			threadCpy!(view(totlFieldsTrg, :, svdItr), workFieldTrg)
			BLAS.gemv!('N', -1.0+0.0im, view(orthBasis, :, 1:(svdItr - 1)), view(projCoeffs, 1:(svdItr - 1)), 1.0+0.0im, workFieldTrg)
			# Normalize vector, recording value
			initNrm = BLAS.nrm2(view(totlFieldsTrg, :, svdItr))
			projNrm = BLAS.nrm2(workFieldTrg)
			BLAS.scal!(trgElms, convert(ComplexF64, (1.0+0.0im) / projNrm), workFieldTrg, 1)	
		else
			threadCpy!(view(totlFieldsTrg, :, svdItr), workFieldTrg)
			initNrm = BLAS.nrm2(view(totlFieldsTrg, :, svdItr))
			projNrm = initNrm
			BLAS.scal!(trgElms, convert(ComplexF64, (1.0+0.0im) / projNrm), workFieldTrg, 1)
		end
		# Update exit condition
		if(convert(Float64, real(projNrm / initNrm)) < successCond)
			successCount += 1
		else
			successCount = 0
		end
		# Add orthogonalized vector to basis 
		threadCpy!(workFieldTrg, view(orthBasis, :, svdItr))
		sysCharac.trialInfo[sysCharacInd] += 1
	end
	return nothing
end
"""

   function svdMemPrep!(sysIds::Array{Int,1}, viSystem::Array{VIObject, 1}, sysInfo::VISysInfo, bdyPairs::Tuple{Array{Int, 1}, Array{Int, 1}}, deflateMode::Int, sysCharacInd::Int, sysCharac::VISysCharac, solverOpts::VISolverOpts, orthBasis::AbstractArray{ComplexF64, 2}, totlFields::AbstractArray{ComplexF64, 2})::Nothing

Coordination function for svdController! and basisOrthog!, returning the calculated fields and
orthogonal basis for use in the singular value decomposition of the Green function.
"""
function svdMemPrep!(sysIds::Array{Int,1}, viSystem::Array{VIObject, 1}, sysInfo::VISysInfo, bdyPairs::Tuple{Array{Int, 1}, Array{Int, 1}}, deflateMode::Int, sysCharacInd::Int, sysCharac::VISysCharac, solverOpts::VISolverOpts, orthBasis::AbstractArray{ComplexF64, 2}, totlFields::AbstractArray{ComplexF64, 2})::Nothing

	svdFlg = 0
	sysNum = length(sysIds)
	
	if(sysCharacInd == 1)
		trgBdyInd = 1
		srcBdyInd = 2
	elseif(sysCharacInd == 2)
		trgBdyInd = 2
		srcBdyInd = 1
	else
		error("Unrecognized solver directions. Valid inputs are 1 (forward) and 2 (reverse).")
	end
	totCellInd = 4
	termnChn = Channel{Int}(1)
	coordChn = Channel{Int}(2 * sysNum)
	prevMaxItr = (sysCharac.trialInfo[sysCharacInd] == 0) ? 0 : sysCharac.trialInfo[sysCharacInd]
	# Reset trial counts
	sysCharac.trialInfo[sysCharacInd] = 0
	# Array sizes
	srcElms = 3 * sum([sysInfo.cellList[itr, totCellInd] for itr in bdyPairs[srcBdyInd]])
	trgElms = 3 * sum([sysInfo.cellList[itr, totCellInd] for itr in bdyPairs[trgBdyInd]])
	# Initial channel signal to start SVD solver.
	put!(termnChn, svdFlg + 1)
	
	@sync begin
		@async svdController!(sysIds, coordChn, termnChn, svdFlg, deflateMode, prevMaxItr, 
			sysInfo, sysCharac, sysCharacInd, viSystem, 
			bdyPairs[srcBdyInd], srcElms, 
			bdyPairs[trgBdyInd], trgElms, 
			solverOpts, totlFields)

		@async basisOrthog!(coordChn, termnChn, svdFlg, sysNum, srcElms, trgElms, sysCharac, 
			sysCharacInd, solverOpts, orthBasis, totlFields)
	end
	svdAcc = solverOpts.svdAccuracy
	@printf(stdout, "Target basis estimation complete: %f accuracy %d trials.\n", svdAcc, sysCharac.trialInfo[sysCharacInd])
	return nothing
end
"""

    function viSVD!(sysIds::Array{Int, 1}, viSystem::Array{VIObject, 1}, sysInfo::VISysInfo, bdyPairs::Tuple{Array{Int, 1}, Array{Int, 1}}, deflateMode::Int, sysCharac::VISysCharac, solverOpts::VISolverOpts)::VIOprSVD

Generates Green function approximation from field and current information. 
"""
function viSVD!(sysIds::Array{Int, 1}, viSystem::Array{VIObject, 1}, sysInfo::VISysInfo, 
	bdyPairs::Tuple{Array{Int, 1}, Array{Int, 1}}, deflateMode::Int, sysCharac::VISysCharac,
	solverOpts::VISolverOpts)::VIOprSVD

	# Calculation of source and target sizes
	trgBdyInd = 1
	srcBdyInd = 2
	totCellInd = 4
	srcElms = 3 * sum([sysInfo.cellList[itr, totCellInd] for itr in bdyPairs[srcBdyInd]])
	trgElms = 3 * sum([sysInfo.cellList[itr, totCellInd] for itr in bdyPairs[trgBdyInd]])
	# Initialize memory
	orthBasisTrg = Array{ComplexF64, 2}(undef, trgElms, sysCharac.maxTrials)
	orthBasisSrc = Array{ComplexF64, 2}(undef, srcElms, sysCharac.maxTrials)

	fieldsTrg = Array{ComplexF64, 2}(undef, trgElms, sysCharac.maxTrials)
	fieldsSrc = Array{ComplexF64, 2}(undef, srcElms, sysCharac.maxTrials)

	svdCompArrTrg = Array{ComplexF64, 2}(undef, sysCharac.maxTrials, sysCharac.maxTrials)
	svdCompArrSrc = Array{ComplexF64, 2}(undef, sysCharac.maxTrials, sysCharac.maxTrials)
	# Calculate svd matrix and orthonormal basis for source and target spaces
	svdMemPrep!(sysIds, viSystem, sysInfo, bdyPairs, deflateMode, 1, sysCharac, solverOpts, orthBasisTrg, fieldsTrg)
	# Switch to adjoint system.
	sysAdjVI!(sysInfo.viCuLibHandle, sysIds)
	@printf(stdout, "Adjoint operation completed across active systems.\n")
	# Calculate src basis.
	svdMemPrep!(sysIds, viSystem, sysInfo, bdyPairs, deflateMode, 2, sysCharac, solverOpts, orthBasisSrc, fieldsSrc)
	# Flip system back.
	sysAdjVI!(sysInfo.viCuLibHandle, sysIds)
	@printf(stdout, "Adjoint operation completed across active systems.\n")
	# Limit svd to smallest discovered basis dimension
	svdDim = (sysCharac.trialInfo[1] > sysCharac.trialInfo[2]) ? sysCharac.trialInfo[2] : sysCharac.trialInfo[1]
	@printf(stdout, "\nSVD operator estimate complete: Rank %d.\n", svdDim)
	# Compute intermediate arrays
	BLAS.gemm!('C', 'N', 1.0+0.0im, view(orthBasisTrg, :, 1:svdDim), view(fieldsTrg, :, 1:svdDim), 0.0+0.0im, view(svdCompArrTrg, 1:svdDim, 1:svdDim))
	BLAS.gemm!('C', 'N', 1.0+0.0im, view(orthBasisSrc, :, 1:svdDim), view(sysCharac.initCurrs[1], :, 1:svdDim), 0.0+0.0im, view(svdCompArrSrc, 1:svdDim, 1:svdDim))
	# Invert sources
	# LU decomposition information
	invSrc = LAPACK.getrf!(view(svdCompArrSrc, 1:svdDim, 1:svdDim))

	if(invSrc[3] < 0)
		error("viSVD! Error: Inversion of source basis and vectors has failed.")
	end
	# Perform inversion
	LAPACK.getri!(invSrc[1], invSrc[2])
	# Generate svd matrix
	svdArr = BLAS.gemm('N', 'N', 1.0+0.0im, view(svdCompArrTrg, 1:svdDim, 1:svdDim), invSrc[1])
	# Get svd
	primitiveSVD = svd(svdArr)
	trgBasis = BLAS.gemm('N', 'N', 1.0+0.0im, view(orthBasisTrg, :, 1:svdDim), primitiveSVD.U)
	srcBasis = BLAS.gemm('N', 'C', 1.0+0.0im, primitiveSVD.Vt, view(orthBasisSrc, :, 1:svdDim))
	return VIOprSVD(trgBasis, primitiveSVD.S, srcBasis, bdyPairs)
end

function genHeatKer(viSystem::Array{VIObject, 1}, sysInfo::VISysInfo, sysCharac::VISysCharac, greenOpr::VIOprSVD)::VIHeatKer

	trgBdyId = 1
	srcBdyId = 2
	svdDim = length(greenOpr.singVals)
	srcElms = length(view(greenOpr.srcBasis, 1, :))
	trgElms = length(view(greenOpr.trgBasis, :, 1))
	# Generate heat basis for target and source
	trgHeatBasis = Array{ComplexF64, 2}(undef, trgElms, svdDim)
	srcHeatBasis = Array{ComplexF64, 2}(undef, svdDim, srcElms) 
	
	srcAbsMat = Array{ComplexF64, 2}(undef, svdDim, svdDim)
	trgAbsMat = Array{ComplexF64, 2}(undef, svdDim, svdDim)

	for rank in 1:svdDim
		
		threadCpy!(view(greenOpr.trgBasis, :, rank), view(trgHeatBasis, :, rank))
		threadCpy!(view(greenOpr.srcBasis, rank, :), view(srcHeatBasis, rank, :))
		# Multiply by square root of imaginary permittivity
		bdyMask!(VIStructs.rHeatTrnf::VIOptMode, sysInfo, sysCharac.bodyPairs[trgBdyId], viSystem, view(trgHeatBasis, :, rank))
		bdyMask!(VIStructs.rHeatTrnf::VIOptMode, sysInfo, sysCharac.bodyPairs[srcBdyId], viSystem, view(srcHeatBasis, rank, :))
	end

	srcAbsMat = BLAS.gemm('N', 'C', 1.0+0.0im, srcHeatBasis, srcHeatBasis)
	trgAbsMat = BLAS.gemm('C', 'N', 1.0+0.0im, trgHeatBasis, trgHeatBasis)
	heat = (2 / pi) * tr(greenOpr.singVals .* trgAbsMat .* greenOpr.singVals .* srcAbsMat)
	return VIHeatKer(heat, greenOpr)
end
end