"""
User interface function for interacting with VI Electromagnetic solvers.
"""
module VIUserInterface
using Printf, ParallelUtilities, VIStructs, VICu, VISolverUtilities 
export glbInitVI!, glbFinlVI!, genSysInfoVI, initSysDefaultVI, genSysCharacVI, slvGreenVI!, slvHeatVI!, loadSingleCurrVI!, fieldSingleSolveVI!, characFinlVI!, finlSysDefaultVI!
"""
	
    glbInitVI!(sysNum::Int)::Ref{Ptr{Nothing}}

Initializes libVICu.so and prepares global pointers to hold sysNum VI systems.
"""
function glbInitVI!(sysNum::Int)::Ref{Ptr{Nothing}}

	viCuLibHandle = libInitVICu!()
	numDevs = devCountVI(viCuLibHandle)
	@printf(stdout, "gVICu library initialized, %d active device(s).\n", numDevs)
	devGlbInitVI!(viCuLibHandle, sysNum, numDevs)

	return viCuLibHandle
end
"""
	
    glbFinlVI!(viCuLibHandle::Ref{Ptr{Nothing}})::Nothing

Free global pointers in libVICu.so in preparation for program termination.
"""
function glbFinlVI!(viCuLibHandle::Ref{Ptr{Nothing}}, sysNum::Int)::Nothing

	devGlbFinlVI!(viCuLibHandle, sysNum)

	return nothing
end
"""

   genSysInfoVI(system::Array{VIObject, 1}, preFacMode::Int, svdTol::Float64, solTol::Float64, basisDim::Int, defltDim::Int, devsGVI::Array{Int, 1}, devsDMR::Array{Int, 1})::VISysInfo

VISysInfo constructor. 
# Arguments
system : array of active VI objects.
preFacMode : prefactor mode for inverse solver.
svdTol : relative accuracy of randomized singular value decomposition (accuracy of Green function).
solTol : relative accuracy for a single inverse solve.
basisDim : dimension of the Krylov space used by the iterative inverse solver
defltDim : dimension of the deflation space used by the iterative inverse solver. 
Note that a deflation space is only useful for high accuracy (solTol < 0.0005) solves. 
For larger solution tolerance the inclusion of a deflation space slows the solver.
devsGVI : array of devices for the linear VI operator. A single device is preferable so long as
the system fits in GPU memory. Expected memory size is approximately 5 * sysVecMem * numBlocks 
+ 3 * sysVecMem * numBodies + 4, i.e. for two bodies ~ 30 * sysVecMem.
devsDMR: array of devices for the inverse DMR solver. A single device is preferable so long as the 
Krylov basis fits in GPU memory. Expected memory size is approximately (basisDim + defltDim + 8) *
sysVecMem, i.e. for a basis of 22 ~ 30 * sysVecMem.
"""
function genSysInfoVI(viCuLibHandle::Ref{Ptr{Nothing}}, system::Array{VIObject, 1}, preFacMode::Int, svdTol::Float64, solTol::Float64, basisDim::Int, defltDim::Int, devsGVI::Array{Int, 1}, devsDMR::Array{Int, 1})::VISysInfo
	
	svdExitCount = 4
	numBdys = length(system)
	cellScale =  system[1].scale
	cellList = Array{Int, 2}(undef, numBdys, 3)

	for bdyItr in 1:numBdys
		@. cellList[bdyItr, :] = system[bdyItr].cells
	end
	
	bdyCellList = [prod(cellList[bdyItr, :]) for bdyItr in 1:numBdys]

	return VISysInfo(numBdys, sum(bdyCellList), 
		hcat(cellList, bdyCellList, 
			[sum(bdyCellList[1:(bdyItr - 1)]) for bdyItr in 1:numBdys]),
		viCuLibHandle, VICompOpts(devsGVI, devsDMR), 
		VISolverOpts(preFacMode, basisDim, defltDim, svdTol, svdExitCount, solTol),
		VIAssemblyOpts(cellScale))
end
"""

    initSysDefaultVI(viCuLibHandle::Ref{Ptr{Nothing}}, sysIds::Array{Int,1}, 
    system::Array{VIObject, 1}, preFacMode::Int, svdTol::Array{Float64, 1}, 
    solTol::Array{Float64, 1}, basisDim::Array{Int, 1}, defltDim::Array{Int, 1},
    devsGVI::Array{Int}, devsDMR::Array{Int})::Array{VISysInfo, 1}

Initialize the a VI system with default settings. For information consult VIStructs.
"""
function initSysDefaultVI(viCuLibHandle::Ref{Ptr{Nothing}}, sysIds::Array{Int,1}, 
    system::Array{VIObject, 1}, preFacMode::Int, svdTol::Array{Float64, 1}, 
    solTol::Array{Float64, 1}, basisDim::Array{Int, 1}, defltDim::Array{Int, 1},
    devsGVI::Array{Int}, devsDMR::Array{Int})::Array{VISysInfo, 1}
	
	numSys = length(sysIds)
	totInfo = Array{VISysInfo, 1}(undef, numSys)

	
	for itr in 1:numSys
		totInfo[itr] = genSysInfoVI(viCuLibHandle, system, preFacMode, 
		svdTol[itr], solTol[itr], basisDim[itr], defltDim[itr], devsGVI[:, itr], devsDMR[:, itr])
	end

	if(devSysInitVI!(sysIds, system, totInfo) != 0)
		error("Device initialization failure!")
		
		for itr in lenght(sysIds)
			totInfo[itr].cellList[0,0] = -1
		end
	end

	return totInfo
end
"""

    genSysCharacVI(maxTrials::Int, bodyPairs::Tuple{Array{Int, 1}, Array{Int, 1}}, sysInfo::VISysInfo)::VISysCharac

Generate characterizing information for singular value decomposition of a VI system using default
parameters. 
# Arguments
.maxTrials : upper limit for number of iteration that can be performed to find the SVD.
.bodyPairs : target bodies, followed by source bodies, determining the Green function that will be 
solved for.
.srcInfo : structure containing VI system options, see VIStructs for information.
"""
function genSysCharacVI(maxTrials::Int, bodyPairs::Tuple{Array{Int, 1}, Array{Int, 1}}, sysInfo::VISysInfo)::VISysCharac
	
	totCellInd = 4 
	totElms = 3 * sysInfo.totalCells
	trgElms = 3 * sum([sysInfo.cellList[itr, totCellInd] for itr in bodyPairs[1]])
	srcElms = 3 * sum([sysInfo.cellList[itr, totCellInd] for itr in bodyPairs[2]])

	return VISysCharac(srcElms, trgElms, totElms, bodyPairs, maxTrials)
end

function genSysCharacVI(bodyPairs::Tuple{Array{Int, 1}, Array{Int, 1}}, sysInfo::VISysInfo)::VISysCharac
	
	sysFreeMem = Sys.free_memory()
	vecSizeMem = 3 * sysInfo.totalCells * sizeof(ComplexF64)
	# Choice of 16 should equate to generated sysCharac occupying roughly 
	# a third of the system memory
	maxTrials = round(Int, sysFreeMem / (16 * vecSizeMem))
	maxTrials = max(trials, 512)
	totCellInd = 4 
	srcElms = 3 * sum([sysInfo.cellList[itr, totCellInd] for itr in bodyPairs[2]])
	totElms = 3 * sysInfo.totalCells
	# Value of five comes from randomized svd argument
	maxTrials = min(maxTrials, totElms + 5)
	
	return VISysCharac(srcElms, totElms, bodyPairs, maxTrials)
end
"""

    genSlvCharacVI(maxSlvs::Int, srcBdys::Array{Int, 1}, sysInfo::VISysInfo)::VISlvCharac

Generate characterizing information for single solves of a VI system using default parameters. 
# Arguments
.maxSlvs : number of different input currents that will be solved with current settings,
determines amount of memory that will be allocated.
.srcInfo : structure containing VI system options, see VIStructs for information.
"""
function genSlvCharacVI(maxSlvs::Int, srcBdys::Array{Int, 1}, sysInfo::VISysInfo)::VISlvCharac

	totCellInd = 4
	totElms = 3 * sysInfo.totalCells
	srcElms = 3 * sum([sysInfo.cellList[itr, totCellInd] for itr in srcBdys])
	return VISlvCharac(maxSlvs, srcBdys, srcElms, totElms)
end
"""

    slvGreenVI!(sysIds::Array{Int, 1}, viSystem::Array{VIObject, 1}, sysInfo::VISysInfo, 
    deflateMode::Int, sysCharac::VISysCharac)::VIOptSVD

Determine a low rank randomized SVD of the electromagnetic Green function between the bodies
specified by sysCharac.bodyPairs for the VI system described by sysInfo and viSystem.
# Arguments
sysIds : Array of the system identifiers associated with the system (parallelization) 
viSystem : VIObject contained in the system, see VIStructs for information
sysInfo : options for the VI solver, see VIStructs for information 
deflateMode : != 0 indicates that deflation space generated in the latest solve call to the VICu
library, on the handle held in sysInfo, should be used to facilitate the present calculation.
sysCharac : Solver memory storage, see VIStructs for information.
"""
function slvGreenVI!(sysIds::Array{Int, 1}, viSystem::Array{VIObject, 1}, sysInfo::VISysInfo, 
	deflateMode::Int, sysCharac::VISysCharac)::VIOprSVD
	
	return viSVD!(sysIds, viSystem, sysInfo, sysCharac.bodyPairs, deflateMode, 
		sysCharac, sysInfo.solverInfo)
end
"""

    slvHeatVI!(sysIds::Array{Int, 1}, viSystem::Array{VIObject, 1}, sysInfo::VISysInfo, 
    deflateMode::Int, sysCharac::VISysCharac)::VIOptSVD

Determine a low rank randomized SVD of the electromagnetic radiative heat transfer
Green function between the bodies specified by sysCharac.bodyPairs for the VI system described 
by sysInfo and viSystem.
# Arguments
sysIds : Array of the system identifiers associated with the system (parallelization) 
viSystem : VIObject contained in the system, see VIStructs for information
sysInfo : options for the VI solver, see VIStructs for information 
deflateMode : != 0 indicates that deflation space generated in the latest solve call to the VICu
library, on the handle held in sysInfo, should be used to facilitate the present calculation.
sysCharac : Solver memory storage, see VIStructs for information.
"""
function slvHeatVI!(sysIds::Array{Int, 1}, viSystem::Array{VIObject, 1}, sysInfo::VISysInfo, deflateMode::Int, sysCharac::VISysCharac)::VIHeatKer
	
	greenOpr = viSVD!(sysIds, viSystem, sysInfo, sysCharac.bodyPairs, deflateMode, sysCharac,
	sysInfo.solverInfo)

	return genHeatKer(viSystem, sysInfo, sysCharac, greenOpr)
end
"""

    fieldSingleSolveVI!(sysItr::Int, viSystem::Array{VIObject, 1}, sysInfo::VISysInfo, slvItr::Int,
    deflateMode::Int, slvCharac::VISlvCharac, eField::Array{ComplexF64, 2})::Tuple{Float64, Int}
"""
function fieldSingleSolveVI!(sysIds::AbstractArray{Int,1}, viSystem::Array{VIObject, 1}, sysInfo::VISysInfo, deflateMode::AbstractArray{Int,1}, initCurr::AbstractArray{ComplexF64, 2}, eField::Array{ComplexF64, 2})::Tuple{Array{Float64, 1}, Array{Int32, 1}}

	sysSize = length(sysIds)
	solTol = Array{Float64}(undef, sysSize)

	for sysInd in 1:sysSize
		solTol[sysInd] = sysInfo.relativeSolutionTolearance 
	end
	solveInfo = solveVI!(sysInfo.CuLibHandle, sysIds, deflateMode, solTol, eField, initCurr)
	fieldConvertVI!(sysInfo.viCuLibHandle, sysIds, eField)
	return solveInfo
end
"""

    characFinlVI!(sysCharac::VISysCharac || slvCharac::VISlvCharac)::Nothing

Removes reference to memory pointed to by a VI characteristic structure, 
allowing it to be garbage collected, and then calls the garbage collector.
"""
function characFinlVI!(sysCharac::VISysCharac)::Nothing

	sysCharac = nothing
	GC.gc()
	return nothing
end
"""

    finlSysDefaultVI!(sysItr::Int, sysInfo::VISysInfo)::Nothing

Finalize the VI system, freeing associated device memory.
"""
function finlSysDefaultVI!(sysIds::Array{Int, 1}, totInfo::Array{VISysInfo, 1})::Nothing

	cnt = 1
	
	for sysItr in sysIds
		devSysFinlVI!(totInfo[cnt].viCuLibHandle, sysItr, totInfo[cnt])
		cnt += 1
	end

	cnt = 1
	for sysItr in sysIds
		devSysMemFinlVI!(totInfo[cnt].viCuLibHandle, sysItr)
		cnt += 1
	end
	totInfo = nothing
	GC.gc()
	return nothing
end
end