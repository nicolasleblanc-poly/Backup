using Distributed
# , MaxGParallelUtilities
@everywhere using Printf, Base.Threads, LinearAlgebra#, MaxGParallelUtilities, MaxGStructs, MaxGCirc

# Here's one of my attempts to import the modules another way than that done with "using"
# include("MaxGParallelUtilities.jl")
# include("MaxGStructs.jl")
# include("MaxGCirc.jl")

using Random #, MaxGParallelUtilities
#export MaxGDom, MaxGVol, genMaxGVol, MaxGAssemblyOpts
"""
Basic spatial domain object.
# Arguments 
.cells : tuple of cells for defining a rectangular prism.
.scale : relative side length of a cubic voxel compared to the wavelength.
.coord : center position of the object. 
"""
struct MaxGDom
	cells::Array{Int, 1}
	scale::Float64
	coord::NTuple{3, Float64}
	# boundary conditions here?
	# Can I do coordinate transformations to generalize the code to any  
	# translationally invariant right-handed coordinate system?
end

"""

    genMaxGVol(domDes::MaxGDom)::MaxGVol

Construct a MaxGVol based on the domain description given by a MaxGDom.
"""
function genMaxGVol(domDes::MaxGDom)::MaxGVol

	bounds = @. domDes.scale * (domDes.cells - 1) / 2.0 
	grid = [(round(-bounds[1] + domDes.coord[1], digits = 6) : domDes.scale : 
	round(bounds[1] + domDes.coord[1], digits = 6)), 
	(round(-bounds[2] + domDes.coord[2], digits = 6) : domDes.scale : 
	round(bounds[2] + domDes.coord[2], digits = 6)), 
	(round(-bounds[3] + domDes.coord[3], digits = 6) : domDes.scale : 
	round(bounds[3] + domDes.coord[3], digits = 6))]
	
	return MaxGVol(domDes.cells, prod(domDes.cells), domDes.scale, 
		domDes.coord, grid)
end	

"""
Characterization information for a domain in MaxG.
# Arguments 
.cells : number of cells in a Cartesian direction, specifying the degrees of 
freedom of the rectangular prism volume.
.totalCells : total number of cells contained in the volume.
.coord : spatial position of the center of the prism. 
.grid : spatial location of the center of each cell contained in the volume. 
"""
struct MaxGVol

	cells::Array{Int, 1}
	totalCells::Int
	scale::Float64
	coord::NTuple{3, Float64}
	grid::Array{<:StepRangeLen, 1}
end

"""
Necessary information for Green function construction.  
# Arguments
.freqPhase : multiplicative scaling factor allowing for complex frequencies. 
.ordGLIntFar : Gauss-Legendre order for far distance cells. 
.ordGLIntMed : Gauss-Legendre order for medium distance cells. 
.ordGLIntFar : Gauss-Legendre order for near distance cells.
.crossNearMed : sets crossover from near to medium regime in terms of cells.
.crossNearMed : sets crossover from medium to far regime in terms of cells.
"""
struct MaxGAssemblyOpts

	freqPhase::ComplexF64
	ordGLIntFar::Int
	ordGLIntMed::Int
	ordGLIntNear::Int
	crossMedFar::Int
	crossNearMed::Int
end
"""
Simplified MaxGAssemblyOpts constructor.
"""
function MaxGAssemblyOpts(scale::Float64)

	return MaxGAssemblyOpts(1.0 + im * 0.0,
		min(max(floor(Int, 22 + 3 * log10(scale)), 1), 1), 
		min(max(floor(Int, 22 + 3 * log10(scale)), 1), 4),
		min(max(floor(Int, 22 + 3 * log10(scale)), 2), 16),
		min(max(round(Int, 2 * ^(2, -log10(scale) * 5 / 2 - 3 / 2)), 4), 16),
		min(max(round(Int, 1 * ^(2, -log10(scale) * 5 / 2 - 3 / 2)), 2), 8))
end

"""
An even simpler MaxGAssemblyOpts constructor.
"""
function MaxGAssemblyOpts()

	return MaxGAssemblyOpts(1.0 + im * 0.0, 16, 16, 16, 16, 8)
end









# Set number of BLAS threads. The number of Julia threads is set as an 
# environment variable. The total number of threads is Julia threads + BLAS 
# threads. VICu is does not call BLAS libraries during threaded operations, 
# so both thread counts can be set near the available number of cores. 
threads = nthreads()
BLAS.set_num_threads(threads)
blasThreads = BLAS.get_num_threads()
println("MaxG test initialized with ", nthreads(), 
	" Julia threads and $blasThreads BLAS threads.")
# Define test volume, all lengths are defined relative to the wavelength. 
# Number of cells in the volume. 
cellsA = [4, 4, 4] # x, y, z
cellsB = [4, 4, 4] 

# Does x=4 mean ji_x should be a 4x1 vector? It sure doesn't seem/look like it...

# Size length of a cell relative to the wavelength. 
scale = 0.01
# Center position of the volume. 
coordA = (0.0, 0.0, 0.0)
coordB = (0.0, 0.0, 1.0)

# Create MaxG volumes.
volA = genMaxGVol(MaxGDom(cellsA, scale, coordA))
volB = genMaxGVol(MaxGDom(cellsB, scale, coordB))
# Information for Green function construction. 
# Complex frequency ratio. 
freqPhase = 1.0 + im * 0.0
# Gauss-Legendre approximation orders. 
ordGLIntFar = 2
ordGLIntMed = 4
ordGLIntNear = 16
# Cross over points for Gauss-Legendre approximation.
crossMedFar = 16
crossNearMed = 8

assemblyInfo = MaxGAssemblyOpts(freqPhase, ordGLIntFar, ordGLIntMed, 
	ordGLIntNear, crossMedFar, crossNearMed)
# Pre-allocate memory for circulant green function vector. 
greenCircAA = Array{ComplexF64}(undef, 3, 3, 2 * cellsA[1], 2 * cellsA[1], 
	2 * cellsA[1])
greenCircBA = Array{ComplexF64}(undef, 3, 3, cellsB[1] + cellsA[1], cellsB[2] +
	cellsA[2], cellsB[3] + cellsA[3])
# So it's here we would differentiate between getting the Gxx or Gxy by changing
# the second and third entries to the greenCircAA and greenCircBA variables?

"""
Valid MaxG operation settings
"""
@enum MaxGOptMode begin

	singleSlv
	greenFunc
end

"""
Settings for GPU computation.
"""
struct MaxGCompOpts

	gBlocks::Int
	gThreads::Int
	deviceListMaxG::Array{Int, 1}
	deviceListDMR::Array{Int, 1}
end
"""
Simplified MaxGCompOpts constructors.
"""
function MaxGCompOpts(deviceListMaxG::Array{Int, 1}, deviceListMDR::Array{Int, 1})

	return MaxGCompOpts(32, 512, deviceListMaxG, deviceListMDR)
end

function MaxGCompOpts()

	return MaxGCompOpts(32, 512, [0], [0])
end
"""
Options for iterative solver. Both basisDimension + 1 and deflateDimension must be divisible by
the by number of MDR devices in use. For example generation see MaxGUserInterface genSolverOptsMaxG.
# Arguments
prefacMode : == 0 solves MaxG as χ + ϵ*G, all other settings Id + χ^{-1}*ϵ*G.	
basisDimension : dimension of Arnoldi basis for iterative inverse solver.
deflateDimension : dimension of deflation space to use in DMRs iterative inverse solver.
svdAccuracy : relative post Gram-Schmidt magnitude for a vector to be considered captured by the
existing basis.
svdExitCount : randomized svd setting, after n success, as defined by svdAccuracy, there is a
1 - 1/(n^n) probability that the randomized svd is accurate to within svdAccuracy. Typically a
number between 3 and 6 should be selected.				
"""
struct MaxGSolverOpts

	prefacMode::Int
	basisDimension::Int
	deflateDimension::Int
	svdAccuracy::Float64
	svdExitCount::Int
	relativeSolutionTolearance::Float64
end
"""
Storage structure for MaxG system parameters.
# Arguments
cellList : a negative value in cellList[0,0] occurs if device initialization does not succeed.
This can be used as a termination flag.
"""
struct MaxGSysInfo

	bodies::Int
	# Total number of cells in the system.
	totalCells::Int
	# Four entries per cell: number of {x,y,z} cells, product of cells, linear cell 
	# starting position.
	cellList::Array{Int, 2}
	# Handle to GPU solver library
	viCuLibHandle::Ref{Ptr{Nothing}}
	# Computation settings
	computeInfo::MaxGCompOpts
	solverInfo::MaxGSolverOpts
	assemblyInfo::MaxGAssemblyOpts
end
"""
Characterizing information for singular value decomposition of a MaxG system. 
# Arguments
.maxTrials : upper limit for number of iteration that can be performed to find the SVD.
.trialInfo : three element array holding the generation mode, followed by the number of trials
performed in determining the target and source bases. A .trialInfo[1] != 0 indicates that 
information from the last solve, stored in the structure, should be used to facilitate the 
current solve.
.bodyPairs : target bodies, followed by source bodies, determining the Green function that will be 
solved for.
.initCurrs : current vectors used in previous SVD solve.
.totlCurrs : solutions found during previous SVD solve, successful trial number followed by total.
"""
struct MaxGSysCharac

	maxTrials::Int
	trialInfo::Array{Int, 1}
	bodyPairs::Tuple{Array{Int, 1}, Array{Int, 1}}
	initCurrs::Tuple{Array{ComplexF64, 2}, Array{ComplexF64, 2}}
	totlCurrs::Tuple{Array{ComplexF64, 2}, Array{ComplexF64, 2}}
end
# Simplified constructor
function MaxGSysCharac(srcElms::Int, trgElms::Int, totElms::Int, bodyPairs::Tuple{Array{Int, 1}, Array{Int, 1}}, maxTrials::Int)

	# Random generator.
	randGen = MersenneTwister(12345)

	return MaxGSysCharac(maxTrials, [0, 0], bodyPairs,
	(randn!(randGen, Array{ComplexF64, 2}(undef, srcElms, maxTrials)), 
		randn!(randGen, Array{ComplexF64, 2}(undef, srcElms, maxTrials))),
	(Array{ComplexF64, 2}(undef, totElms, maxTrials), 
		Array{ComplexF64, 2}(undef, totElms, maxTrials)))
end
"""
Characterizing information for single solves of a MaxG system. 
# Arguments
.maxSlvs : number of different input currents that will be solved with current settings,
determines amount of memory that will be allocated.
.srcBodys : list of bodies where non-zero source currents will be placed.
.initCurrs : storage for initial (source) currents.
.totlCurrs : storage for total (solution) currents.
"""
struct MaxGSlvCharac

	maxSlvs::Int
	srcBdys::Array{Int, 1}
	initCurrs::Array{ComplexF64, 2}
	totlCurrs::Array{ComplexF64, 2}
end
"""
Simplified constructor.
"""
function MaxGSlvCharac(maxSlvs::Int, srcBdys::Array{Int, 1}, srcElms::Int, totlElms::Int)::MaxGSlvCharac

	return MaxGSlvCharac(maxSlvs, srcBdys, Array{ComplexF64, 2}(undef, srcElms, maxSlvs), Array{ComplexF64, 2}(undef, totlElms, maxSlvs))
end
"""
Structure for singular value decompositions computed by the MaxG solver. srcBasis is a stored as 
a dual.
"""
struct MaxGOprSVD
	
	trgBasis::Array{ComplexF64, 2}
	singVals::Array{ComplexF64, 1}
	srcBasis::Array{ComplexF64, 2}
	bdyPairs::Tuple{Array{Int, 1}, Array{Int, 1}}
end
"""
Structure for singular value decompositions of heat kernel computed by the MaxG solver. srcHeatBasis
is stored as a dual.
"""
struct MaxGHeatKer

	heat::Float64
	greenOpr::MaxGOprSVD
end

using Distributed, Base.Threads, LinearAlgebra
#export ParallelConfig, nodeArrayW!, threadArrayW!, threadPtrW!, threadPtrR!, 
#threadCpy!, genWorkBounds
"""

    genWorkBounds(procBounds::Tuple{Int,Int}, 
    numGroups::Int)::Array{Tuple{Int, Int},1}

Partitions integers from procBounds[1] to procBounds[2] into work-sets. 
"""
function genWorkBounds(procBounds::Tuple{Int,Int}, 
	numGroups::Int)::Array{Tuple{Int,Int},1}
	
	workBounds = Array{Tuple{Int,Int},1}(undef, numGroups)
	splits = [ceil(Int, (procBounds[2] - procBounds[1] + 1) * s / numGroups) 
	for s in 1:numGroups] .+ (procBounds[1] - 1)

	for grp in 1 : numGroups
		
		if (grp == 1)
			
			workBounds[grp] = (procBounds[1], splits[1])

		else
			
			if ((splits[grp] - splits[grp - 1]) == 0)
			
				workBounds[grp] = (0, 0)
			else
			
				workBounds[grp] = (splits[grp - 1] + 1, splits[grp])
			end
		end
	end

	return workBounds
end
"""
	cInds(sA::AbstractArray{T} where {T <: Number})::UnitRange{Int}

Returns a C style linear range for shared array sA.
"""
function cInds(sA::AbstractArray{T} where {T <: Number})::UnitRange{Int}

	procID = indexpids(sA)
	numSplits = length(procs(sA)) + 1
	#Unassigned workers get a zero range
	if procID == 0 
	
		return 1 : 0
	end
	
	splits = [round(Int, s) for s in range(0, stop = prod(size(sA)), 
		length = numSplits)]
	
	return ((splits[procID] + 1) : splits[procID + 1])
end
"""

    function tCInds(ind::Int, cells::Array{Int,1})::Array{Int,1}

Convert a linear index into a Cartesian index. The first number is treated as 
the outermost index, the next three indices follow column major order. 
"""
@inline function tCInds(ind::Int, cells::Array{Int,1})::Array{Int,1}
 	
	return [1 + div(ind - 1, cells[1] * cells[2] * cells[3]), 
	1 + (ind - 1) % (cells[1] * cells[2] * cells[3]) % (cells[1]), 
	1 + div((ind - 1) % (cells[1] * cells[2] * cells[3]) % (cells[1] * 
		cells[2]), cells[1]), 
	1 + div((ind - 1) % (cells[1] * cells[2] * cells[3]), cells[1] * cells[2])]
end
"""

    function tJInds(ind::Int, cells::Array{Int,1})::Array{Int,1}

Convert a linear index into a tensor index following column major order. 
"""
@inline function tJInds(ind::Int, cells::Array{Int,1})::Array{Int,1}
 	
	return [1 + (ind - 1) % (cells[1] * cells[2] * cells[3]) % (cells[1]), 
	1 + div((ind - 1) % (cells[1] * cells[2] * cells[3]) % (cells[1] * 
		cells[2]), cells[1]), 
	1 + div((ind - 1) % (cells[1] * cells[2] * cells[3]), cells[1] * cells[2]),
	1 + div(ind - 1, cells[1] * cells[2] * cells[3])]
end
"""
	coreInds(sA::AbstractArray{T} where {T <: Number}, uBound::NTuple{N,Int} 
	where {N})::Array{UnitRange{Int},1}

Returns range of indices for a particular worker by splitting sA along its last 
index. 
""" 
function coreInds(sA::AbstractArray{T} where {T <: Number}, 
	uBound::NTuple{N,Int} where {N})::Array{UnitRange{Int},1}

	procID = indexpids(sA)
	# Reduce array split if array is small
	if length(procs(sA)) > uBound[end] 
		
		numSplits = uBound[end] + 1

	else
		
		numSplits = length(procs(sA)) + 1

	end
	# Reduce array split.
	if procID > uBound[end] 
		
		return repeat([1 : 0], outer = ndims(sA))

	end
	# Head worker is not assigned a range
	if procID == 0 
		
		return repeat([1 : 0], outer = ndims(sA))

	end
	
	splits = [round(Int, s) for s in range(0, stop = uBound[end], 
		length = numSplits)]
	
	indRangeArray = Array{UnitRange{Int},1}(undef, ndims(sA))
	
	for i = 1 : ndims(sA)
		
		if i == ndims(sA)
			
			indRangeArray[i] = (splits[procID] + 1) : splits[procID + 1]

		else
			
			indRangeArray[i] = 1 : uBound[i]  
		end
	end
	
	return indRangeArray
end
"""
	function coreLoopW!(sA::AbstractArray{T} where {T <: Number}, 
	indVec::Array{Int,1}, indRangeWrite::Array{UnitRange{Int},1}, func)::Nothing

Write func values to a shared array by looping over an array of ranges in column 
major order---generating nested for loops.

The function func s presumed to have an argument tuple of indices consistent 
with indRangeArray.
"""
@inline function coreLoopW!(sA::AbstractArray{T} where {T <: Number}, 
	indVec::Array{Int,1}, indRangeWrite::Array{UnitRange{Int},1}, func)::Nothing

	if length(indRangeWrite) == 1
		
		@inbounds for i in indRangeWrite[1] 
			
			indVec[1] = i
			sA[indVec...] = func(indVec...)
		end

	else
		
		@inbounds for i in indRangeWrite[end]

			indVec[length(indRangeWrite)] = i
			coreLoopW!(sA, indVec, indRangeWrite[1 : (end - 1)], func)
		end
	end
	
	return nothing
end
"""

    ptrW!(sA::AbstractArray{T} where {T <: Number}, ptr::Ptr{T} 
    where {T <: Number}, indRange::UnitRange{Int})::Nothing

Write the contents of a shared array to a ptr for memory not managed by Julia.
"""
@inline function ptrW!(sA::AbstractArray{T} where {T <: Number}, 
	ptr::Ptr{T} where {T <: Number}, indRange::UnitRange{Int})::Nothing

	@inbounds for ind in indRange
	
		unsafe_store!(ptr, convert(eltype(ptr), sA[ind]), ind)
	end
	
	return nothing
end
"""

    ptrR!(ptr::Ptr{T} where {T <: Number}, sA::AbstractArray{T} 
    where {T <: Number}, indRange::UnitRange{Int})::Nothing

Read the contents of a ptr, for memory not managed by Julia, into a 
shared array.
"""
@inline function ptrR!(ptr::Ptr{T} where {T <: Number}, 
	sA::AbstractArray{T} where {T <: Number}, indRange::UnitRange{Int})::Nothing

	@inbounds for ind in indRange
		sA[ind] = convert(eltype(sA), unsafe_load(ptr, ind))
	end
	
	return nothing
end
"""
	coreLoopVW!(sA::AbstractArray{T} where {T <: Number}, indVec::Array{Int,1}, 
	writeSubA::Array{UnitRange{Int},1}, indRangeWrite::Array{UnitRange{Int},1},
	 func!)::Nothing

Version of coreLoopW! using views to avoid internally generating write matrices.
"""
@inline function coreLoopVW!(sA::AbstractArray{T} where {T <: Number}, 
	indVec::Array{Int,1}, writeSubA::Array{UnitRange{Int},1}, 
	indRangeWrite::Array{UnitRange{Int},1}, func!)::Nothing

	if length(indRangeWrite) == 1
		
		@inbounds for i in indRangeWrite[1] 
		
			indVec[1] = i
			func!(view(sA, writeSubA..., indVec...), indVec...)
		end
	else
		
		@inbounds for i in indRangeWrite[end]
			
			indVec[length(indRangeWrite)] = i
			coreLoopVW!(sA, indVec, writeSubA, indRangeWrite[1 : (end - 1)], 
				func!)
		end
	end

	return nothing
end
"""
	coreArrayW!(sA::AbstractArray{T} where {T <: Number}, indSplit::Int, 
	uBound::NTuple{N,Int} where {N}, func)::Nothing

Writes values to a shared array by looping over an array of ranges in 
column major using coreLoopW! with the function func.

indSplit indicates the first index set to begin looping over, assuming that func 
returns a subArray of filling all preceding indexes. func is presumed to take a
tuple of indices consistent with indRangeArray. 
"""
function coreArrayW!(sA::AbstractArray{T} where {T <: Number}, indSplit::Int, 
	uBound::NTuple{N,Int} where {N}, func)::Nothing

	indRangeArray = coreInds(sA, uBound)
	writeSubA = indRangeArray[1 : (indSplit - 1)]
	indRangeWrite = indRangeArray[indSplit : end]
	indVec = ones(Int, length(indRangeWrite))
	
	if indSplit > 1
		
		coreLoopVW!(sA, indVec, writeSubA, indRangeWrite, func)

	else
		
		coreLoopW!(sA, indVec, indRangeWrite, func)

	end

	return nothing
end
"""

   corePtrW!(sA::AbstractArray{T} where {T <: Number}, ptr::Ptr{T} 
   where {T <: Number})::Nothing

Write values stored in an array to the pointer location for memory not managed 
by Julia. 
"""
function corePtrW!(sA::AbstractArray{T} where {T <: Number}, 
	ptr::Ptr{T} where {T <: Number})::Nothing

	indRange = cInds(sA)
	ptrW!(sA, ptr, indRange)
	
	return nothing
end
"""

   corePtrR!(ptr::Ptr{T} where {T <: Number}, 
   sA::AbstractArray{T} where {T <: Number})::Nothing

Write values stored at a pointer location, for memory not managed by Julia, 
into to a shared array. 
"""
function corePtrR!(ptr::Ptr{T} where {T <: Number}, 
	sA::AbstractArray{T} where {T <: Number})::Nothing

	indRange = cInds(sA)
	ptrR!(ptr, sA, indRange)
	
	return nothing
end
"""
	nodeArrayW!(sA::AbstractArray{T} where {T <: Number}, indSplit::Int, 
	uBound::NTuple{N,Int} where {N}, func)::Nothing

Write values to a shared array over a node, using coreArrayW! The function is 
presumed to take and argument of an array of indices consistent with 
indRangeArray. For more details see this function.
"""
function nodeArrayW!(sA::AbstractArray{T} where {T <: Number}, indSplit::Int, 
	uBound::NTuple{N,Int} where {N}, func)::Nothing

	@sync begin
		
		for p in procs(sA)

			@async remotecall_wait(coreArrayW!, p, sA, indSplit, uBound, func)

		end
	end

	return nothing
end
"""

   nodePtrW!(sA::AbstractArray{T} where {T <: Number}, 
   ptr::Ptr{T} where {T <: Number})::Nothing

Asynchronously writes the contents of sA to the location specified by a pointer. 
The memory specified by the pointer location is not managed by julia. 
"""
function nodePtrW!(sA::AbstractArray{T} where {T <: Number}, 
	ptr::Ptr{T} where {T <: Number})::Nothing

	@sync begin
		
		for p in procs(sA)
	
			@async remotecall_wait(corePtrW!, p, sA, ptr)
	
		end
	end

	return nothing
end
"""

   nodePtrR!(ptr::Ptr{T} where {T <: Number}, 
   sA::AbstractArray{T} where {T <: Number})::Nothing

Asynchronously reads memory from location specified by the pointer into sA. 
The memory specified by the pointer location is not managed by julia. 
"""
function nodePtrR!(ptr::Ptr{T} where {T <: Number}, 
	sA::AbstractArray{T} where {T <: Number})::Nothing

	@sync begin
		
		for p in procs(sA)
	
			@async remotecall_wait(corePtrR!, p, ptr, sA)

		end
	end

	return nothing
end
"""

    threadArrayW!(sA::AbstractArray{T} where {T <: Number}, 
    indSplit::Int, uBound::NTuple{N,Int} where {N}, func)::Nothing

Write values to a array over a node, using threadLoopW! The function, func, is
presumed to tale an argument of an array of indices consistent with 
indRangeArray. For more details see this function. 
"""
function threadArrayW!(sA::AbstractArray{T} where {T <: Number}, indSplit::Int, 
	uBound::NTuple{N,Int} where {N}, func)::Nothing
	# Figuring out which dimension the array should be split in. 
	if indSplit > 1
		
		writeSubA = Array{UnitRange{Int}}(undef, indSplit - 1)
		
		for i in 1 : length(writeSubA)
		
			writeSubA[i] = 1 : uBound[i]

		end
	end	
	# Augment the split dimension, ``moving in'', if the number of index 
	# sub-blocks is smaller than the number of threads. 
	splitInd = 1
	threads = nthreads()
	threadPaths = uBound[end]
	arrayDims = length(uBound)
	
	while (threadPaths < threads) && (splitInd < arrayDims)
		
		threadPaths = threadPaths * uBound[end - splitInd]
		splitInd += 1

	end

	@threads for t in 1 : threads
		
		outerInd = Array{UnitRange{Int}}(undef, splitInd)
		ind = 1
		oi = 1
		# Write ranges for active thread. 
		indRangeWrite = Array{UnitRange{Int}}(undef, 
			length(uBound) - indSplit + 1)
		indVec = ones(Int, length(indRangeWrite))
		# Thread paths that will be handled by active thread. 
		for tInd in ((div(threadPaths * (t - 1), threads) + 1) : 
			div(threadPaths * t, threads)) 
			
			oi = 1
			tInd -= 1
			# Assign outer indices associated with thread path. 
			while oi < splitInd
				
				ind = div(mod(tInd, prod(uBound[((end - (splitInd - 1)) : 
					(end - (oi - 1)))])), prod(uBound[((end - (splitInd - 1)) : 
				(end - oi))])) + 1
				outerInd[end - (oi - 1)] = ind : ind
				oi += 1
			end

			ind = mod(tInd,uBound[end - (splitInd - 1)]) + 1
			outerInd[1] = ind:ind
			# Assign inner write operation range, common among all thread paths.
			for i in 1 : length(indRangeWrite)
				
				if i < (length(indRangeWrite) - (splitInd - 1))
					
					indRangeWrite[i] = 1 : uBound[indSplit + (i - 1)]

				else
					
					indRangeWrite[i] = 
					outerInd[i - (length(indRangeWrite) - splitInd)]
				end
			end 
			# Perform write operations.
			if indSplit > 1

				coreLoopVW!(sA, indVec, writeSubA, indRangeWrite, func)

			else

				coreLoopW!(sA, indVec, indRangeWrite, func)

			end
		end
	end

	return nothing 
end
"""

    threadPtrW!(indSplit::Int, sA::AbstractArray{T} where {T <: Number}, 
    ptr::Ptr{T} where {T <: Number})::Nothing

Copies information from sA into the vector specified by the pointer location. 
The index specified by indSplit is treated as the inner most index, wrapping 
from left to right. ie. if indSplit = 2 then [w,x,y,z] is iterated as [x,y,z,w] 
in column major order.
"""
function threadPtrW!(indSplit::Int, sA::AbstractArray{T} where {T <: Number}, 
	ptr::Ptr{T} where {T <: Number})::Nothing

	threads = nthreads()
	dimCells = ndims(sA) - (indSplit - 1)
	numCells = prod(size(sA)[indSplit : end])
	fillInds = Array{UnitRange{Int}}(undef, dimCells)
	# Write dimensions treated by each path. 
	for i in 1 : dimCells

		fillInds[i] = 1 : (size(sA)[indSplit + (i - 1)])
	end
	
	for subView in CartesianIndices(size(sA)[1 : (indSplit - 1)])
		
		offset = 0 
		vA = view(sA, subView, fillInds...)
		
		for lV in 1:length(subView)
			
			if lV == 1
				
				offset = subView[1] - 1
				continue

			else
	
				for j in 2 : lV
				
					offset = offset + (subView[j] - 1) * 
					prod(size(sA)[1 : (j - 1)])
				end
			end
		end

		offset = offset * numCells
		
		@threads for t in 1 : threads
			
			@inbounds for tInd in ((div(numCells * (t - 1), threads) + 1) : 
			div(numCells * t,threads)) 
				
				unsafe_store!(ptr, convert(ComplexF64,vA[tInd]), tInd + offset)
			end
		end
	end

	return nothing
end
"""

    threadPtrR!(indSplit::Int, ptr::Ptr{T} where {T <: Number}, 
    sA::AbstractArray{T} where {T <: Number})::Nothing

Copies information from the vector specified by the pointer location ptr into 
the abstract array sA. The index specified by indSplit is treated as the inner 
most index, wrapping from left to right. ie. if indSplit = 2 then [w,x,y,z] is 
iterated as [x,y,z,w] in column major order.
"""
function threadPtrR!(indSplit::Int, ptr::Ptr{T} where {T <: Number}, 
	sA::AbstractArray{T} where {T <: Number})::Nothing

	threads = nthreads()
	dimCells = ndims(sA) - (indSplit - 1)
	numCells = prod(size(sA)[indSplit : end])
	fillInds = Array{UnitRange{Int}}(undef, dimCells)
	
	for i in 1 : dimCells
		
		fillInds[i] = 1 : (size(sA)[indSplit + (i - 1)])
	end

	for subView in CartesianIndices(size(sA)[1 : (indSplit - 1)])

		vA = view(sA, subView, fillInds...)
		offset = 0 
		
		for lV in 1 : length(subView)
			
			if lV == 1
		
				offset = subView[1] - 1
				continue
		
			else
				
				for j in 2 : lV
		
					offset = offset + (subView[j] - 1) * 
					prod(size(sA)[1 : (j - 1)])
				end
			end
		end
		
		offset = offset * numCells
		
		@threads for t in 1 : threads
			
			@inbounds for tInd in ((div(numCells * (t - 1), threads) + 1) : 
			div(numCells * t, threads)) 

				vA[tInd] = convert(eltype(sA), unsafe_load(ptr, tInd + offset))
			end
		end
	end

	return nothing
end
"""
	
    threadCpy!(srcMem::Array{T, 1} where {T <: Number}, 
    trgMem::Array{T, 1} where {T <: Number})::Nothing

Copy information between two memory locations using all available threads.  
"""
function threadCpy!(srcMem::AbstractArray{T, 1} where {T <: Number}, 
	trgMem::AbstractArray{T, 1} where {T <: Number})::Nothing

	if(eltype(srcMem) <: eltype(trgMem))
		
		workBounds = genWorkBounds((1, length(srcMem)), nthreads())
		thrdBounds = (nthreads() < length(srcMem)) ? nthreads() : length(srcMem)

		@threads for tItr in 1 : thrdBounds
			
			@inbounds for itr in workBounds[tItr][1] : workBounds[tItr][2]

				trgMem[itr] = srcMem[itr]
			end
		end

	else
		
		error("Source and target memory types are not compatible.")
		return nothing
	end
end

"""
	
    threadCpy!(srcMem::AbstractArray{T, 1} where {T <: Number}, 
    trgMem::AbstractArray{T, 1} where {T <: Number}, threadNum::Int)::Nothing

Copy information between two memory locations using a set number of threads.  
"""
function threadCpy!(srcMem::AbstractArray{T, 1} where {T <: Number}, 
	trgMem::AbstractArray{T, 1} where {T <: Number}, threadNum::Int)::Nothing

	if(eltype(srcMem) <: eltype(trgMem))
		
		threadNum = (threadNum < nthreads()) ? threadNum : nthreads()
		workBounds = genWorkBounds((1, length(srcMem)), threadNum)
		thrdBounds = (threadNum < length(srcMem)) ? threadNum : length(srcMem)

		@threads for tItr in 1 : thrdBounds
			
			@inbounds for itr in workBounds[tItr][1] : workBounds[tItr][2]
				
				trgMem[itr] = srcMem[itr]
			end
		end
	else
		error("Source and target memory types are not compatible.")
		return nothing
	end
end
"""
	
    threadUpd!(updateMode::Int, mutateMem::Array{T, 1} where {T <: Number}, 
    updateMem::Array{T, 1} where {T <: Number})::Nothing

Add or subtract update values to an array using all available Julia threads.  
"""
function threadUpd!(updateMode::Int, mutateMem::Array{T, 1} where {T <: Number}, 
	updateMem::Array{T, 1} where {T <: Number})::Nothing

	if(eltype(srcMem) <: eltype(trgMem))

		workBounds = genWorkBounds((1, length(srcMem)), nthreads())
		thrdBounds = (nthreads() < length(srcMem)) ? nthreads() : length(srcMem)

		if(updateMode == 1)
			
			@threads for tItr in 1 : thrdBounds

				@inbounds for itr in workBounds[tItr][1] : workBounds[tItr][2]
					
					mutateMem[itr] += updateMem[itr]
				end
			end

			return nothing

		elseif(updateMode == 2)

			@threads for tItr in 1 : thrdBounds
				
				@inbounds for itr in workBounds[tItr][1] : workBounds[tItr][2]

					mutateMem[itr] -= updateMem[itr]
				end
			end

			return nothing

		else
			
			error("Unrecognized update mode.")
			return nothing	
		end
	else
		error("Source and target memory types are not compatible.")
		return nothing
	end
end
"""
	
    threadEleWise!(srcMem1::Array{T, 1} where {T <: Number}, 
    srcMem2::Array{T, 1} where {T <: Number}, 
    trgMem::Array{T, 1} where {T <: Number})::Nothing

Elementwise multiplication using all available Julia threads.  
"""
function threadEleWise!(srcMem1::Array{T, 1} where {T <: Number}, 
	srcMem2::Array{T, 1} where {T <: Number}, 
	trgMem::Array{T, 1} where {T <: Number})::Nothing

	if((eltype(srcMem) <: eltype(trgMem)) && (eltype(srcMem) <: eltype(trgMem)))

		workBounds = genWorkBounds((1, length(srcMem)), nthreads())
		thrdBounds = (nthreads() < length(srcMem)) ? nthreads() : length(srcMem)

		@threads for tItr in 1 : thrdBounds

			@inbounds for itr in workBounds[tItr][1] : workBounds[tItr][2]
			
				trgMem[itr] += srcMem1[itr] * srcMem2[itr]
			end
		end
	else
		error("Memory types are not compatible.")
		return nothing
	end
	return nothing
end
"""
	
    threadEleWise!(srcMem1::Array{T, 1} where {T <: Number}, 
    srcMem2::Array{T, 1} where {T <: Number}, 
    trgMem::Array{T, 1} where {T <: Number}, numThreads::Int)::Nothing

Elementwise multiplication using a set number of Julia threads.  
"""
function threadEleWise!(srcMem1::Array{T, 1} where {T <: Number}, 
	srcMem2::Array{T, 1} where {T <: Number}, 
	trgMem::Array{T, 1} where {T <: Number}, numThreads::Int)::Nothing

	if((eltype(srcMem) <: eltype(trgMem)) && (eltype(srcMem) <: eltype(trgMem)))
		
		workBounds = genWorkBounds((1, length(srcMem)), numThreads)
		thrdBounds = (numThreads < length(srcMem)) ? numThreads : length(srcMem)

		@threads for tItr in 1 : thrdBounds

			@inbounds for itr in workBounds[tItr][1] : workBounds[tItr][2]
				
				trgMem[itr] += srcMem1[itr] * srcMem2[itr]
			end
		end

	else
		
		error("Memory types are not compatible.")
		return nothing
	end

	return nothing
end

# using MaxGParallelUtilities, MaxGStructs, MaxGBasisIntegrals  
# export genGreenExt!, genGreenSlf!
# export facePairs, cubeFaces, separationGrid, greenSlfFunc!
"""

	genGreenExt!(greenCirc::Array{ComplexF64}, srcVol::MaxGVol, 
	trgVol::MaxGVol, assemblyInfo::MaxGAssemblyOpts)::Nothing

Calculate circulant form for the discrete Green function between a target 
volume, trgVol, and a source domain, srcVol.
"""
function genGreenExt!(greenCirc::Array{ComplexF64}, trgVol::MaxGVol, 
	srcVol::MaxGVol, assemblyInfo::MaxGAssemblyOpts) # ::Nothing

	fPairs = facePairs()
	trgFaces = cubeFaces(trgVol.scale)
	srcFaces = cubeFaces(srcVol.scale)
	sGridT = separationGrid(trgVol, srcVol, 0)
	sGridS = separationGrid(trgVol, srcVol, 1)
	glQuadFar = gaussQuad2(assemblyInfo.ordGLIntFar)
	glQuadMed = gaussQuad2(assemblyInfo.ordGLIntMed)
	glQuadNear = gaussQuad2(assemblyInfo.ordGLIntNear)
	G = assembleGreenCircExt!(greenCirc, trgVol, srcVol, sGridT, sGridS, glQuadFar, 
		glQuadMed, glQuadNear, trgFaces, srcFaces, fPairs, assemblyInfo)
	
	return G #nothing
end
"""
	
	genGreenSlf!(greenCirc::Array{ComplexF64}, slfVol::MaxGVol, 
	assemblyInfo::MaxGAssemblyOpts)::Nothing

Calculate circulant form for the discrete Green function on a single domain.
"""
function genGreenSlf!(greenCirc::Array{ComplexF64}, slfVol::MaxGVol, 
	assemblyInfo::MaxGAssemblyOpts)#::Nothing

	fPairs = facePairs()
	trgFaces = cubeFaces(slfVol.scale)
	srcFaces = cubeFaces(slfVol.scale)
	glQuadFar = gaussQuad2(assemblyInfo.ordGLIntFar)
	glQuadMed = gaussQuad2(assemblyInfo.ordGLIntMed)
	glQuadNear = gaussQuad2(assemblyInfo.ordGLIntNear)
	sGrid = separationGrid(slfVol, slfVol, 0)
	G = assembleGreenCircSelf!(greenCirc, slfVol, sGrid, glQuadFar, glQuadMed, 
		glQuadNear, trgFaces, srcFaces, fPairs, assemblyInfo)
	
	return G #nothing
end
"""
Generate the circulant form of the external Green function between a pair of 
distinct domains. 
"""
function assembleGreenCircExt!(greenCirc::Array{ComplexF64}, trgVol::MaxGVol, 
	srcVol::MaxGVol, sGridT::Array{<:StepRangeLen,1}, 
	sGridS::Array{<:StepRangeLen,1}, glQuadFar::Array{Float64,2}, 
	glQuadMed::Array{Float64,2}, glQuadNear::Array{Float64,2}, 
	trgFaces::Array{Float64,3}, srcFaces::Array{Float64,3}, 
	fPairs::Array{Int64,2}, assemblyInfo::MaxGAssemblyOpts)#::Nothing

	scaleT = trgVol.scale
	scaleS = srcVol.scale
	indSplit1 = trgVol.cells[1]
	indSplit2 = trgVol.cells[2]
	indSplit3 = trgVol.cells[3]

	greenExt! = (greenMat::SubArray{ComplexF64,2}, ind1::Int64, ind2::Int64, 
		ind3::Int64) -> 
	greenExtFunc!(greenMat, ind1, ind2, ind3, indSplit1, indSplit2, indSplit3, 
		sGridT, sGridS, glQuadFar, glQuadMed, glQuadNear, scaleT, scaleS, 
		trgFaces, srcFaces, fPairs, assemblyInfo)
	threadArrayW!(greenCirc, 3, size(greenCirc), greenExt!)
	G = greenCirc # Am I returning the correct Green's function?
	return G #nothing
end
"""
Generate Green function self interaction circulant vector.
"""
function assembleGreenCircSelf!(greenCirc::Array{ComplexF64}, slfVol::MaxGVol, 
	sGrid::Array{<:StepRangeLen,1}, glQuadFar::Array{Float64,2}, 
	glQuadMed::Array{Float64,2}, glQuadNear::Array{Float64,2}, 
	trgFaces::Array{Float64,3}, srcFaces::Array{Float64,3}, 
	fPairs::Array{Int64,2}, assemblyInfo::MaxGAssemblyOpts)# ::Nothing

	# Allocate array to store intermediate Toeplitz interaction form.
	greenToe = Array{ComplexF64}(undef, 3, 3, slfVol.cells[1], slfVol.cells[2], 
		slfVol.cells[3])
	# Write Green function, ignoring singular integrals. 
	greenFunc! = (greenMat::SubArray{ComplexF64,2}, ind1::Int64, ind2::Int64, 
		ind3::Int64) -> greenSlfFunc!(greenMat, ind1, ind2, ind3, sGrid, 
		glQuadFar, glQuadMed, glQuadNear, slfVol.scale, trgFaces, srcFaces, 
		fPairs, assemblyInfo)
	threadArrayW!(greenToe, 3, size(greenToe), greenFunc!)
	# 1D quadrature points for singular integrals.
	glQuadNear1 = gaussQuad1(assemblyInfo.ordGLIntNear) 
	# Correction values for singular integrals.
	wS = ^(slfVol.scale, -3) * weakS(slfVol.scale, glQuadNear1, assemblyInfo)
	wE = ^(slfVol.scale, -3) .* weakE(slfVol.scale, glQuadNear1, assemblyInfo)
	wV = ^(slfVol.scale, -3) .* weakV(slfVol.scale, glQuadNear1, assemblyInfo)
	# Correct singular integrals for coincident and adjacent cells.
	greenFunc! = (greenMat::SubArray{ComplexF64,2}, ind1::Int64, ind2::Int64, 
		ind3::Int64) -> greenSingFunc!(greenMat, ind1, ind2, ind3, wS, wE, wV, 
		sGrid, glQuadNear1, glQuadNear, slfVol.scale, trgFaces, srcFaces, 
		fPairs, assemblyInfo)
	threadArrayW!(greenToe, 3, (3, 3, min(slfVol.cells[1], 2), 
		min(slfVol.cells[2], 2), min(slfVol.cells[1], 2)), greenFunc!)
	# Embed self Green function into a circulant form
	indSplit1 = div(size(greenCirc)[3], 2)
	indSplit2 = div(size(greenCirc)[4], 2)
	indSplit3 = div(size(greenCirc)[5], 2)
	embedFunc = (greenMat::SubArray{ComplexF64,2}, ind1::Int64, ind2::Int64, 
		ind3::Int64) -> greenToeToCirc!(greenMat, greenToe, ind1, ind2, ind3, 
		indSplit1, indSplit2, indSplit3)
	threadArrayW!(greenCirc, 3, size(greenCirc), embedFunc)
	# print("greenCirc", greenCirc, "\n")
	G = greenCirc # This is the Green's function we want, right? Is this the correct one we want to be returning?
	# Why do all the function return nothing and aren't able to return anything else?
	# It's because of th ::Nothing after naming the function with it's inputs.
	return G # nothing
end
# print("G", G, "\n")
"""
Generate circulant self Green function from Toeplitz self Green function. The 
implemented mask takes into account the relative flip in the assumed dipole 
direction under a coordinate reflection. 
"""
function greenToeToCirc!(greenMat::SubArray{ComplexF64,2}, 
	greenToe::Array{ComplexF64}, ind1::Int64, ind2::Int64, ind3::Int64, 
	indSplit1::Int64, indSplit2::Int64, indSplit3::Int64)::Nothing
	
	fi = indFlip(ind1, indSplit1)
	fj = indFlip(ind2, indSplit2)
	fk = indFlip(ind3, indSplit3)

	greenMat .= (greenToe[:, :, indSelect(ind1, indSplit1), 
	indSelect(ind2, indSplit2), indSelect(ind3, indSplit3)] .* 
	(convert(Array{ComplexF64}, [
		1.0       (fi * fj) (fi * fk) 
		(fj * fi) 1.0       (fj * fk)
		(fk * fi) (fk * fj) 1.0])))

	return nothing
end
"""
Write Green function element for a pair of cubes in distinct domains. Recall 
that grids span the separations between a pair of volumes. 
"""
@inline function greenExtFunc!(greenMat::SubArray{ComplexF64}, ind1::Int64, 
	ind2::Int64, ind3::Int64, indSplit1::Int64, indSplit2::Int64, 
	indSplit3::Int64, sGridT::Array{<:StepRangeLen,1}, 
	sGridS::Array{<:StepRangeLen,1}, glQuadFar::Array{Float64,2}, 
	glQuadMed::Array{Float64,2}, glQuadNear::Array{Float64,2}, scaleT::Float64, 
	scaleS::Float64, trgFaces::Array{Float64,3}, srcFaces::Array{Float64,3}, 
	fPairs::Array{Int64,2}, assemblyInfo::MaxGAssemblyOpts)::Nothing
	
	greenFunc!(greenMat, gridSelect(ind1, indSplit1, 1, sGridT, sGridS), 
		gridSelect(ind2, indSplit2, 2, sGridT, sGridS), 
		gridSelect(ind3, indSplit3, 3, sGridT, sGridS), 
		quadSelect(gridSelect(ind1, indSplit1, 1, sGridT, sGridS), 
			gridSelect(ind2, indSplit2, 2, sGridT, sGridS), 
			gridSelect(ind3, indSplit3, 3, sGridT, sGridS), min(scaleT, scaleS), 
			glQuadFar, glQuadMed, glQuadNear, assemblyInfo), scaleT, scaleS, 
		trgFaces, srcFaces, fPairs, assemblyInfo)
end
"""
Write Green element for a pair of cubes in a common domain. 
"""
@inline function greenSlfFunc!(greenMat::SubArray{ComplexF64}, ind1::Int64, 
	ind2::Int64, ind3::Int64, sGrid::Array{<:StepRangeLen,1}, 
	glQuadFar::Array{Float64,2}, glQuadMed::Array{Float64,2}, 
	glQuadNear::Array{Float64,2}, scale::Float64, trgFaces::Array{Float64,3}, 
	srcFaces::Array{Float64,3}, fPairs::Array{Int64,2}, 
	assemblyInfo::MaxGAssemblyOpts)::Nothing
	
	greenFunc!(greenMat, sGrid[1][ind1], sGrid[2][ind2], sGrid[3][ind3], 
		quadSelect(sGrid[1][ind1], sGrid[2][ind2], sGrid[3][ind3], scale, 
			glQuadFar, glQuadMed, glQuadNear, assemblyInfo), scale, scale, 
		trgFaces, srcFaces, fPairs, assemblyInfo)
end
"""
Write a general Green function element to a shared memory array. 
"""
function greenFunc!(greenMat::SubArray{ComplexF64,2}, gridX::Float64, 
	gridY::Float64, gridZ::Float64, glQuad2::Array{Float64,2}, scaleT::Float64, 
	scaleS::Float64, trgFaces::Array{Float64,3}, srcFaces::Array{Float64,3}, 
	fPairs::Array{Int64,2}, assemblyInfo::MaxGAssemblyOpts)::Nothing

	surfMat = zeros(ComplexF64, 36)
	# Green function between all cube faces.
	greenSurfs!(gridX, gridY, gridZ, surfMat, glQuad2, scaleT, scaleS, trgFaces, 
		srcFaces, fPairs, assemblyInfo)
	# Add cube face contributions depending on source and target current 
	# orientation. 
	surfSums!(greenMat::SubArray{ComplexF64,2}, surfMat::Array{ComplexF64,1})
	
	return nothing
end
"""
Generate Green elements for adjacent cubes, assumed to be in the same domain. 
wS, wE, and wV refer to self-intersecting, edge intersecting, and vertex 
intersecting cube face integrals respectively. In the wE and wV cases, the 
first value returned is for in-plane faces, and the second value is for 
``corned'' faces. 
"""
function greenSingFunc!(greenMat::SubArray{ComplexF64,2}, ind1::Int64, 
	ind2::Int64, ind3::Int64, wS::ComplexF64, wE::Tuple{ComplexF64,ComplexF64}, 
	wV::Tuple{ComplexF64,ComplexF64}, sGrid::Array{<:StepRangeLen,1}, 
	glQuad1::Array{Float64,2}, glQuad2::Array{Float64,2}, scale::Float64, 
	trgFaces::Array{Float64,3}, srcFaces::Array{Float64,3}, 
	fPairs::Array{Int64,2}, assemblyInfo::MaxGAssemblyOpts)

	# Uncorrected surface integrals.
	surfMat = zeros(ComplexF64, 36)
	greenSurfs!(sGrid[1][ind1], sGrid[2][ind2], sGrid[2][ind3], surfMat, 
		glQuad2, scale, scale, trgFaces, srcFaces, fPairs, assemblyInfo)
	# Index based corrections.
	if (ind1, ind2, ind3) == (1, 1, 1) 
		
		correctionVal = [
		wS 	   0.0    wE[2]  wE[2]  wE[2]  wE[2]
		0.0    wS     wE[2]  wE[2]  wE[2]  wE[2]
		wE[2]  wE[2]  wS     0.0    wE[2]  wE[2]
		wE[2]  wE[2]  0.0    wS     wE[2]  wE[2]
		wE[2]  wE[2]  wE[2]  wE[2]  wS     0.0
		wE[2]  wE[2]  wE[2]  wE[2]  0.0    wS]
		
		mask =[
		1 0 1 1 1 1
		0 1 1 1 1 1
		1 1 1 0 1 1
		1 1 0 1 1 1
		1 1 1 1 1 0
		1 1 1 1 0 1]
	
	elseif (ind1, ind2, ind3) == (2, 1, 1) 
		
		correctionVal = [
		0.0  wS     wE[2]  wE[2]  wE[2]  wE[2]
		0.0  0.0    0.0    0.0    0.0    0.0
		0.0  wE[2]  wE[1]  0.0    wV[2]  wV[2]
		0.0  wE[2]  0.0    wE[1]  wV[2]  wV[2]
		0.0  wE[2]  wV[2]  wV[2]  wE[1]  0.0
		0.0  wE[2]  wV[2]  wV[2]  0.0    wE[1]]
		
		mask = [
		0 1 1 1 1 1
		0 0 0 0 0 0
		0 1 1 0 1 1
		0 1 0 1 1 1
		0 1 1 1 1 0
		0 1 1 1 0 1]

	elseif (ind1, ind2, ind3) == (2, 1, 2)
		
		correctionVal = [
		0.0  wE[1]  wV[2]  wV[2]  0.0  wE[2]
		0.0  0.0    0.0    0.0    0.0  0.0
		0.0  wV[2]  wV[1]  0.0    0.0  wV[2]
		0.0  wV[2]  0.0    wV[1]  0.0  wV[2]
		0.0  wE[2]  wV[2]  wV[2]  0.0  wE[1]
		0.0  0.0    0.0    0.0    0.0  0.0]
		
		mask = [
		0 1 1 1 0 1
		0 0 0 0 0 0
		0 1 1 0 0 1
		0 1 0 1 0 1
		0 1 1 1 0 1
		0 0 0 0 0 0] 

	elseif (ind1, ind2, ind3) == (1, 1, 2) 
		
		correctionVal = [
		wE[1]  0.0    wV[2]  wV[2]  0.0  wE[2]
		0.0    wE[1]  wV[2]  wV[2]  0.0  wE[2]
		wV[2]  wV[2]  wE[1]  0.0    0.0  wE[2]
		wV[2]  wV[2]  0.0    wE[1]  0.0  wE[2]
		wE[2]  wE[2]  wE[2]  wE[2]  0.0  wS
		0.0    0.0    0.0    0.0    0.0  0.0]
		
		mask = [
		1 0 1 1 0 1
		0 1 1 1 0 1
		1 1 1 0 0 1
		1 1 0 1 0 1
		1 1 1 1 0 1
		0 0 0 0 0 0]

	elseif (ind1, ind2, ind3) == (1, 2, 1) 
		
		correctionVal = [
		wE[1]  0.0    0.0  wE[2]  wV[2]  wV[2]
		0.0    wE[1]  0.0  wE[2]  wV[2]  wV[2]
		wE[2]  wE[2]  0.0  wS     wE[2]  wE[2]
		0.0    0.0    0.0  0.0    0.0    0.0
		wV[2]  wV[2]  0.0  wE[2]  wE[1]  0.0
		wV[2]  wV[2]  0.0  wE[2]  0.0    wE[1]]
		
		mask = [
		1 0 0 1 1 1
		0 1 0 1 1 1
		1 1 0 1 1 1
		0 0 0 0 0 0
		1 1 0 1 1 0
		1 1 0 1 0 1]

	elseif (ind1, ind2, ind3) == (2, 2, 1) 

		correctionVal = [
		0.0  wE[1]  0.0  wE[2]  wV[2]  wV[2]
		0.0  0.0    0.0  0.0    0.0    0.0
		0.0  wE[2]  0.0  wE[1]  wV[2]  wV[2]
		0.0  0.0    0.0  0.0    0.0    0.0
		0.0  wV[2]  0.0  wV[2]  wV[1]  0.0
		0.0  wV[2]  0.0  wV[2]  0.0    wV[1]]

		mask = [
		0 1 0 1 1 1
		0 0 0 0 0 0
		0 1 0 1 1 1
		0 0 0 0 0 0
		0 1 0 1 1 0
		0 1 0 1 0 1]  

	elseif (ind1, ind2, ind3) == (1, 2, 2) 
		
		correctionVal = [
		wV[1]  0.0    0.0  wV[2]  0.0  wV[2]
		0.0    wV[1]  0.0  wV[2]  0.0  wV[2]
		wV[2]  wV[2]  0.0  wE[1]  0.0  wE[2]
		0.0    0.0    0.0  0.0    0.0  0.0
		wV[2]  wV[2]  0.0  wE[2]  0.0  wE[1]
		0.0    0.0    0.0  0.0    0.0  0.0]
		
		mask = [
		1 0 0 1 0 1
		0 1 0 1 0 1
		1 1 0 1 0 1
		0 0 0 0 0 0
		1 1 0 1 0 1
		0 0 0 0 0 0]  

	elseif (ind1, ind2, ind3) == (2, 2, 2) 
		
		correctionVal = [
		0.0  wV[1]  0.0  wV[2]  0.0  wV[2]
		0.0  0.0    0.0  0.0    0.0  0.0
		0.0  wV[2]  0.0  wV[1]  0.0  wV[2]
		0.0  0.0    0.0  0.0    0.0  0.0
		0.0  wV[2]  0.0  wV[2]  0.0  wV[1]
		0.0  0.0    0.0  0.0    0.0  0.0]
		
		mask = [
		0 1 0 1 0 1
		0 0 0 0 0 0
		0 1 0 1 0 1
		0 0 0 0 0 0
		0 1 0 1 0 1
		0 0 0 0 0 0]

	else
		
		println(ind1, ind2, ind3)
		error("Attempted to access improper case.")
	end
	# Correct values of surfMat where needed
	for n in 1 : 36
		
		if mask[fPairs[n,1], fPairs[n,2]] == 1
		
			surfMat[n] = correctionVal[fPairs[n,1], fPairs[n,2]]
		end
	end
	# Overwrite problematic elements of Green function matrix.
	surfSums!(greenMat::SubArray{ComplexF64,2}, surfMat::Array{ComplexF64,1})
	
	return nothing
end
"""
Mutate greenMat to hold Green function interactions.

The storage format of greenMat, see documentation for explanation, is 
[[ii, ji, ki]^{T}; [ij, jj, kj]^{T}; [ik, jk, kk]^{T}].
"""
function surfSums!(greenMat::SubArray{ComplexF64,2}, 
	surfMat::Array{ComplexF64,1})::Nothing

	# ii
	greenMat[1,1] = surfMat[15] - surfMat[16] - surfMat[21] + surfMat[22] + 
	surfMat[29] - surfMat[30] - surfMat[35] + surfMat[36]
	# ji
	greenMat[2,1] = - surfMat[13] + surfMat[14] + surfMat[19] - surfMat[20] 
	# ki
	greenMat[3,1] = - surfMat[25] + surfMat[26] + surfMat[31] - surfMat[32] 
	# ij
	greenMat[1,2] = - surfMat[3] + surfMat[4] + surfMat[9] - surfMat[10]
	# jj
	greenMat[2,2] = surfMat[1] - surfMat[2] - surfMat[7] + surfMat[8] + 
	surfMat[29] - surfMat[30] - surfMat[35] + surfMat[36]
	# kj
	greenMat[3,2] = - surfMat[27] + surfMat[28] + surfMat[33] - surfMat[34]
	# ik
	greenMat[1,3] = - surfMat[5] + surfMat[6] + surfMat[11] - surfMat[12]
	# jk
	greenMat[2,3] = - surfMat[17] + surfMat[18] + surfMat[23] - surfMat[24]
	# kk
	greenMat[3,3] = surfMat[1] - surfMat[2] - surfMat[7] + surfMat[8] + 
	surfMat[15] - surfMat[16] - surfMat[21] + surfMat[22]

	return nothing
end
"""
Calculate the integral of the Green function over all face pairs interactions. 
"""
function greenSurfs!(gridX::Float64, gridY::Float64, gridZ::Float64, 
	surfMat::Array{ComplexF64,1}, glQuad2::Array{Float64,2}, scaleT::Float64, 
	scaleS::Float64, trgFaces::Array{Float64,3}, srcFaces::Array{Float64,3}, 
	fPairs::Array, assemblyInfo::MaxGAssemblyOpts)::Nothing

	cVX = (iP1::Int64, iP2::Int64, fp::Int64) -> 
	cubeVecAlt(1, iP1, iP2, fp, glQuad2, trgFaces, srcFaces, fPairs)
	
	cVY = (iP1::Int64, iP2::Int64, fp::Int64) -> 
	cubeVecAlt(2, iP1, iP2, fp, glQuad2, trgFaces, srcFaces, fPairs)
	
	cVZ = (iP1::Int64, iP2::Int64, fp::Int64) -> 
	cubeVecAlt(3, iP1, iP2, fp, glQuad2, trgFaces, srcFaces, fPairs)
	
	@inbounds for fp in 1 : 36

		surfMat[fp] = 0.0 + 0.0im
		# Integrate Green function over unique face pairs
		
		@inbounds for iP1 in 1 : size(glQuad2,1), iP2 in 1 : size(glQuad2,1)
			
			surfMat[fp] += ^(scaleT, -1) * ^(scaleS, 2) * glQuad2[iP1,3] * 
			glQuad2[iP2,3] * scaleGreen(distMag(cVX(iP1, iP2, fp) + gridX, 
				cVY(iP1, iP2, fp) + gridY, cVZ(iP1, iP2, fp) + gridZ), 
			assemblyInfo.freqPhase)
		end
	end

return nothing
end
"""
Generate all unique pairs of cube faces. No symmetry reductions are possible 
under the generalization that cells in distinct domains need not be of the same 
size.  
"""
function facePairs()::Array{Int64,2}

	fPairs = Array{Int64,2}(undef, 36, 2)
	
	for i in 1 : 6, j in 1 : 6

		k = (i - 1) * 6 + j
		fPairs[k,1] = i
		fPairs[k,2] = j	
	end
	
	return fPairs
end
"""
Determine a directional component, set by dir, of the separation vector for a 
pair points, set by iP1 and iP2 through glQuad2, for a given pair of source and
target faces.

The relative positions used here are supplied by Gauss-Legendre quadrature, 
glQuad2, with respect to the edge vectors of the cube faces. For more 
information on these quantities, see the gaussQuad2 and cubeFaces functions.  
"""
@inline function cubeVecAlt(dir::Int64, iP1::Int64, iP2::Int64, fp::Int64, 
	glQuad2::Array{Float64,2}, trgFaces::Array{Float64,3}, 
	srcFaces::Array{Float64,3}, fPairs::Array{Int64,2})::Float64
	
	return (trgFaces[dir, 1, fPairs[fp, 1]] +
		glQuad2[iP1, 1] * (trgFaces[dir, 2, fPairs[fp, 1]] - 
			trgFaces[dir, 1, fPairs[fp, 1]]) +
		glQuad2[iP1, 2] * (trgFaces[dir, 4, fPairs[fp, 1]] - 
			trgFaces[dir, 1, fPairs[fp, 1]])) -
	(srcFaces[dir, 1, fPairs[fp,2]] +
		glQuad2[iP2, 1] * (srcFaces[dir, 2, fPairs[fp, 2]] - 
			srcFaces[dir, 1, fPairs[fp, 2]]) +
		glQuad2[iP2, 2] * (srcFaces[dir, 4, fPairs[fp, 2]] - 
			srcFaces[dir, 1, fPairs[fp, 2]]))
end
"""
Generate array of cube faces based from a characteristic length, l. 
L and U reference relative positions on the corresponding normal axis.
"""
function cubeFaces(l::Float64)::Array{Float64,3}
	
	yzL = hcat([0.0, 0.0, 0.0], [0.0, l, 0.0], [0.0, l, l], [0.0, 0.0, l])
	yzU = hcat([l, 0.0, 0.0], [l, l, 0.0], [l, l, l], [l, 0.0, l])
	xzL = hcat([0.0, 0.0, 0.0], [l, 0.0, 0.0], [l, 0.0, l], [0.0 ,0.0, l])
	xzU = hcat([0.0, l, 0.0], [l, l, 0.0], [l, l, l], [0.0, l, l])
	xyL = hcat([0.0, 0.0, 0.0], [0.0, l, 0.0], [l, l, 0.0], [l, 0.0, 0.0])
	xyU = hcat([0.0, 0.0, l], [0.0, l, l], [l, l, l], [l, 0.0, l])
	
	return cat(yzL, yzU, xzL, xzU, xyL, xyU, dims = 3)
end
"""
Generates grid of spanning separations for a pair of volumes. The flipped 
separation grid is used in the generation of the circulant form.  
"""
function separationGrid(trgVol::MaxGVol, srcVol::MaxGVol, 
	flip::Int64)::Array{<:StepRangeLen,1}
	
	start = zeros(3)
	stop = zeros(3)
	gridS = srcVol.grid
	gridT = trgVol.grid

	if flip == 1

		cellSize = round(srcVol.scale, digits = 4)
		sep = ones(3) .* cellSize

		for i in 1 : 3
		
			start[i] = round(gridT[i][1] - gridS[i][end], digits = 4)
			stop[i] = round(gridT[i][1] - gridS[i][1], digits = 4)

			if stop[i] < start[i]
		
				sep[i] *= -1.0; 
			end
		end
		
		return [start[1] : sep[1] : stop[1], start[2] : sep[2] : stop[2], 
		start[3] : sep[3] : stop[3]]
	else
		
		cellSize = round(trgVol.scale, digits = 4)
		sep = ones(3) .* cellSize
		
		for i in 1 : 3
		
			start[i] = round(gridT[i][1] - gridS[i][1], digits = 4)
			stop[i] =  round(gridT[i][end] - gridS[i][1], digits = 4)
			
			if stop[i] < start[i]
		
				sep[i] *= -1.0; 
			end
		end
		
		return [start[1] : sep[1] : stop[1], start[2] : sep[2] : stop[2], 
		start[3] : sep[3] : stop[3]]
	end 
end
"""
Select quadrature approximation based on cell proximity.
"""
@inline function quadSelect(vX::Float64, vY::Float64, vZ::Float64, 
	scale::Float64, glQuadFar::Array{Float64,2}, glQuadMed::Array{Float64,2}, 
	glQuadNear::Array{Float64,2}, 
	assemblyInfo::MaxGAssemblyOpts)::Array{Float64,2}

	if (distMag(vX, vY, vZ) / scale) > assemblyInfo.crossMedFar
		
		return glQuadFar

	elseif (distMag(vX, vY, vZ) / scale) > assemblyInfo.crossNearMed
		
		return glQuadMed
	# The value of 1.733 is slightly larger than sqrt(3). Any smaller separation 
	# between cell centers indicates the presence of a common vertex, edge, 
	# face, or volume. Such cases are treated independently by direct evaluation 
	# methods.
	elseif (distMag(vX, vY, vZ) / scale) < 1.733

		return glQuadFar

	else
		
		return glQuadNear
	end
end
"""
Return the separation between two elements from circulant embedding indices and 
domain grids. 
"""
@inline function gridSelect(ind::Int64, indSplit::Int64, dir::Int64, 
	gridT::Array{<:StepRangeLen,1}, gridS::Array{<:StepRangeLen,1})::Float64

	if ind <= indSplit
		
		return (gridT[dir][ind])

	else

		if ind > (1 + indSplit)
		
			ind -= 1
		end		
		
		return (gridS[dir][ind - indSplit])
	end
end
"""
Return a reference index relative to the embedding index of the Green function. 
"""
@inline function indSelect(ind::Int64, indSplit::Int64)::Int64

	if ind <= indSplit
		
		return ind

	else
		
		if ind == (1 + indSplit)
			
			ind -= 1

		else

			ind -= 2
		end
		
		return 2 * indSplit - ind
	end
end
"""
Flip effective dipole direction based on index values. 
"""
@inline function indFlip(ind::Int64, indSplit::Int64)::Float64

	if ind <= indSplit
		
		return 1.0

	else
		
		return -1.0
	end
end

using LinearAlgebra # , MaxGStructs
# export weakS, weakE, weakV, gaussQuad2, gaussQuad1, scaleGreen, distMag

const π = 3.14159265358979323846264338328
# Point locations for triangular tesselation.
const p0 = [0; 0; 0] 
const p1 = [1; 0; 0] 
const p2 = [1; 1; 0] 
const p3 = [0; 1; 0] 
const p4 = [2; 1; 0] 
const p5 = [2; 2; 0] 
const p7 = [1; 2; 0] 
const q0 = [1; 1; 1] 
const q1 = [1; 2; 1] 
const q2 = [1; 2; 0] 
const r0 = [2; 0; 0]
const r1 = [1; 0; 1]

"""

scaleGreen(distMag::Float64, freqPhase::ComplexF64)::ComplexF64

Returns the scalar (Helmholtz) Green function. The separation distance mag is 
assumed to be scaled (divided) by the wavelength. 
"""
@inline function scaleGreen(distMag::Float64, freqPhase::ComplexF64)::ComplexF64

	return exp(2im * π * distMag * freqPhase) / (4 * π * distMag)
end
"""

distMag(v1::Float64, v2::Float64, v3::Float64)::Float64

Returns the Euclidean norm for a three dimensional vector. 
"""
@inline function distMag(v1::Float64, v2::Float64, v3::Float64)::Float64
	
	return sqrt(v1^2 + v2^2 + v3^2)
end
"""

weakS(scale::Float64, glQuad1::Array{Float64,2}, 
assemblyOpts::MaxGAssemblyOpts)::Tuple{ComplexF64,ComplexF64}	

Head function for integration over coincident square panels. The scale 
parameter is the size of a cubic voxel relative to the wavelength. glQuad1 is 
an array of Gauss-Legendre quadrature weights and positions. The assemblyOps 
parameter determines the level of precision used for integral calculations. 
Namely, assemblyOpts.ordGLIntNear is used internally in all weakly singular 
integral computations. 
"""
function weakS(scale::Float64, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::ComplexF64

	return weakSInt(hcat(p0, p1, p2) .* scale, glQuad1, assemblyOpts) +
	weakEInt(hcat(p0, p2, p1, p3, p2, p0) .* scale, glQuad1, assemblyOpts) +
	weakEInt(hcat(p0, p2, p3, p1, p2, p0) .* scale, glQuad1, assemblyOpts) + 
	weakSInt(hcat(p0, p2, p3) .* scale, glQuad1, assemblyOpts)
end
"""

weakE(scale::Float64, glQuad1::Array{Float64,2}, 
assemblyOpts::MaxGAssemblyOpts)::Tuple{ComplexF64,ComplexF64}	

Head function for integration over edge adjacent square panels. See weakS for 
input parameter descriptions. 
"""
function weakE(scale::Float64, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::Tuple{ComplexF64,ComplexF64}
	
	# Evaluation of in-plane (flat) panels followed by cornered panels. 
	
	return (weakEInt(hcat(p1, p2, p0, r0, p2, p1) .* scale, glQuad1, assemblyOpts) +
		weakVInt(hcat(p2, p0, p1, r0, p4, p2) .* scale, glQuad1, assemblyOpts) +
		weakVInt(hcat(p2, p3, p0, p1, r0, p2) .* scale, glQuad1, assemblyOpts) +
		weakVInt(hcat(p2, p3, p0, r0, p4, p2) .* scale, glQuad1, assemblyOpts), 
		weakEInt(hcat(p1, p2, p0, r1, p2, p1) .* scale, glQuad1, assemblyOpts) +
		weakVInt(hcat(p2, p0, p1, r1, q0, p2) .* scale, glQuad1, assemblyOpts) +
		weakVInt(hcat(p2, p3, p0, p1, r1, p2) .* scale, glQuad1, assemblyOpts) +
		weakVInt(hcat(p2, p3, p0, r1, q0, p2) .* scale, glQuad1, assemblyOpts))
end
"""

weakV(scale::Float64, glQuad1::Array{Float64,2}, 
assemblyOpts::MaxGAssemblyOpts)::Tuple{ComplexF64,ComplexF64}	

Head function returning integral values for the Green function over vertex 
adjacent square panels. See weakS for input parameter descriptions. 
"""
function weakV(scale::Float64, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::Tuple{ComplexF64,ComplexF64}

	# Evaluation for in-plane (flat) panels followed by cornered panels. 
	return (
		weakVInt(hcat(p2, p0, p1, p4, p5, p2) .* scale, glQuad1, assemblyOpts) +
		weakVInt(hcat(p2, p0, p1, p5, p7, p2) .* scale, glQuad1, assemblyOpts) +
		weakVInt(hcat(p2, p3, p0, p4, p5, p2) .* scale, glQuad1, assemblyOpts) +
		weakVInt(hcat(p2, p3, p0, p5, p7, p2) .* scale, glQuad1, assemblyOpts), 
		weakVInt(hcat(p2, p0, p1, q0, q1, p2) .* scale, glQuad1, assemblyOpts) +
		weakVInt(hcat(p2, p0, p1, q1, q2, p2) .* scale, glQuad1, assemblyOpts) +
		weakVInt(hcat(p2, p3, p0, q0, q1, p2) .* scale, glQuad1, assemblyOpts) +
		weakVInt(hcat(p2, p3, p0, q1, q2, p2) .* scale, glQuad1, assemblyOpts))
end
#=
The code contained in transformBasisIntegrals evaluates the integrands called 
by the weakS, weakE, and weakV head functions using a series of variable 
transformations and analytic integral evaluations---reducing the four 
dimensional surface integrals performed for ``standard'' cells to chains of one 
dimensional integrals. No comments are included in this low level code, which 
is simply a julia translation of DIRECTFN_E by Athanasios Polimeridis. For a 
complete description of the steps being performed see the article cited above 
and references included therein. 
=#
include("transformBasisIntegrals.jl")
"""
gaussQuad2(ord::Int64)::Array{Float64,2}

Returns locations and weights for 2D Gauss-Legendre quadrature. Order must be 
an integer between 1 and 32, or equal to 64. The first column of the returned 
array is the ``x-position'', on the interval [0,1]. The second column is the 
``y-positions'', also on the interval [0,1]. The third is column is the 
evaluation weights. 
"""
function gaussQuad2(ord::Int64)::Array{Float64,2}
	
	glQuad2 = Array{Float64,2}(undef, ord * ord, 3)
	glQuad1 = gaussQuad1(ord)

	for j in 1:ord, i in 1:ord

		glQuad2[i + (j - 1) * ord, 1] = (glQuad1[i, 1] + 1.0) / 2.0
		glQuad2[i + (j - 1) * ord, 2] = (glQuad1[j, 1] + 1.0) / 2.0
		glQuad2[i + (j - 1) * ord, 3] = glQuad1[i, 2] * glQuad1[j, 2] / 4.0
	end
	
	return glQuad2
end

"""
gaussQuad1(ord::Int64)::Array{Float64,2}

Returns locations and weights for 1D Gauss-Legendre quadrature. Order must be  
an integer between 1 and 32, or equal to 64. The first column of the returned 
array is positions, on the interval [-1,1], the second column contains the 
associated weights.
"""
function gaussQuad1(ord::Int64)::Array{Float64,2}

	if (ord > 32 && ord != 64) || ord < 1
		
		error("Order parameter must be an integer between 1 and 32, or 64.")
	
	elseif ord == 64

		glQuad = Array{Float64,2}(undef, ord, 2)
		glQuad[1,1] = -0.999305041735772139456905624346
		glQuad[2,1] = -0.996340116771955279346924500676
		glQuad[3,1] = -0.991013371476744320739382383443
		glQuad[4,1] = -0.983336253884625956931299302157
		glQuad[5,1] = -0.973326827789910963741853507352
		glQuad[6,1] = -0.961008799652053718918614121897
		glQuad[7,1] = -0.946411374858402816062481491347
		glQuad[8,1] = -0.929569172131939575821490154559
		glQuad[9,1] = -0.910522137078502805756380668008
		glQuad[10,1] = -0.889315445995114105853404038273
		glQuad[11,1] = -0.865999398154092819760783385070
		glQuad[12,1] = -0.840629296252580362751691544696
		glQuad[13,1] = -0.813265315122797559741923338086
		glQuad[14,1] = -0.783972358943341407610220525214
		glQuad[15,1] = -0.752819907260531896611863774886
		glQuad[16,1] = -0.719881850171610826848940217832
		glQuad[17,1] = -0.685236313054233242563558371031
		glQuad[18,1] = -0.648965471254657339857761231993
		glQuad[19,1] = -0.611155355172393250248852971019
		glQuad[20,1] = -0.571895646202634034283878116659
		glQuad[21,1] = -0.531279464019894545658013903544
		glQuad[22,1] = -0.489403145707052957478526307022
		glQuad[23,1] = -0.446366017253464087984947714759
		glQuad[24,1] = -0.402270157963991603695766771260
		glQuad[25,1] = -0.357220158337668115950442615046
		glQuad[26,1] = -0.311322871990210956157512698560
		glQuad[27,1] = -0.264687162208767416373964172510
		glQuad[28,1] = -0.217423643740007084149648748989
		glQuad[29,1] = -0.169644420423992818037313629748
		glQuad[30,1] = -0.121462819296120554470376463492
		glQuad[31,1] = -0.729931217877990394495429419403f-01
		glQuad[32,1] = -0.243502926634244325089558428537f-01
		glQuad[33,1] = 0.243502926634244325089558428537f-01
		glQuad[34,1] = 0.729931217877990394495429419403f-01
		glQuad[35,1] = 0.121462819296120554470376463492
		glQuad[36,1] = 0.169644420423992818037313629748
		glQuad[37,1] = 0.217423643740007084149648748989
		glQuad[38,1] = 0.264687162208767416373964172510
		glQuad[39,1] = 0.311322871990210956157512698560
		glQuad[40,1] = 0.357220158337668115950442615046
		glQuad[41,1] = 0.402270157963991603695766771260
		glQuad[42,1] = 0.446366017253464087984947714759
		glQuad[43,1] = 0.489403145707052957478526307022
		glQuad[44,1] = 0.531279464019894545658013903544
		glQuad[45,1] = 0.571895646202634034283878116659
		glQuad[46,1] = 0.611155355172393250248852971019
		glQuad[47,1] = 0.648965471254657339857761231993
		glQuad[48,1] = 0.685236313054233242563558371031
		glQuad[49,1] = 0.719881850171610826848940217832
		glQuad[50,1] = 0.752819907260531896611863774886
		glQuad[51,1] = 0.783972358943341407610220525214
		glQuad[52,1] = 0.813265315122797559741923338086
		glQuad[53,1] = 0.840629296252580362751691544696
		glQuad[54,1] = 0.865999398154092819760783385070
		glQuad[55,1] = 0.889315445995114105853404038273
		glQuad[56,1] = 0.910522137078502805756380668008
		glQuad[57,1] = 0.929569172131939575821490154559
		glQuad[58,1] = 0.946411374858402816062481491347
		glQuad[59,1] = 0.961008799652053718918614121897
		glQuad[60,1] = 0.973326827789910963741853507352
		glQuad[61,1] = 0.983336253884625956931299302157
		glQuad[62,1] = 0.991013371476744320739382383443
		glQuad[63,1] = 0.996340116771955279346924500676
		glQuad[64,1] = 0.999305041735772139456905624346
		glQuad[1,2] = 0.178328072169643294729607914497f-02
		glQuad[2,2] = 0.414703326056246763528753572855f-02
		glQuad[3,2] = 0.650445796897836285611736039998f-02
		glQuad[4,2] = 0.884675982636394772303091465973f-02
		glQuad[5,2] = 0.111681394601311288185904930192f-01
		glQuad[6,2] = 0.134630478967186425980607666860f-01
		glQuad[7,2] = 0.157260304760247193219659952975f-01
		glQuad[8,2] = 0.179517157756973430850453020011f-01
		glQuad[9,2] = 0.201348231535302093723403167285f-01
		glQuad[10,2] = 0.222701738083832541592983303842f-01
		glQuad[11,2] = 0.243527025687108733381775504091f-01
		glQuad[12,2] = 0.263774697150546586716917926252f-01
		glQuad[13,2] = 0.283396726142594832275113052002f-01
		glQuad[14,2] = 0.302346570724024788679740598195f-01
		glQuad[15,2] = 0.320579283548515535854675043479f-01
		glQuad[16,2] = 0.338051618371416093915654821107f-01
		glQuad[17,2] = 0.354722132568823838106931467152f-01
		glQuad[18,2] = 0.370551285402400460404151018096f-01
		glQuad[19,2] = 0.385501531786156291289624969468f-01
		glQuad[20,2] = 0.399537411327203413866569261283f-01
		glQuad[21,2] = 0.412625632426235286101562974736f-01
		glQuad[22,2] = 0.424735151236535890073397679088f-01
		glQuad[23,2] = 0.435837245293234533768278609737f-01
		glQuad[24,2] = 0.445905581637565630601347100309f-01
		glQuad[25,2] = 0.454916279274181444797709969713f-01
		glQuad[26,2] = 0.462847965813144172959532492323f-01
		glQuad[27,2] = 0.469681828162100173253262857546f-01
		glQuad[28,2] = 0.475401657148303086622822069442f-01
		glQuad[29,2] = 0.479993885964583077281261798713f-01
		glQuad[30,2] = 0.483447622348029571697695271580f-01
		glQuad[31,2] = 0.485754674415034269347990667840f-01
		glQuad[32,2] = 0.486909570091397203833653907347f-01
		glQuad[33,2] = 0.486909570091397203833653907347f-01
		glQuad[34,2] = 0.485754674415034269347990667840f-01
		glQuad[35,2] = 0.483447622348029571697695271580f-01
		glQuad[36,2] = 0.479993885964583077281261798713f-01
		glQuad[37,2] = 0.475401657148303086622822069442f-01
		glQuad[38,2] = 0.469681828162100173253262857546f-01
		glQuad[39,2] = 0.462847965813144172959532492323f-01
		glQuad[40,2] = 0.454916279274181444797709969713f-01
		glQuad[41,2] = 0.445905581637565630601347100309f-01
		glQuad[42,2] = 0.435837245293234533768278609737f-01
		glQuad[43,2] = 0.424735151236535890073397679088f-01
		glQuad[44,2] = 0.412625632426235286101562974736f-01
		glQuad[45,2] = 0.399537411327203413866569261283f-01
		glQuad[46,2] = 0.385501531786156291289624969468f-01
		glQuad[47,2] = 0.370551285402400460404151018096f-01
		glQuad[48,2] = 0.354722132568823838106931467152f-01
		glQuad[49,2] = 0.338051618371416093915654821107f-01
		glQuad[50,2] = 0.320579283548515535854675043479f-01
		glQuad[51,2] = 0.302346570724024788679740598195f-01
		glQuad[52,2] = 0.283396726142594832275113052002f-01
		glQuad[53,2] = 0.263774697150546586716917926252f-01
		glQuad[54,2] = 0.243527025687108733381775504091f-01
		glQuad[55,2] = 0.222701738083832541592983303842f-01
		glQuad[56,2] = 0.201348231535302093723403167285f-01
		glQuad[57,2] = 0.179517157756973430850453020011f-01
		glQuad[58,2] = 0.157260304760247193219659952975f-01
		glQuad[59,2] = 0.134630478967186425980607666860f-01
		glQuad[60,2] = 0.111681394601311288185904930192f-01
		glQuad[61,2] = 0.884675982636394772303091465973f-02
		glQuad[62,2] = 0.650445796897836285611736039998f-02
		glQuad[63,2] = 0.414703326056246763528753572855f-02
		glQuad[64,2] = 0.178328072169643294729607914497f-02
	
		return glQuad

	elseif ord <= 16
		
		glQuad = Array{Float64,2}(undef, ord, 2)

		if ord <= 8

			if ord <= 4

				if ord == 1
					glQuad[1,1] = 0.0
					glQuad[1,2] = 2.0
				
					return glQuad

				elseif ord == 2

					glQuad[1,1] = -0.577350269189625764509148780502
					glQuad[2,1] = 0.577350269189625764509148780502
					glQuad[1,2] = 1.0
					glQuad[2,2] = 1.0
				
					return glQuad 

				elseif ord == 3    

					glQuad[1,1] = -0.774596669241483377035853079956
					glQuad[2,1] = 0.0
					glQuad[3,1] = 0.774596669241483377035853079956
					glQuad[1,2] = 5.0 / 9.0
					glQuad[2,2] = 8.0 / 9.0
					glQuad[3,2] = 5.0 / 9.0
				
					return glQuad

				else

					glQuad[1,1] = -0.861136311594052575223946488893
					glQuad[2,1] = -0.339981043584856264802665759103
					glQuad[3,1] = 0.339981043584856264802665759103
					glQuad[4,1] = 0.861136311594052575223946488893
					glQuad[1,2] = 0.347854845137453857373063949222
					glQuad[2,2] = 0.652145154862546142626936050778
					glQuad[3,2] = 0.652145154862546142626936050778
					glQuad[4,2] = 0.347854845137453857373063949222
				
					return glQuad
				end
				
			else

				if ord == 5

					glQuad[1,1] = -0.906179845938663992797626878299
					glQuad[2,1] = -0.538469310105683091036314420700
					glQuad[3,1] = 0.0
					glQuad[4,1] = 0.538469310105683091036314420700
					glQuad[5,1] = 0.906179845938663992797626878299
					glQuad[1,2] = 0.236926885056189087514264040720
					glQuad[2,2] = 0.478628670499366468041291514836
					glQuad[3,2] = 0.568888888888888888888888888889
					glQuad[4,2] = 0.478628670499366468041291514836
					glQuad[5,2] = 0.236926885056189087514264040720
				
					return glQuad

				elseif ord == 6	

					glQuad[1,1] = -0.932469514203152027812301554494
					glQuad[2,1] = -0.661209386466264513661399595020
					glQuad[3,1] = -0.238619186083196908630501721681
					glQuad[4,1] = 0.238619186083196908630501721681
					glQuad[5,1] = 0.661209386466264513661399595020
					glQuad[6,1] = 0.932469514203152027812301554494
					glQuad[1,2] = 0.171324492379170345040296142173
					glQuad[2,2] = 0.360761573048138607569833513838
					glQuad[3,2] = 0.467913934572691047389870343990
					glQuad[4,2] = 0.467913934572691047389870343990
					glQuad[5,2] = 0.360761573048138607569833513838
					glQuad[6,2] = 0.171324492379170345040296142173
				
					return glQuad

				elseif ord == 7

					glQuad[1,1] = -0.949107912342758524526189684048
					glQuad[2,1] = -0.741531185599394439863864773281
					glQuad[3,1] = -0.405845151377397166906606412077
					glQuad[4,1] = 0.0
					glQuad[5,1] = 0.405845151377397166906606412077
					glQuad[6,1] = 0.741531185599394439863864773281
					glQuad[7,1] = 0.949107912342758524526189684048
					glQuad[1,2] = 0.129484966168869693270611432679
					glQuad[2,2] = 0.279705391489276667901467771424
					glQuad[3,2] = 0.381830050505118944950369775489
					glQuad[4,2] = 0.417959183673469387755102040816
					glQuad[5,2] = 0.381830050505118944950369775489
					glQuad[6,2] = 0.279705391489276667901467771424
					glQuad[7,2] = 0.129484966168869693270611432679
				
					return glQuad

				else

					glQuad[1,1] = -0.960289856497536231683560868569
					glQuad[2,1] = -0.796666477413626739591553936476
					glQuad[3,1] = -0.525532409916328985817739049189
					glQuad[4,1] = -0.183434642495649804939476142360
					glQuad[5,1] = 0.183434642495649804939476142360
					glQuad[6,1] = 0.525532409916328985817739049189
					glQuad[7,1] = 0.796666477413626739591553936476
					glQuad[8,1] = 0.960289856497536231683560868569
					glQuad[1,2] = 0.101228536290376259152531354310
					glQuad[2,2] = 0.222381034453374470544355994426
					glQuad[3,2] = 0.313706645877887287337962201987
					glQuad[4,2] = 0.362683783378361982965150449277
					glQuad[5,2] = 0.362683783378361982965150449277
					glQuad[6,2] = 0.313706645877887287337962201987
					glQuad[7,2] = 0.222381034453374470544355994426
					glQuad[8,2] = 0.101228536290376259152531354310
				
					return glQuad
				end
			end

		else

			if ord <= 12

				if ord == 9

					glQuad[1,1] = -0.968160239507626089835576202904
					glQuad[2,1] = -0.836031107326635794299429788070
					glQuad[3,1] = -0.613371432700590397308702039341
					glQuad[4,1] = -0.324253423403808929038538014643
					glQuad[5,1] = 0.0
					glQuad[6,1] = 0.324253423403808929038538014643
					glQuad[7,1] = 0.613371432700590397308702039341
					glQuad[8,1] = 0.836031107326635794299429788070
					glQuad[9,1] = 0.968160239507626089835576202904
					glQuad[1,2] = 0.812743883615744119718921581105f-01
					glQuad[2,2] = 0.180648160694857404058472031243
					glQuad[3,2] = 0.260610696402935462318742869419
					glQuad[4,2] = 0.312347077040002840068630406584
					glQuad[5,2] = 0.330239355001259763164525069287
					glQuad[6,2] = 0.312347077040002840068630406584
					glQuad[7,2] = 0.260610696402935462318742869419
					glQuad[8,2] = 0.180648160694857404058472031243
					glQuad[9,2] = 0.812743883615744119718921581105f-01
				
					return glQuad

				elseif ord == 10

					glQuad[1,1] = -0.973906528517171720077964012084
					glQuad[2,1] = -0.865063366688984510732096688423
					glQuad[3,1] = -0.679409568299024406234327365115
					glQuad[4,1] = -0.433395394129247190799265943166
					glQuad[5,1] = -0.148874338981631210884826001130
					glQuad[6,1] = 0.148874338981631210884826001130
					glQuad[7,1] = 0.433395394129247190799265943166
					glQuad[8,1] = 0.679409568299024406234327365115
					glQuad[9,1] = 0.865063366688984510732096688423
					glQuad[10,1] = 0.973906528517171720077964012084
					glQuad[1,2] = 0.666713443086881375935688098933f-01
					glQuad[2,2] = 0.149451349150580593145776339658
					glQuad[3,2] = 0.219086362515982043995534934228
					glQuad[4,2] = 0.269266719309996355091226921569
					glQuad[5,2] = 0.295524224714752870173892994651
					glQuad[6,2] = 0.295524224714752870173892994651
					glQuad[7,2] = 0.269266719309996355091226921569
					glQuad[8,2] = 0.219086362515982043995534934228
					glQuad[9,2] = 0.149451349150580593145776339658
					glQuad[10,2] = 0.666713443086881375935688098933f-01
				
					return glQuad

				elseif ord == 11

					glQuad[1,1] = -0.978228658146056992803938001123
					glQuad[2,1] = -0.887062599768095299075157769304
					glQuad[3,1] = -0.730152005574049324093416252031
					glQuad[4,1] = -0.519096129206811815925725669459
					glQuad[5,1] = -0.269543155952344972331531985401
					glQuad[6,1] = 0.0
					glQuad[7,1] = 0.269543155952344972331531985401
					glQuad[8,1] = 0.519096129206811815925725669459
					glQuad[9,1] = 0.730152005574049324093416252031
					glQuad[10,1] = 0.887062599768095299075157769304
					glQuad[11,1] = 0.978228658146056992803938001123
					glQuad[1,2] = 0.556685671161736664827537204425f-01
					glQuad[2,2] = 0.125580369464904624634694299224
					glQuad[3,2] = 0.186290210927734251426097641432
					glQuad[4,2] = 0.233193764591990479918523704843
					glQuad[5,2] = 0.262804544510246662180688869891
					glQuad[6,2] = 0.272925086777900630714483528336
					glQuad[7,2] = 0.262804544510246662180688869891
					glQuad[8,2] = 0.233193764591990479918523704843
					glQuad[9,2] = 0.186290210927734251426097641432
					glQuad[10,2] = 0.125580369464904624634694299224
					glQuad[11,2] = 0.556685671161736664827537204425f-01
				
					return glQuad

				else

					glQuad[1,1] = -0.981560634246719250690549090149
					glQuad[2,1] = -0.904117256370474856678465866119
					glQuad[3,1] = -0.769902674194304687036893833213
					glQuad[4,1] = -0.587317954286617447296702418941
					glQuad[5,1] = -0.367831498998180193752691536644
					glQuad[6,1] = -0.125233408511468915472441369464
					glQuad[7,1] = 0.125233408511468915472441369464
					glQuad[8,1] = 0.367831498998180193752691536644
					glQuad[9,1] = 0.587317954286617447296702418941
					glQuad[10,1] = 0.769902674194304687036893833213
					glQuad[11,1] = 0.904117256370474856678465866119
					glQuad[12,1] = 0.981560634246719250690549090149
					glQuad[1,2] = 0.471753363865118271946159614850f-01
					glQuad[2,2] = 0.106939325995318430960254718194
					glQuad[3,2] = 0.160078328543346226334652529543
					glQuad[4,2] = 0.203167426723065921749064455810
					glQuad[5,2] = 0.233492536538354808760849898925
					glQuad[6,2] = 0.249147045813402785000562436043
					glQuad[7,2] = 0.249147045813402785000562436043
					glQuad[8,2] = 0.233492536538354808760849898925
					glQuad[9,2] = 0.203167426723065921749064455810
					glQuad[10,2] = 0.160078328543346226334652529543
					glQuad[11,2] = 0.106939325995318430960254718194
					glQuad[12,2] = 0.471753363865118271946159614850f-01
				
					return glQuad
				end

			else

				if ord == 13

					glQuad[1,1] = -0.984183054718588149472829448807
					glQuad[2,1] = -0.917598399222977965206547836501
					glQuad[3,1] = -0.801578090733309912794206489583
					glQuad[4,1] = -0.642349339440340220643984606996
					glQuad[5,1] = -0.448492751036446852877912852128
					glQuad[6,1] = -0.230458315955134794065528121098
					glQuad[7,1] = 0.0
					glQuad[8,1] = 0.230458315955134794065528121098
					glQuad[9,1] = 0.448492751036446852877912852128
					glQuad[10,1] = 0.642349339440340220643984606996
					glQuad[11,1] = 0.801578090733309912794206489583
					glQuad[12,1] = 0.917598399222977965206547836501
					glQuad[13,1] = 0.984183054718588149472829448807
					glQuad[1,2] = 0.404840047653158795200215922010f-01
					glQuad[2,2] = 0.921214998377284479144217759538f-01
					glQuad[3,2] = 0.138873510219787238463601776869
					glQuad[4,2] = 0.178145980761945738280046691996
					glQuad[5,2] = 0.207816047536888502312523219306
					glQuad[6,2] = 0.226283180262897238412090186040
					glQuad[7,2] = 0.232551553230873910194589515269
					glQuad[8,2] = 0.226283180262897238412090186040
					glQuad[9,2] = 0.207816047536888502312523219306
					glQuad[10,2] = 0.178145980761945738280046691996
					glQuad[11,2] = 0.138873510219787238463601776869
					glQuad[12,2] = 0.921214998377284479144217759538f-01
					glQuad[13,2] = 0.404840047653158795200215922010f-01
				
					return glQuad

				elseif ord == 14

					glQuad[1,1] = -0.986283808696812338841597266704
					glQuad[2,1] = -0.928434883663573517336391139378
					glQuad[3,1] = -0.827201315069764993189794742650
					glQuad[4,1] = -0.687292904811685470148019803019
					glQuad[5,1] = -0.515248636358154091965290718551
					glQuad[6,1] = -0.319112368927889760435671824168
					glQuad[7,1] = -0.108054948707343662066244650220
					glQuad[8,1] = 0.108054948707343662066244650220
					glQuad[9,1] = 0.319112368927889760435671824168
					glQuad[10,1] = 0.515248636358154091965290718551
					glQuad[11,1] = 0.687292904811685470148019803019
					glQuad[12,1] = 0.827201315069764993189794742650
					glQuad[13,1] = 0.928434883663573517336391139378
					glQuad[14,1] = 0.986283808696812338841597266704
					glQuad[1,2] = 0.351194603317518630318328761382f-01
					glQuad[2,2] = 0.801580871597602098056332770629f-01
					glQuad[3,2] = 0.121518570687903184689414809072
					glQuad[4,2] = 0.157203167158193534569601938624
					glQuad[5,2] = 0.185538397477937813741716590125
					glQuad[6,2] = 0.205198463721295603965924065661
					glQuad[7,2] = 0.215263853463157790195876443316
					glQuad[8,2] = 0.215263853463157790195876443316
					glQuad[9,2] = 0.205198463721295603965924065661
					glQuad[10,2] = 0.185538397477937813741716590125
					glQuad[11,2] = 0.157203167158193534569601938624
					glQuad[12,2] = 0.121518570687903184689414809072
					glQuad[13,2] = 0.801580871597602098056332770629f-01
					glQuad[14,2] = 0.351194603317518630318328761382f-01
				
					return glQuad

				elseif ord == 15

					glQuad[1,1] = -0.987992518020485428489565718587
					glQuad[2,1] = -0.937273392400705904307758947710
					glQuad[3,1] = -0.848206583410427216200648320774
					glQuad[4,1] = -0.724417731360170047416186054614
					glQuad[5,1] = -0.570972172608538847537226737254
					glQuad[6,1] = -0.394151347077563369897207370981
					glQuad[7,1] = -0.201194093997434522300628303395
					glQuad[8,1] = 0.0
					glQuad[9,1] = 0.201194093997434522300628303395
					glQuad[10,1] = 0.394151347077563369897207370981
					glQuad[11,1] = 0.570972172608538847537226737254
					glQuad[12,1] = 0.724417731360170047416186054614
					glQuad[13,1] = 0.848206583410427216200648320774
					glQuad[14,1] = 0.937273392400705904307758947710
					glQuad[15,1] = 0.987992518020485428489565718587
					glQuad[1,2] = 0.307532419961172683546283935772f-01
					glQuad[2,2] = 0.703660474881081247092674164507f-01
					glQuad[3,2] = 0.107159220467171935011869546686
					glQuad[4,2] = 0.139570677926154314447804794511
					glQuad[5,2] = 0.166269205816993933553200860481
					glQuad[6,2] = 0.186161000015562211026800561866
					glQuad[7,2] = 0.198431485327111576456118326444
					glQuad[8,2] = 0.202578241925561272880620199968
					glQuad[9,2] = 0.198431485327111576456118326444
					glQuad[10,2] = 0.186161000015562211026800561866
					glQuad[11,2] = 0.166269205816993933553200860481
					glQuad[12,2] = 0.139570677926154314447804794511
					glQuad[13,2] = 0.107159220467171935011869546686
					glQuad[14,2] = 0.703660474881081247092674164507f-01
					glQuad[15,2] = 0.307532419961172683546283935772f-01
				
					return glQuad

				else

					glQuad[1,1] = -0.989400934991649932596154173450
					glQuad[2,1] = -0.944575023073232576077988415535
					glQuad[3,1] = -0.865631202387831743880467897712
					glQuad[4,1] = -0.755404408355003033895101194847
					glQuad[5,1] = -0.617876244402643748446671764049
					glQuad[6,1] = -0.458016777657227386342419442984
					glQuad[7,1] = -0.281603550779258913230460501460
					glQuad[8,1] = -0.950125098376374401853193354250f-01
					glQuad[9,1] = 0.950125098376374401853193354250f-01
					glQuad[10,1] = 0.281603550779258913230460501460
					glQuad[11,1] = 0.458016777657227386342419442984
					glQuad[12,1] = 0.617876244402643748446671764049
					glQuad[13,1] = 0.755404408355003033895101194847
					glQuad[14,1] = 0.865631202387831743880467897712
					glQuad[15,1] = 0.944575023073232576077988415535
					glQuad[16,1] = 0.989400934991649932596154173450
					glQuad[1,2] = 0.271524594117540948517805724560f-01
					glQuad[2,2] = 0.622535239386478928628438369944f-01
					glQuad[3,2] = 0.951585116824927848099251076022f-01
					glQuad[4,2] = 0.124628971255533872052476282192
					glQuad[5,2] = 0.149595988816576732081501730547
					glQuad[6,2] = 0.169156519395002538189312079030
					glQuad[7,2] = 0.182603415044923588866763667969
					glQuad[8,2] = 0.189450610455068496285396723208
					glQuad[9,2] = 0.189450610455068496285396723208
					glQuad[10,2] = 0.182603415044923588866763667969
					glQuad[11,2] = 0.169156519395002538189312079030
					glQuad[12,2] = 0.149595988816576732081501730547
					glQuad[13,2] = 0.124628971255533872052476282192
					glQuad[14,2] = 0.951585116824927848099251076022f-01
					glQuad[15,2] = 0.622535239386478928628438369944f-01
					glQuad[16,2] = 0.271524594117540948517805724560f-01
				
					return glQuad
				end
			end
		end

	else

		glQuad = Array{Float64,2}(undef, ord, 2)
		if ord <= 24

			if ord <= 20

				if ord == 17

					glQuad[1,1] = -0.990575475314417335675434019941
					glQuad[2,1] = -0.950675521768767761222716957896
					glQuad[3,1] = -0.880239153726985902122955694488
					glQuad[4,1] = -0.781514003896801406925230055520
					glQuad[5,1] = -0.657671159216690765850302216643
					glQuad[6,1] = -0.512690537086476967886246568630
					glQuad[7,1] = -0.351231763453876315297185517095
					glQuad[8,1] = -0.178484181495847855850677493654
					glQuad[9,1] = 0.0
					glQuad[10,1] = 0.178484181495847855850677493654
					glQuad[11,1] = 0.351231763453876315297185517095
					glQuad[12,1] = 0.512690537086476967886246568630
					glQuad[13,1] = 0.657671159216690765850302216643
					glQuad[14,1] = 0.781514003896801406925230055520
					glQuad[15,1] = 0.880239153726985902122955694488
					glQuad[16,1] = 0.950675521768767761222716957896
					glQuad[17,1] = 0.990575475314417335675434019941
					glQuad[1,2] = 0.241483028685479319601100262876f-01
					glQuad[2,2] = 0.554595293739872011294401653582f-01
					glQuad[3,2] = 0.850361483171791808835353701911f-01
					glQuad[4,2] = 0.111883847193403971094788385626
					glQuad[5,2] = 0.135136368468525473286319981702
					glQuad[6,2] = 0.154045761076810288081431594802
					glQuad[7,2] = 0.168004102156450044509970663788
					glQuad[8,2] = 0.176562705366992646325270990113
					glQuad[9,2] = 0.179446470356206525458265644262
					glQuad[10,2] = 0.176562705366992646325270990113
					glQuad[11,2] = 0.168004102156450044509970663788
					glQuad[12,2] = 0.154045761076810288081431594802
					glQuad[13,2] = 0.135136368468525473286319981702
					glQuad[14,2] = 0.111883847193403971094788385626
					glQuad[15,2] = 0.850361483171791808835353701911f-01
					glQuad[16,2] = 0.554595293739872011294401653582f-01
					glQuad[17,2] = 0.241483028685479319601100262876f-01
				
					return glQuad

				elseif ord == 18

					glQuad[1,1] = -0.991565168420930946730016004706
					glQuad[2,1] = -0.955823949571397755181195892930
					glQuad[3,1] = -0.892602466497555739206060591127
					glQuad[4,1] = -0.803704958972523115682417455015
					glQuad[5,1] = -0.691687043060353207874891081289
					glQuad[6,1] = -0.559770831073947534607871548525
					glQuad[7,1] = -0.411751161462842646035931793833
					glQuad[8,1] = -0.251886225691505509588972854878
					glQuad[9,1] = -0.847750130417353012422618529358f-01
					glQuad[10,1] = 0.847750130417353012422618529358f-01
					glQuad[11,1] = 0.251886225691505509588972854878
					glQuad[12,1] = 0.411751161462842646035931793833
					glQuad[13,1] = 0.559770831073947534607871548525
					glQuad[14,1] = 0.691687043060353207874891081289
					glQuad[15,1] = 0.803704958972523115682417455015
					glQuad[16,1] = 0.892602466497555739206060591127
					glQuad[17,1] = 0.955823949571397755181195892930
					glQuad[18,1] = 0.991565168420930946730016004706
					glQuad[1,2] = 0.216160135264833103133427102665f-01
					glQuad[2,2] = 0.497145488949697964533349462026f-01
					glQuad[3,2] = 0.764257302548890565291296776166f-01
					glQuad[4,2] = 0.100942044106287165562813984925
					glQuad[5,2] = 0.122555206711478460184519126800
					glQuad[6,2] = 0.140642914670650651204731303752
					glQuad[7,2] = 0.154684675126265244925418003836
					glQuad[8,2] = 0.164276483745832722986053776466
					glQuad[9,2] = 0.169142382963143591840656470135
					glQuad[10,2] = 0.169142382963143591840656470135
					glQuad[11,2] = 0.164276483745832722986053776466
					glQuad[12,2] = 0.154684675126265244925418003836
					glQuad[13,2] = 0.140642914670650651204731303752
					glQuad[14,2] = 0.122555206711478460184519126800
					glQuad[15,2] = 0.100942044106287165562813984925
					glQuad[16,2] = 0.764257302548890565291296776166f-01
					glQuad[17,2] = 0.497145488949697964533349462026f-01
					glQuad[18,2] = 0.216160135264833103133427102665f-01
					
					return glQuad

				elseif ord == 19

					glQuad[1,1] = -0.992406843843584403189017670253
					glQuad[2,1] = -0.960208152134830030852778840688
					glQuad[3,1] = -0.903155903614817901642660928532
					glQuad[4,1] = -0.822714656537142824978922486713
					glQuad[5,1] = -0.720966177335229378617095860824
					glQuad[6,1] = -0.600545304661681023469638164946
					glQuad[7,1] = -0.464570741375960945717267148104
					glQuad[8,1] = -0.316564099963629831990117328850
					glQuad[9,1] = -0.160358645640225375868096115741
					glQuad[10,1] = 0.0
					glQuad[11,1] = 0.160358645640225375868096115741
					glQuad[12,1] = 0.316564099963629831990117328850
					glQuad[13,1] = 0.464570741375960945717267148104
					glQuad[14,1] = 0.600545304661681023469638164946
					glQuad[15,1] = 0.720966177335229378617095860824
					glQuad[16,1] = 0.822714656537142824978922486713
					glQuad[17,1] = 0.903155903614817901642660928532
					glQuad[18,1] = 0.960208152134830030852778840688
					glQuad[19,1] = 0.992406843843584403189017670253
					glQuad[1,2] = 0.194617882297264770363120414644f-01
					glQuad[2,2] = 0.448142267656996003328381574020f-01
					glQuad[3,2] = 0.690445427376412265807082580060f-01
					glQuad[4,2] = 0.914900216224499994644620941238f-01
					glQuad[5,2] = 0.111566645547333994716023901682
					glQuad[6,2] = 0.128753962539336227675515784857
					glQuad[7,2] = 0.142606702173606611775746109442
					glQuad[8,2] = 0.152766042065859666778855400898
					glQuad[9,2] = 0.158968843393954347649956439465
					glQuad[10,2] = 0.161054449848783695979163625321
					glQuad[11,2] = 0.158968843393954347649956439465
					glQuad[12,2] = 0.152766042065859666778855400898
					glQuad[13,2] = 0.142606702173606611775746109442
					glQuad[14,2] = 0.128753962539336227675515784857
					glQuad[15,2] = 0.111566645547333994716023901682
					glQuad[16,2] = 0.914900216224499994644620941238f-01
					glQuad[17,2] = 0.690445427376412265807082580060f-01
					glQuad[18,2] = 0.448142267656996003328381574020f-01
					glQuad[19,2] = 0.194617882297264770363120414644f-01
				
					return glQuad

				else

					glQuad[1,1] = -0.993128599185094924786122388471
					glQuad[2,1] = -0.963971927277913791267666131197
					glQuad[3,1] = -0.912234428251325905867752441203
					glQuad[4,1] = -0.839116971822218823394529061702
					glQuad[5,1] = -0.746331906460150792614305070356
					glQuad[6,1] = -0.636053680726515025452836696226
					glQuad[7,1] = -0.510867001950827098004364050955
					glQuad[8,1] = -0.373706088715419560672548177025
					glQuad[9,1] = -0.227785851141645078080496195369
					glQuad[10,1] = -0.765265211334973337546404093988f-01
					glQuad[11,1] = 0.765265211334973337546404093988f-01
					glQuad[12,1] = 0.227785851141645078080496195369
					glQuad[13,1] = 0.373706088715419560672548177025
					glQuad[14,1] = 0.510867001950827098004364050955
					glQuad[15,1] = 0.636053680726515025452836696226
					glQuad[16,1] = 0.746331906460150792614305070356
					glQuad[17,1] = 0.839116971822218823394529061702
					glQuad[18,1] = 0.912234428251325905867752441203
					glQuad[19,1] = 0.963971927277913791267666131197
					glQuad[20,1] = 0.993128599185094924786122388471
					glQuad[1,2] = 0.176140071391521183118619623519f-01
					glQuad[2,2] = 0.406014298003869413310399522749f-01
					glQuad[3,2] = 0.626720483341090635695065351870f-01
					glQuad[4,2] = 0.832767415767047487247581432220f-01
					glQuad[5,2] = 0.101930119817240435036750135480
					glQuad[6,2] = 0.118194531961518417312377377711
					glQuad[7,2] = 0.131688638449176626898494499748
					glQuad[8,2] = 0.142096109318382051329298325067
					glQuad[9,2] = 0.149172986472603746787828737002
					glQuad[10,2] = 0.152753387130725850698084331955
					glQuad[11,2] = 0.152753387130725850698084331955
					glQuad[12,2] = 0.149172986472603746787828737002
					glQuad[13,2] = 0.142096109318382051329298325067
					glQuad[14,2] = 0.131688638449176626898494499748
					glQuad[15,2] = 0.118194531961518417312377377711
					glQuad[16,2] = 0.101930119817240435036750135480
					glQuad[17,2] = 0.832767415767047487247581432220f-01
					glQuad[18,2] = 0.626720483341090635695065351870f-01
					glQuad[19,2] = 0.406014298003869413310399522749f-01
					glQuad[20,2] = 0.176140071391521183118619623519f-01
				
					return glQuad
				end

			else

				if ord == 21

					glQuad[1,1] = -0.9937521706203896f+00
					glQuad[2,1] = -0.9672268385663063f+00
					glQuad[3,1] = -0.9200993341504008f+00
					glQuad[4,1] = -0.8533633645833173f+00
					glQuad[5,1] = -0.7684399634756779f+00
					glQuad[6,1] = -0.6671388041974123f+00
					glQuad[7,1] = -0.5516188358872198f+00
					glQuad[8,1] = -0.4243421202074388f+00
					glQuad[9,1] = -0.2880213168024011f+00
					glQuad[10,1] = -0.1455618541608951f+00
					glQuad[11,1] = 0.0000000000000000f+00
					glQuad[12,1] = 0.1455618541608951f+00
					glQuad[13,1] = 0.2880213168024011f+00
					glQuad[14,1] = 0.4243421202074388f+00
					glQuad[15,1] = 0.5516188358872198f+00
					glQuad[16,1] = 0.6671388041974123f+00
					glQuad[17,1] = 0.7684399634756779f+00
					glQuad[18,1] = 0.8533633645833173f+00
					glQuad[19,1] = 0.9200993341504008f+00
					glQuad[20,1] = 0.9672268385663063f+00
					glQuad[21,1] = 0.9937521706203896f+00
					glQuad[1,2] = 0.1601722825777420f-01
					glQuad[2,2] = 0.3695378977085242f-01
					glQuad[3,2] = 0.5713442542685715f-01
					glQuad[4,2] = 0.7610011362837928f-01
					glQuad[5,2] = 0.9344442345603393f-01
					glQuad[6,2] = 0.1087972991671484f+00
					glQuad[7,2] = 0.1218314160537285f+00
					glQuad[8,2] = 0.1322689386333373f+00
					glQuad[9,2] = 0.1398873947910731f+00
					glQuad[10,2] = 0.1445244039899700f+00
					glQuad[11,2] = 0.1460811336496904f+00
					glQuad[12,2] = 0.1445244039899700f+00
					glQuad[13,2] = 0.1398873947910731f+00
					glQuad[14,2] = 0.1322689386333373f+00
					glQuad[15,2] = 0.1218314160537285f+00
					glQuad[16,2] = 0.1087972991671484f+00
					glQuad[17,2] = 0.9344442345603393f-01
					glQuad[18,2] = 0.7610011362837928f-01
					glQuad[19,2] = 0.5713442542685715f-01
					glQuad[20,2] = 0.3695378977085242f-01
					glQuad[21,2] = 0.1601722825777420f-01
				
					return glQuad

				elseif ord == 22

					glQuad[1,1] = -0.9942945854823994f+00
					glQuad[2,1] = -0.9700604978354287f+00
					glQuad[3,1] = -0.9269567721871740f+00
					glQuad[4,1] = -0.8658125777203002f+00
					glQuad[5,1] = -0.7878168059792081f+00
					glQuad[6,1] = -0.6944872631866827f+00
					glQuad[7,1] = -0.5876404035069116f+00
					glQuad[8,1] = -0.4693558379867570f+00
					glQuad[9,1] = -0.3419358208920842f+00
					glQuad[10,1] = -0.2078604266882213f+00
					glQuad[11,1] = -0.6973927331972223f-01
					glQuad[12,1] = 0.6973927331972223f-01
					glQuad[13,1] = 0.2078604266882213f+00
					glQuad[14,1] = 0.3419358208920842f+00
					glQuad[15,1] = 0.4693558379867570f+00
					glQuad[16,1] = 0.5876404035069116f+00
					glQuad[17,1] = 0.6944872631866827f+00
					glQuad[18,1] = 0.7878168059792081f+00
					glQuad[19,1] = 0.8658125777203002f+00
					glQuad[20,1] = 0.9269567721871740f+00
					glQuad[21,1] = 0.9700604978354287f+00
					glQuad[22,1] = 0.9942945854823994f+00
					glQuad[1,2] = 0.1462799529827203f-01
					glQuad[2,2] = 0.3377490158481413f-01
					glQuad[3,2] = 0.5229333515268327f-01
					glQuad[4,2] = 0.6979646842452038f-01
					glQuad[5,2] = 0.8594160621706777f-01
					glQuad[6,2] = 0.1004141444428809f+00
					glQuad[7,2] = 0.1129322960805392f+00
					glQuad[8,2] = 0.1232523768105124f+00
					glQuad[9,2] = 0.1311735047870623f+00
					glQuad[10,2] = 0.1365414983460152f+00
					glQuad[11,2] = 0.1392518728556321f+00
					glQuad[12,2] = 0.1392518728556321f+00
					glQuad[13,2] = 0.1365414983460152f+00
					glQuad[14,2] = 0.1311735047870623f+00
					glQuad[15,2] = 0.1232523768105124f+00
					glQuad[16,2] = 0.1129322960805392f+00
					glQuad[17,2] = 0.1004141444428809f+00
					glQuad[18,2] = 0.8594160621706777f-01
					glQuad[19,2] = 0.6979646842452038f-01
					glQuad[20,2] = 0.5229333515268327f-01
					glQuad[21,2] = 0.3377490158481413f-01
					glQuad[22,2] = 0.1462799529827203f-01
				
					return glQuad

				elseif ord == 23

					glQuad[1,1] = -0.9947693349975522f+00
					glQuad[2,1] = -0.9725424712181152f+00
					glQuad[3,1] = -0.9329710868260161f+00
					glQuad[4,1] = -0.8767523582704416f+00
					glQuad[5,1] = -0.8048884016188399f+00
					glQuad[6,1] = -0.7186613631319502f+00
					glQuad[7,1] = -0.6196098757636461f+00
					glQuad[8,1] = -0.5095014778460075f+00
					glQuad[9,1] = -0.3903010380302908f+00
					glQuad[10,1] = -0.2641356809703449f+00
					glQuad[11,1] = -0.1332568242984661f+00
					glQuad[12,1] = 0.0000000000000000f+00
					glQuad[13,1] = 0.1332568242984661f+00
					glQuad[14,1] = 0.2641356809703449f+00
					glQuad[15,1] = 0.3903010380302908f+00
					glQuad[16,1] = 0.5095014778460075f+00
					glQuad[17,1] = 0.6196098757636461f+00
					glQuad[18,1] = 0.7186613631319502f+00
					glQuad[19,1] = 0.8048884016188399f+00
					glQuad[20,1] = 0.8767523582704416f+00
					glQuad[21,1] = 0.9329710868260161f+00
					glQuad[22,1] = 0.9725424712181152f+00
					glQuad[23,1] = 0.9947693349975522f+00
					glQuad[1,2] = 0.1341185948714167f-01
					glQuad[2,2] = 0.3098800585697944f-01
					glQuad[3,2] = 0.4803767173108464f-01
					glQuad[4,2] = 0.6423242140852586f-01
					glQuad[5,2] = 0.7928141177671895f-01
					glQuad[6,2] = 0.9291576606003514f-01
					glQuad[7,2] = 0.1048920914645414f+00
					glQuad[8,2] = 0.1149966402224114f+00
					glQuad[9,2] = 0.1230490843067295f+00
					glQuad[10,2] = 0.1289057221880822f+00
					glQuad[11,2] = 0.1324620394046967f+00
					glQuad[12,2] = 0.1336545721861062f+00
					glQuad[13,2] = 0.1324620394046967f+00
					glQuad[14,2] = 0.1289057221880822f+00
					glQuad[15,2] = 0.1230490843067295f+00
					glQuad[16,2] = 0.1149966402224114f+00
					glQuad[17,2] = 0.1048920914645414f+00
					glQuad[18,2] = 0.9291576606003514f-01
					glQuad[19,2] = 0.7928141177671895f-01
					glQuad[20,2] = 0.6423242140852586f-01
					glQuad[21,2] = 0.4803767173108464f-01
					glQuad[22,2] = 0.3098800585697944f-01
					glQuad[23,2] = 0.1341185948714167f-01
					
					return glQuad

				else

					glQuad[1,1] = -0.9951872199970213f+00
					glQuad[2,1] = -0.9747285559713095f+00
					glQuad[3,1] = -0.9382745520027327f+00
					glQuad[4,1] = -0.8864155270044011f+00
					glQuad[5,1] = -0.8200019859739029f+00
					glQuad[6,1] = -0.7401241915785544f+00
					glQuad[7,1] = -0.6480936519369755f+00
					glQuad[8,1] = -0.5454214713888396f+00
					glQuad[9,1] = -0.4337935076260451f+00
					glQuad[10,1] = -0.3150426796961634f+00
					glQuad[11,1] = -0.1911188674736163f+00
					glQuad[12,1] = -0.6405689286260562f-01
					glQuad[13,1] = 0.6405689286260562f-01
					glQuad[14,1] = 0.1911188674736163f+00
					glQuad[15,1] = 0.3150426796961634f+00
					glQuad[16,1] = 0.4337935076260451f+00
					glQuad[17,1] = 0.5454214713888396f+00
					glQuad[18,1] = 0.6480936519369755f+00
					glQuad[19,1] = 0.7401241915785544f+00
					glQuad[20,1] = 0.8200019859739029f+00
					glQuad[21,1] = 0.8864155270044011f+00
					glQuad[22,1] = 0.9382745520027327f+00
					glQuad[23,1] = 0.9747285559713095f+00
					glQuad[24,1] = 0.9951872199970213f+00
					glQuad[1,2] = 0.1234122979998730f-01
					glQuad[2,2] = 0.2853138862893375f-01
					glQuad[3,2] = 0.4427743881741982f-01
					glQuad[4,2] = 0.5929858491543672f-01
					glQuad[5,2] = 0.7334648141108031f-01
					glQuad[6,2] = 0.8619016153195320f-01
					glQuad[7,2] = 0.9761865210411380f-01
					glQuad[8,2] = 0.1074442701159656f+00
					glQuad[9,2] = 0.1155056680537256f+00
					glQuad[10,2] = 0.1216704729278035f+00
					glQuad[11,2] = 0.1258374563468283f+00
					glQuad[12,2] = 0.1279381953467521f+00
					glQuad[13,2] = 0.1279381953467521f+00
					glQuad[14,2] = 0.1258374563468283f+00
					glQuad[15,2] = 0.1216704729278035f+00
					glQuad[16,2] = 0.1155056680537256f+00
					glQuad[17,2] = 0.1074442701159656f+00
					glQuad[18,2] = 0.9761865210411380f-01
					glQuad[19,2] = 0.8619016153195320f-01
					glQuad[20,2] = 0.7334648141108031f-01
					glQuad[21,2] = 0.5929858491543672f-01
					glQuad[22,2] = 0.4427743881741982f-01
					glQuad[23,2] = 0.2853138862893375f-01
					glQuad[24,2] = 0.1234122979998730f-01
					
					return glQuad
				end
			end

		else

			if ord <= 28

				if ord == 25

					glQuad[1,1] = -0.9955569697904981f+00
					glQuad[2,1] = -0.9766639214595175f+00
					glQuad[3,1] = -0.9429745712289743f+00
					glQuad[4,1] = -0.8949919978782754f+00
					glQuad[5,1] = -0.8334426287608340f+00
					glQuad[6,1] = -0.7592592630373577f+00
					glQuad[7,1] = -0.6735663684734684f+00
					glQuad[8,1] = -0.5776629302412229f+00
					glQuad[9,1] = -0.4730027314457150f+00
					glQuad[10,1] = -0.3611723058093879f+00
					glQuad[11,1] = -0.2438668837209884f+00
					glQuad[12,1] = -0.1228646926107104f+00
					glQuad[13,1] = 0.0000000000000000f+00
					glQuad[14,1] = 0.1228646926107104f+00
					glQuad[15,1] = 0.2438668837209884f+00
					glQuad[16,1] = 0.3611723058093879f+00
					glQuad[17,1] = 0.4730027314457150f+00
					glQuad[18,1] = 0.5776629302412229f+00
					glQuad[19,1] = 0.6735663684734684f+00
					glQuad[20,1] = 0.7592592630373577f+00
					glQuad[21,1] = 0.8334426287608340f+00
					glQuad[22,1] = 0.8949919978782754f+00
					glQuad[23,1] = 0.9429745712289743f+00
					glQuad[24,1] = 0.9766639214595175f+00
					glQuad[25,1] = 0.9955569697904981f+00
					glQuad[1,2] = 0.1139379850102617f-01
					glQuad[2,2] = 0.2635498661503214f-01
					glQuad[3,2] = 0.4093915670130639f-01
					glQuad[4,2] = 0.5490469597583517f-01
					glQuad[5,2] = 0.6803833381235694f-01
					glQuad[6,2] = 0.8014070033500101f-01
					glQuad[7,2] = 0.9102826198296370f-01
					glQuad[8,2] = 0.1005359490670506f+00
					glQuad[9,2] = 0.1085196244742637f+00
					glQuad[10,2] = 0.1148582591457116f+00
					glQuad[11,2] = 0.1194557635357847f+00
					glQuad[12,2] = 0.1222424429903101f+00
					glQuad[13,2] = 0.1231760537267154f+00
					glQuad[14,2] = 0.1222424429903101f+00
					glQuad[15,2] = 0.1194557635357847f+00
					glQuad[16,2] = 0.1148582591457116f+00
					glQuad[17,2] = 0.1085196244742637f+00
					glQuad[18,2] = 0.1005359490670506f+00
					glQuad[19,2] = 0.9102826198296370f-01
					glQuad[20,2] = 0.8014070033500101f-01
					glQuad[21,2] = 0.6803833381235694f-01
					glQuad[22,2] = 0.5490469597583517f-01
					glQuad[23,2] = 0.4093915670130639f-01
					glQuad[24,2] = 0.2635498661503214f-01
					glQuad[25,2] = 0.1139379850102617f-01	
				
					return glQuad

				elseif ord == 26

					glQuad[1,1] = -0.9958857011456169f+00
					glQuad[2,1] = -0.9783854459564710f+00
					glQuad[3,1] = -0.9471590666617142f+00
					glQuad[4,1] = -0.9026378619843071f+00
					glQuad[5,1] = -0.8454459427884981f+00
					glQuad[6,1] = -0.7763859488206789f+00
					glQuad[7,1] = -0.6964272604199573f+00
					glQuad[8,1] = -0.6066922930176181f+00
					glQuad[9,1] = -0.5084407148245057f+00
					glQuad[10,1] = -0.4030517551234863f+00
					glQuad[11,1] = -0.2920048394859569f+00
					glQuad[12,1] = -0.1768588203568902f+00
					glQuad[13,1] = -0.5923009342931320f-01
					glQuad[14,1] = 0.5923009342931320f-01
					glQuad[15,1] = 0.1768588203568902f+00
					glQuad[16,1] = 0.2920048394859569f+00
					glQuad[17,1] = 0.4030517551234863f+00
					glQuad[18,1] = 0.5084407148245057f+00
					glQuad[19,1] = 0.6066922930176181f+00
					glQuad[20,1] = 0.6964272604199573f+00
					glQuad[21,1] = 0.7763859488206789f+00
					glQuad[22,1] = 0.8454459427884981f+00
					glQuad[23,1] = 0.9026378619843071f+00
					glQuad[24,1] = 0.9471590666617142f+00
					glQuad[25,1] = 0.9783854459564710f+00
					glQuad[26,1] = 0.9958857011456169f+00
					glQuad[1,2] = 0.1055137261734304f-01
					glQuad[2,2] = 0.2441785109263173f-01
					glQuad[3,2] = 0.3796238329436282f-01
					glQuad[4,2] = 0.5097582529714782f-01
					glQuad[5,2] = 0.6327404632957484f-01
					glQuad[6,2] = 0.7468414976565967f-01
					glQuad[7,2] = 0.8504589431348521f-01
					glQuad[8,2] = 0.9421380035591416f-01
					glQuad[9,2] = 0.1020591610944255f+00
					glQuad[10,2] = 0.1084718405285765f+00
					glQuad[11,2] = 0.1133618165463197f+00
					glQuad[12,2] = 0.1166604434852967f+00
					glQuad[13,2] = 0.1183214152792622f+00
					glQuad[14,2] = 0.1183214152792622f+00
					glQuad[15,2] = 0.1166604434852967f+00
					glQuad[16,2] = 0.1133618165463197f+00
					glQuad[17,2] = 0.1084718405285765f+00
					glQuad[18,2] = 0.1020591610944255f+00
					glQuad[19,2] = 0.9421380035591416f-01
					glQuad[20,2] = 0.8504589431348521f-01
					glQuad[21,2] = 0.7468414976565967f-01
					glQuad[22,2] = 0.6327404632957484f-01
					glQuad[23,2] = 0.5097582529714782f-01
					glQuad[24,2] = 0.3796238329436282f-01
					glQuad[25,2] = 0.2441785109263173f-01
					glQuad[26,2] = 0.1055137261734304f-01
				
					return glQuad

				elseif ord == 27

					glQuad[1,1] = -0.9961792628889886f+00
					glQuad[2,1] = -0.9799234759615012f+00
					glQuad[3,1] = -0.9509005578147051f+00
					glQuad[4,1] = -0.9094823206774911f+00
					glQuad[5,1] = -0.8562079080182945f+00
					glQuad[6,1] = -0.7917716390705082f+00
					glQuad[7,1] = -0.7170134737394237f+00
					glQuad[8,1] = -0.6329079719464952f+00
					glQuad[9,1] = -0.5405515645794569f+00
					glQuad[10,1] = -0.4411482517500269f+00
					glQuad[11,1] = -0.3359939036385089f+00
					glQuad[12,1] = -0.2264593654395369f+00
					glQuad[13,1] = -0.1139725856095300f+00
					glQuad[14,1] = 0.0000000000000000f+00
					glQuad[15,1] = 0.1139725856095300f+00
					glQuad[16,1] = 0.2264593654395369f+00
					glQuad[17,1] = 0.3359939036385089f+00
					glQuad[18,1] = 0.4411482517500269f+00
					glQuad[19,1] = 0.5405515645794569f+00
					glQuad[20,1] = 0.6329079719464952f+00
					glQuad[21,1] = 0.7170134737394237f+00
					glQuad[22,1] = 0.7917716390705082f+00
					glQuad[23,1] = 0.8562079080182945f+00
					glQuad[24,1] = 0.9094823206774911f+00
					glQuad[25,1] = 0.9509005578147051f+00
					glQuad[26,1] = 0.9799234759615012f+00
					glQuad[27,1] = 0.9961792628889886f+00
					glQuad[1,2] = 0.9798996051294232f-02
					glQuad[2,2] = 0.2268623159618062f-01
					glQuad[3,2] = 0.3529705375741969f-01
					glQuad[4,2] = 0.4744941252061504f-01
					glQuad[5,2] = 0.5898353685983366f-01
					glQuad[6,2] = 0.6974882376624561f-01
					glQuad[7,2] = 0.7960486777305781f-01
					glQuad[8,2] = 0.8842315854375689f-01
					glQuad[9,2] = 0.9608872737002842f-01
					glQuad[10,2] = 0.1025016378177459f+00
					glQuad[11,2] = 0.1075782857885332f+00
					glQuad[12,2] = 0.1112524883568452f+00
					glQuad[13,2] = 0.1134763461089651f+00
					glQuad[14,2] = 0.1142208673789570f+00
					glQuad[15,2] = 0.1134763461089651f+00
					glQuad[16,2] = 0.1112524883568452f+00
					glQuad[17,2] = 0.1075782857885332f+00
					glQuad[18,2] = 0.1025016378177459f+00
					glQuad[19,2] = 0.9608872737002842f-01
					glQuad[20,2] = 0.8842315854375689f-01
					glQuad[21,2] = 0.7960486777305781f-01
					glQuad[22,2] = 0.6974882376624561f-01
					glQuad[23,2] = 0.5898353685983366f-01
					glQuad[24,2] = 0.4744941252061504f-01
					glQuad[25,2] = 0.3529705375741969f-01
					glQuad[26,2] = 0.2268623159618062f-01
					glQuad[27,2] = 0.9798996051294232f-02	
					
					return glQuad

				else

					glQuad[1,1] = -0.9964424975739544f+00
					glQuad[2,1] = -0.9813031653708728f+00
					glQuad[3,1] = -0.9542592806289382f+00
					glQuad[4,1] = -0.9156330263921321f+00
					glQuad[5,1] = -0.8658925225743951f+00
					glQuad[6,1] = -0.8056413709171791f+00
					glQuad[7,1] = -0.7356108780136318f+00
					glQuad[8,1] = -0.6566510940388650f+00
					glQuad[9,1] = -0.5697204718114017f+00
					glQuad[10,1] = -0.4758742249551183f+00
					glQuad[11,1] = -0.3762515160890787f+00
					glQuad[12,1] = -0.2720616276351780f+00
					glQuad[13,1] = -0.1645692821333808f+00
					glQuad[14,1] = -0.5507928988403427f-01
					glQuad[15,1] = 0.5507928988403427f-01
					glQuad[16,1] = 0.1645692821333808f+00
					glQuad[17,1] = 0.2720616276351780f+00
					glQuad[18,1] = 0.3762515160890787f+00
					glQuad[19,1] = 0.4758742249551183f+00
					glQuad[20,1] = 0.5697204718114017f+00
					glQuad[21,1] = 0.6566510940388650f+00
					glQuad[22,1] = 0.7356108780136318f+00
					glQuad[23,1] = 0.8056413709171791f+00
					glQuad[24,1] = 0.8658925225743951f+00
					glQuad[25,1] = 0.9156330263921321f+00
					glQuad[26,1] = 0.9542592806289382f+00
					glQuad[27,1] = 0.9813031653708728f+00
					glQuad[28,1] = 0.9964424975739544f+00
					glQuad[1,2] = 0.9124282593094672f-02
					glQuad[2,2] = 0.2113211259277118f-01
					glQuad[3,2] = 0.3290142778230441f-01
					glQuad[4,2] = 0.4427293475900429f-01
					glQuad[5,2] = 0.5510734567571667f-01
					glQuad[6,2] = 0.6527292396699959f-01
					glQuad[7,2] = 0.7464621423456877f-01
					glQuad[8,2] = 0.8311341722890127f-01
					glQuad[9,2] = 0.9057174439303289f-01
					glQuad[10,2] = 0.9693065799792999f-01
					glQuad[11,2] = 0.1021129675780608f+00
					glQuad[12,2] = 0.1060557659228464f+00
					glQuad[13,2] = 0.1087111922582942f+00
					glQuad[14,2] = 0.1100470130164752f+00
					glQuad[15,2] = 0.1100470130164752f+00
					glQuad[16,2] = 0.1087111922582942f+00
					glQuad[17,2] = 0.1060557659228464f+00
					glQuad[18,2] = 0.1021129675780608f+00
					glQuad[19,2] = 0.9693065799792999f-01
					glQuad[20,2] = 0.9057174439303289f-01
					glQuad[21,2] = 0.8311341722890127f-01
					glQuad[22,2] = 0.7464621423456877f-01
					glQuad[23,2] = 0.6527292396699959f-01
					glQuad[24,2] = 0.5510734567571667f-01
					glQuad[25,2] = 0.4427293475900429f-01
					glQuad[26,2] = 0.3290142778230441f-01
					glQuad[27,2] = 0.2113211259277118f-01
					glQuad[28,2] = 0.9124282593094672f-02				
					
					return glQuad
				end

			else

				if ord == 29

					glQuad[1,1] = -0.9966794422605966f+00
					glQuad[2,1] = -0.9825455052614132f+00
					glQuad[3,1] = -0.9572855957780877f+00
					glQuad[4,1] = -0.9211802329530588f+00
					glQuad[5,1] = -0.8746378049201028f+00
					glQuad[6,1] = -0.8181854876152524f+00
					glQuad[7,1] = -0.7524628517344771f+00
					glQuad[8,1] = -0.6782145376026865f+00
					glQuad[9,1] = -0.5962817971382278f+00
					glQuad[10,1] = -0.5075929551242276f+00
					glQuad[11,1] = -0.4131528881740087f+00
					glQuad[12,1] = -0.3140316378676399f+00
					glQuad[13,1] = -0.2113522861660011f+00
					glQuad[14,1] = -0.1062782301326792f+00
					glQuad[15,1] = 0.0000000000000000f+00
					glQuad[16,1] = 0.1062782301326792f+00
					glQuad[17,1] = 0.2113522861660011f+00
					glQuad[18,1] = 0.3140316378676399f+00
					glQuad[19,1] = 0.4131528881740087f+00
					glQuad[20,1] = 0.5075929551242276f+00
					glQuad[21,1] = 0.5962817971382278f+00
					glQuad[22,1] = 0.6782145376026865f+00
					glQuad[23,1] = 0.7524628517344771f+00
					glQuad[24,1] = 0.8181854876152524f+00
					glQuad[25,1] = 0.8746378049201028f+00
					glQuad[26,1] = 0.9211802329530588f+00
					glQuad[27,1] = 0.9572855957780877f+00
					glQuad[28,1] = 0.9825455052614132f+00
					glQuad[29,1] = 0.9966794422605966f+00
					glQuad[1,2] = 0.8516903878746365f-02
					glQuad[2,2] = 0.1973208505612276f-01
					glQuad[3,2] = 0.3074049220209360f-01
					glQuad[4,2] = 0.4140206251868281f-01
					glQuad[5,2] = 0.5159482690249799f-01
					glQuad[6,2] = 0.6120309065707916f-01
					glQuad[7,2] = 0.7011793325505125f-01
					glQuad[8,2] = 0.7823832713576385f-01
					glQuad[9,2] = 0.8547225736617248f-01
					glQuad[10,2] = 0.9173775713925882f-01
					glQuad[11,2] = 0.9696383409440862f-01
					glQuad[12,2] = 0.1010912737599150f+00
					glQuad[13,2] = 0.1040733100777293f+00
					glQuad[14,2] = 0.1058761550973210f+00
					glQuad[15,2] = 0.1064793817183143f+00
					glQuad[16,2] = 0.1058761550973210f+00
					glQuad[17,2] = 0.1040733100777293f+00
					glQuad[18,2] = 0.1010912737599150f+00
					glQuad[19,2] = 0.9696383409440862f-01
					glQuad[20,2] = 0.9173775713925882f-01
					glQuad[21,2] = 0.8547225736617248f-01
					glQuad[22,2] = 0.7823832713576385f-01
					glQuad[23,2] = 0.7011793325505125f-01
					glQuad[24,2] = 0.6120309065707916f-01
					glQuad[25,2] = 0.5159482690249799f-01
					glQuad[26,2] = 0.4140206251868281f-01
					glQuad[27,2] = 0.3074049220209360f-01
					glQuad[28,2] = 0.1973208505612276f-01
					glQuad[29,2] = 0.8516903878746365f-02	
					return glQuad

				elseif ord == 30

					glQuad[1,1] = -0.9968934840746495f+00
					glQuad[2,1] = -0.9836681232797472f+00
					glQuad[3,1] = -0.9600218649683075f+00
					glQuad[4,1] = -0.9262000474292743f+00
					glQuad[5,1] = -0.8825605357920526f+00
					glQuad[6,1] = -0.8295657623827684f+00
					glQuad[7,1] = -0.7677774321048262f+00
					glQuad[8,1] = -0.6978504947933158f+00
					glQuad[9,1] = -0.6205261829892429f+00
					glQuad[10,1] = -0.5366241481420199f+00
					glQuad[11,1] = -0.4470337695380892f+00
					glQuad[12,1] = -0.3527047255308781f+00
					glQuad[13,1] = -0.2546369261678899f+00
					glQuad[14,1] = -0.1538699136085835f+00
					glQuad[15,1] = -0.5147184255531770f-01
					glQuad[16,1] = 0.5147184255531770f-01
					glQuad[17,1] = 0.1538699136085835f+00
					glQuad[18,1] = 0.2546369261678899f+00
					glQuad[19,1] = 0.3527047255308781f+00
					glQuad[20,1] = 0.4470337695380892f+00
					glQuad[21,1] = 0.5366241481420199f+00
					glQuad[22,1] = 0.6205261829892429f+00
					glQuad[23,1] = 0.6978504947933158f+00
					glQuad[24,1] = 0.7677774321048262f+00
					glQuad[25,1] = 0.8295657623827684f+00
					glQuad[26,1] = 0.8825605357920526f+00
					glQuad[27,1] = 0.9262000474292743f+00
					glQuad[28,1] = 0.9600218649683075f+00
					glQuad[29,1] = 0.9836681232797472f+00
					glQuad[30,1] = 0.9968934840746495f+00
					glQuad[1,2] = 0.7968192496166648f-02
					glQuad[2,2] = 0.1846646831109099f-01
					glQuad[3,2] = 0.2878470788332330f-01
					glQuad[4,2] = 0.3879919256962704f-01
					glQuad[5,2] = 0.4840267283059405f-01
					glQuad[6,2] = 0.5749315621761905f-01
					glQuad[7,2] = 0.6597422988218052f-01
					glQuad[8,2] = 0.7375597473770516f-01
					glQuad[9,2] = 0.8075589522942023f-01
					glQuad[10,2] = 0.8689978720108314f-01
					glQuad[11,2] = 0.9212252223778619f-01
					glQuad[12,2] = 0.9636873717464424f-01
					glQuad[13,2] = 0.9959342058679524f-01
					glQuad[14,2] = 0.1017623897484056f+00
					glQuad[15,2] = 0.1028526528935587f+00
					glQuad[16,2] = 0.1028526528935587f+00
					glQuad[17,2] = 0.1017623897484056f+00
					glQuad[18,2] = 0.9959342058679524f-01
					glQuad[19,2] = 0.9636873717464424f-01
					glQuad[20,2] = 0.9212252223778619f-01
					glQuad[21,2] = 0.8689978720108314f-01
					glQuad[22,2] = 0.8075589522942023f-01
					glQuad[23,2] = 0.7375597473770516f-01
					glQuad[24,2] = 0.6597422988218052f-01
					glQuad[25,2] = 0.5749315621761905f-01
					glQuad[26,2] = 0.4840267283059405f-01
					glQuad[27,2] = 0.3879919256962704f-01
					glQuad[28,2] = 0.2878470788332330f-01
					glQuad[29,2] = 0.1846646831109099f-01
					glQuad[30,2] = 0.7968192496166648f-02
					return glQuad

				elseif ord == 31

					glQuad[1,1] = -0.9970874818194770f+00
					glQuad[2,1] = -0.9846859096651525f+00
					glQuad[3,1] = -0.9625039250929497f+00
					glQuad[4,1] = -0.9307569978966481f+00
					glQuad[5,1] = -0.8897600299482711f+00
					glQuad[6,1] = -0.8399203201462674f+00
					glQuad[7,1] = -0.7817331484166250f+00
					glQuad[8,1] = -0.7157767845868533f+00
					glQuad[9,1] = -0.6427067229242603f+00
					glQuad[10,1] = -0.5632491614071492f+00
					glQuad[11,1] = -0.4781937820449025f+00
					glQuad[12,1] = -0.3883859016082329f+00
					glQuad[13,1] = -0.2947180699817016f+00
					glQuad[14,1] = -0.1981211993355706f+00
					glQuad[15,1] = -0.9955531215234151f-01
					glQuad[16,1] = 0.0000000000000000f+00
					glQuad[17,1] = 0.9955531215234151f-01
					glQuad[18,1] = 0.1981211993355706f+00
					glQuad[19,1] = 0.2947180699817016f+00
					glQuad[20,1] = 0.3883859016082329f+00
					glQuad[21,1] = 0.4781937820449025f+00
					glQuad[22,1] = 0.5632491614071492f+00
					glQuad[23,1] = 0.6427067229242603f+00
					glQuad[24,1] = 0.7157767845868533f+00
					glQuad[25,1] = 0.7817331484166250f+00
					glQuad[26,1] = 0.8399203201462674f+00
					glQuad[27,1] = 0.8897600299482711f+00
					glQuad[28,1] = 0.9307569978966481f+00
					glQuad[29,1] = 0.9625039250929497f+00
					glQuad[30,1] = 0.9846859096651525f+00
					glQuad[31,1] = 0.9970874818194770f+00
					glQuad[1,2] = 0.7470831579248783f-02
					glQuad[2,2] = 0.1731862079031058f-01
					glQuad[3,2] = 0.2700901918497941f-01
					glQuad[4,2] = 0.3643227391238550f-01
					glQuad[5,2] = 0.4549370752720110f-01
					glQuad[6,2] = 0.5410308242491679f-01
					glQuad[7,2] = 0.6217478656102854f-01
					glQuad[8,2] = 0.6962858323541037f-01
					glQuad[9,2] = 0.7639038659877659f-01
					glQuad[10,2] = 0.8239299176158929f-01
					glQuad[11,2] = 0.8757674060847785f-01
					glQuad[12,2] = 0.9189011389364142f-01
					glQuad[13,2] = 0.9529024291231955f-01
					glQuad[14,2] = 0.9774333538632875f-01
					glQuad[15,2] = 0.9922501122667234f-01
					glQuad[16,2] = 0.9972054479342644f-01
					glQuad[17,2] = 0.9922501122667234f-01
					glQuad[18,2] = 0.9774333538632875f-01
					glQuad[19,2] = 0.9529024291231955f-01
					glQuad[20,2] = 0.9189011389364142f-01
					glQuad[21,2] = 0.8757674060847785f-01
					glQuad[22,2] = 0.8239299176158929f-01
					glQuad[23,2] = 0.7639038659877659f-01
					glQuad[24,2] = 0.6962858323541037f-01
					glQuad[25,2] = 0.6217478656102854f-01
					glQuad[26,2] = 0.5410308242491679f-01
					glQuad[27,2] = 0.4549370752720110f-01
					glQuad[28,2] = 0.3643227391238550f-01
					glQuad[29,2] = 0.2700901918497941f-01
					glQuad[30,2] = 0.1731862079031058f-01
					glQuad[31,2] = 0.7470831579248783f-02
					return glQuad

				else

					glQuad[1,1] = -0.997263861849481563544981128665
					glQuad[2,1] = -0.985611511545268335400175044631
					glQuad[3,1] = -0.964762255587506430773811928118
					glQuad[4,1] = -0.934906075937739689170919134835
					glQuad[5,1] = -0.896321155766052123965307243719
					glQuad[6,1] = -0.849367613732569970133693004968
					glQuad[7,1] = -0.794483795967942406963097298970
					glQuad[8,1] = -0.732182118740289680387426665091
					glQuad[9,1] = -0.663044266930215200975115168663
					glQuad[10,1] = -0.587715757240762329040745476402
					glQuad[11,1] = -0.506899908932229390023747474378
					glQuad[12,1] = -0.421351276130635345364119436172
					glQuad[13,1] = -0.331868602282127649779916805730
					glQuad[14,1] = -0.239287362252137074544603209166
					glQuad[15,1] = -0.144471961582796493485186373599
					glQuad[16,1] = -0.483076656877383162348125704405f-01
					glQuad[17,1] = 0.483076656877383162348125704405f-01
					glQuad[18,1] = 0.144471961582796493485186373599
					glQuad[19,1] = 0.239287362252137074544603209166
					glQuad[20,1] = 0.331868602282127649779916805730
					glQuad[21,1] = 0.421351276130635345364119436172
					glQuad[22,1] = 0.506899908932229390023747474378
					glQuad[23,1] = 0.587715757240762329040745476402
					glQuad[24,1] = 0.663044266930215200975115168663
					glQuad[25,1] = 0.732182118740289680387426665091
					glQuad[26,1] = 0.794483795967942406963097298970
					glQuad[27,1] = 0.849367613732569970133693004968
					glQuad[28,1] = 0.896321155766052123965307243719
					glQuad[29,1] = 0.934906075937739689170919134835
					glQuad[30,1] = 0.964762255587506430773811928118
					glQuad[31,1] = 0.985611511545268335400175044631
					glQuad[32,1] = 0.997263861849481563544981128665
					glQuad[1,2] = 0.701861000947009660040706373885f-02
					glQuad[2,2] = 0.162743947309056706051705622064f-01
					glQuad[3,2] = 0.253920653092620594557525897892f-01
					glQuad[4,2] = 0.342738629130214331026877322524f-01
					glQuad[5,2] = 0.428358980222266806568786466061f-01
					glQuad[6,2] = 0.509980592623761761961632446895f-01
					glQuad[7,2] = 0.586840934785355471452836373002f-01
					glQuad[8,2] = 0.658222227763618468376500637069f-01
					glQuad[9,2] = 0.723457941088485062253993564785f-01
					glQuad[10,2] = 0.781938957870703064717409188283f-01
					glQuad[11,2] = 0.833119242269467552221990746043f-01
					glQuad[12,2] = 0.876520930044038111427714627518f-01
					glQuad[13,2] = 0.911738786957638847128685771116f-01
					glQuad[14,2] = 0.938443990808045656391802376681f-01
					glQuad[15,2] = 0.956387200792748594190820022041f-01
					glQuad[16,2] = 0.965400885147278005667648300636f-01
					glQuad[17,2] = 0.965400885147278005667648300636f-01
					glQuad[18,2] = 0.956387200792748594190820022041f-01
					glQuad[19,2] = 0.938443990808045656391802376681f-01
					glQuad[20,2] = 0.911738786957638847128685771116f-01
					glQuad[21,2] = 0.876520930044038111427714627518f-01
					glQuad[22,2] = 0.833119242269467552221990746043f-01
					glQuad[23,2] = 0.781938957870703064717409188283f-01
					glQuad[24,2] = 0.723457941088485062253993564785f-01
					glQuad[25,2] = 0.658222227763618468376500637069f-01
					glQuad[26,2] = 0.586840934785355471452836373002f-01
					glQuad[27,2] = 0.509980592623761761961632446895f-01
					glQuad[28,2] = 0.428358980222266806568786466061f-01
					glQuad[29,2] = 0.342738629130214331026877322524f-01
					glQuad[30,2] = 0.253920653092620594557525897892f-01
					glQuad[31,2] = 0.162743947309056706051705622064f-01
					glQuad[32,2] = 0.701861000947009660040706373885f-02
					
					return glQuad
				end
			end
		end
	end
end

# Self panels.
function weakSInt(rPoints::Array{Float64,2}, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::ComplexF64

	intVals = zeros(ComplexF64, 3, 1)
	ψA = 0.0 
	ψB = 0.0
	ηA = 0.0
	ηB = 0.0
	θ = 0.0

	for kk in 1:3
		
		for n1 in 1:8

			(ψA, ψB) = ψlimS(n1)
			intVals[2] = 0.0 + 0.0im

			for n2 in 1:assemblyOpts.ordGLIntNear
				
				θ = θf(ψA, ψB, glQuad1[n2,1])
				(ηA, ηB) = ηlimS(n1, θ)
				intVals[3] = 0.0 + 0.0im
				
				for n3 in 1:assemblyOpts.ordGLIntNear
 
					intVals[3] += glQuad1[n3,2] * 
					nS(kk, n1, θ, θf(ηA, ηB, glQuad1[n3,1]), rPoints, glQuad1, 
						assemblyOpts)
				end
				intVals[2] += (glQuad1[n2,2] * (ηB - ηA) * sin(θ) * intVals[3] 
				/ 2.0)
			end
			intVals[1] += (ψB - ψA) * intVals[2] / 2.0
		end
	end	
	return equiJacobianS(rPoints) * intVals[1]
end
# Edge panels.
function weakEInt(rPoints::Array{Float64,2}, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::ComplexF64

	intVals = zeros(ComplexF64, 3, 1)
	ψA = 0.0 
	ψB = 0.0
	ηA = 0.0
	ηB = 0.0
	θA = 0.0
	θB = 0.0
	
	for n1 in 1:6

		(ψA, ψB) = ψlimE(n1)
		intVals[2] = 0.0 + 0.0im

		for n2 in 1:assemblyOpts.ordGLIntNear

			θB = θf(ψA, ψB, glQuad1[n2, 1])
			(ηA, ηB) = ηlimE(n1, θB)
			intVals[3] = 0.0 + 0.0im

			for n3 in 1:assemblyOpts.ordGLIntNear

				θA = θf(ηA, ηB, glQuad1[n3, 1])
				intVals[3] += glQuad1[n3, 2] * cos(θA) *  
				(nE(n1, 1, θA, θB, rPoints, glQuad1, assemblyOpts) + 
				nE(n1, -1, θA, θB, rPoints, glQuad1, assemblyOpts)) 
			end
			intVals[2] += glQuad1[n2, 2] * (ηB - ηA) * intVals[3] / 2.0
		end
		intVals[1] += (ψB - ψA) * intVals[2] / 2.0
	end
	
	return equiJacobianEV(rPoints) * intVals[1]
end
# Vertex panels.
function weakVInt(rPoints::Array{Float64,2}, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::ComplexF64

	xPoints = Array{Float64,2}(undef, 3, 2)
	intVals = zeros(ComplexF64, 5, 1)
	θ1 = 0.0 
	θ2 = 0.0 
	θ3 = 0.0 
	θ4 = 0.0 
	θ5 = 0.0 
	θ6 = 0.0 
	L1 = 0.0
	L2 = 0.0
	
	for n1 in 1:assemblyOpts.ordGLIntNear

		θ1 = θf(0.0, π / 3.0, glQuad1[n1,1])
		L1 = 2.0 * sqrt(3.0) / (sin(θ1) + sqrt(3.0) * cos(θ1)) 
		intVals[2] = 0.0 + 0.0im

		for n2 in 1:assemblyOpts.ordGLIntNear

			θ2 = θf(0.0, π / 3.0, glQuad1[n2,1])
			L2 = 2.0 * sqrt(3.0) / (sin(θ2) + sqrt(3.0) * cos(θ2))
			intVals[3] = 0.0 + 0.0im
			
			for n3 in 1:assemblyOpts.ordGLIntNear

				θ3 = θf(0.0, atan(L2 / L1), glQuad1[n3,1])
				intVals[4] = 0.0 + 0.0im

				for n4 in 1:assemblyOpts.ordGLIntNear

					θ4 = θf(0.0, L1 / cos(θ3), glQuad1[n4,1])
					simplexV!(xPoints, θ4, θ3, θ2, θ1)
					intVals[4] += glQuad1[n4,2] * (θ4^3) * 
					kernelEV(rPoints, xPoints, assemblyOpts.freqPhase)
				end
				intVals[4] *= L1 * sin(θ3) * cos(θ3) / (2.0 * cos(θ3))
				θ5 = θf(atan(L2 / L1), π / 2.0, glQuad1[n3,1])
				intVals[5] = 0.0 + 0.0im

				for n5 in 1:assemblyOpts.ordGLIntNear

					θ6 = θf(0.0, L2 / sin(θ5), glQuad1[n5,1])
					simplexV!(xPoints, θ6, θ5, θ2, θ1)
					intVals[5] += glQuad1[n5,2] * (θ6^3) *  
					kernelEV(rPoints, xPoints, assemblyOpts.freqPhase)
				end
				intVals[5] *= L2 * sin(θ5) * cos(θ5) / (2.0 * sin(θ5))
				intVals[3] += glQuad1[n3, 2] * (atan(L2/L1) * 
					(intVals[4] - intVals[5]) + π * intVals[5] / 2.0) / 2.0
			end
			intVals[2] += glQuad1[n2, 2] * intVals[3]
		end
		intVals[2] *= π / 6.0
		intVals[1] += glQuad1[n1, 2] * intVals[2]
	end
	return equiJacobianEV(rPoints) * π * intVals[1] / 6.0
end

function ψlimS(case::Int64)::Tuple{Float64,Float64}
	
	if case == 1 || case == 5 || case == 6
		
		return (0.0, π / 3.0)

	elseif case == 2 || case == 7
		
		return (π / 3.0, 2.0 * π / 3.0)

	elseif case == 3 || case == 4 || case == 8
		
		return (2.0 * π / 3.0, π)

	else

		error("Unrecognized case.")
	end
end

function ψlimE(case::Int64)::Tuple{Float64, Float64}
	
	if case == 1
		
		return (0.0, π / 3.0)

	elseif case == 2 || case == 3
		
		return (π / 3.0, π / 2.0)

	elseif case == 4 || case == 6
		
		return (π / 2.0, π)

	elseif case == 5
		
		return (0.0, π / 2.0)

	else
		
		error("Unrecognized case.")
	end
end

function ηlimS(case::Int64, θ::Float64)::Tuple{Float64,Float64}
	
	if case == 1 || case == 2
		
		return (0.0, 1.0)

	elseif case == 3
		
		return ((1 - tan(π - θ) / sqrt(3.0)) / (1 + tan(π - θ) / sqrt(3.0)), 
			1.0)

	elseif case == 4
		
		return (0.0, 
			(1 - tan(π - θ) / sqrt(3.0)) / (1 + tan(π - θ) / sqrt(3.0)))

	elseif case == 5
		
		return ((tan(θ) / sqrt(3.0) - 1.0) / (1 + tan(θ) / sqrt(3.0)), 0.0)

	elseif case == 6
		
		return (-1.0, (tan(θ) / sqrt(3.0) - 1.0) / (1.0 + tan(θ) / sqrt(3.0)))

	elseif case == 7 || case == 8
		
		return (-1.0, 0.0)

	else

		error("Unrecognized case.")
	end
end

function ηlimE(case::Int64, θ::Float64)::Tuple{Float64, Float64}
	
	if case == 1
		
		return (0.0, atan(sin(θ) + sqrt(3.0) * cos(θ)))

	elseif case == 2
		
		return (atan(sin(θ) - sqrt(3.0) * cos(θ)), 
			atan(sin(θ) + sqrt(3.0) * cos(θ)))

	elseif case == 3 || case == 4
		
		return (0.0, atan(sin(θ) - sqrt(3.0) * cos(θ)))

	elseif case == 5
		
		return (atan(sin(θ) + sqrt(3.0) * cos(θ)), π / 2.0)

	elseif case == 6
		
		return (atan(sin(θ) - sqrt(3.0) * cos(θ)), π / 2.0)

	else

		error("Unrecognized case.")
	end
end	

function nS(dir::Int64, case::Int64, θ1::Float64, θ2::Float64, 
	rPoints::Array{Float64,2}, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::ComplexF64

	IS = 0.0 + 0.0im

	if case == 1 || case == 5
		
		for n in 1:assemblyOpts.ordGLIntNear

			IS += glQuad1[n,2] * aS(rPoints, θ1, θ2, 
				θf(0.0, (1.0 - θ2) / cos(θ1), glQuad1[n,1]), dir, glQuad1, 
				assemblyOpts)
		end
		return (1.0 - θ2) / (2.0 * cos(θ1)) * IS

	elseif case == 2 || case == 3

		for n in 1:assemblyOpts.ordGLIntNear

			IS += glQuad1[n,2] * aS(rPoints, θ1, θ2, 
				θf(0.0, sqrt(3.0) * (1.0 - θ2) / sin(θ1), glQuad1[n,1]), dir, 
				glQuad1, assemblyOpts)
		end
		return sqrt(3.0) * (1.0 - θ2) / (2.0 * sin(θ1)) * IS

	elseif case == 6 || case == 7
		
		for n in 1:assemblyOpts.ordGLIntNear

			IS += glQuad1[n,2] * aS(rPoints, θ1, θ2, 
				θf(0.0, sqrt(3.0) * (1.0 + θ2) / sin(θ1), glQuad1[n,1]), dir, 
				glQuad1, assemblyOpts)
		end
		return sqrt(3.0) * (1.0 + θ2) / (2.0 * sin(θ1)) * IS

	elseif case == 4 || case == 8

		for n in 1:assemblyOpts.ordGLIntNear

			IS += glQuad1[n,2] * aS(rPoints, θ1, θ2, 
				θf(0.0, -(1.0 + θ2) / cos(θ1), glQuad1[n,1]), dir, glQuad1, 
				assemblyOpts)
		end
		return -(1.0 + θ2) / (2.0 * cos(θ1)) * IS

	else
		
		error("Unrecognized case.")
	end
end

function nE(case1::Int64, case2::Int64, θ2::Float64, θ1::Float64, 
	rPoints::Array{Float64,2}, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::ComplexF64

	γ = 0.0 
	intVal1 = 0.0 + 0.0im 
	intVal2 = 0.0 + 0.0im

	if case1 == 1 || case1 == 2 
		
		γ = (sin(θ1) + sqrt(3.0) * cos(θ1) - tan(θ2)) / 
		(sin(θ1) + sqrt(3.0) * cos(θ1) + tan(θ2))

		for n in 1 : assemblyOpts.ordGLIntNear
			
			intVal1 += glQuad1[n, 2] * intNE(n, 1, γ, θ2, θ1, rPoints, glQuad1, 
				case2, assemblyOpts)
			intVal2 += glQuad1[n, 2] * intNE(n, 2, γ, θ2, θ1, rPoints, glQuad1, 
				case2, assemblyOpts)
		end
		return intVal2 / 2.0 + γ * (intVal1-intVal2) / 2.0
	
	elseif case1 == 3
		
		γ = sqrt(3.0) / tan(θ1)

		for n in 1 : assemblyOpts.ordGLIntNear

			intVal1 += glQuad1[n, 2] * intNE(n, 1, γ, θ2, θ1, rPoints, glQuad1, 
				case2, assemblyOpts)
			intVal2 += glQuad1[n, 2] * intNE(n, 3, γ, θ2, θ1, rPoints, glQuad1, 
				case2, assemblyOpts)
		end
		return intVal2 / 2.0 + γ * (intVal1 - intVal2) / 2.0

	elseif case1 == 4

		for n in 1 : assemblyOpts.ordGLIntNear
			
			intVal1 += glQuad1[n, 2] * intNE(n, 4, 1.0, θ2, θ1, rPoints, 
				glQuad1, case2, assemblyOpts)
		end
		
		return intVal1 / 2.0
	
	elseif case1 == 5 || case1 == 6
		
		for n in 1 : assemblyOpts.ordGLIntNear
			
			intVal1 += glQuad1[n, 2] * intNE(n, 5, 1.0, θ2, θ1, rPoints, 
				glQuad1, case2, assemblyOpts)
		end
		return intVal1 / 2.0

	else

		error("Unrecognized case.")
	end
end

function intNE(n::Int64, case1::Int64, γ::Float64, θ2::Float64, θ1::Float64, 
	rPoints::Array{Float64,2}, glQuad1::Array{Float64,2}, case2::Int64,
	assemblyOpts::MaxGAssemblyOpts)::ComplexF64
	
	if case1 == 1
		
		η = θf(0.0, γ, glQuad1[n,1])
		λ = sqrt(3.0) * (1 + η)  /  (cos(θ2) * (sin(θ1) + sqrt(3.0) * cos(θ1)))
	
	elseif case1 == 2
	
		η = θf(γ, 1.0, glQuad1[n,1])
		λ = sqrt(3.0) * (1.0 - abs(η)) / sin(θ2)
	
	elseif case1 == 3
	
		η = θf(γ, 1.0, glQuad1[n,1])
		λ = sqrt(3.0) * (1.0 - abs(η)) / 
		(cos(θ2) * (sin(θ1) - sqrt(3.0) * cos(θ1)))
	
	elseif case1 == 4
	
		η = θf(0.0, 1.0, glQuad1[n,1])
		λ = sqrt(3.0) * (1.0 - abs(η)) / 
		(cos(θ2) * (sin(θ1) - sqrt(3.0) * cos(θ1)))
	
	elseif case1 == 5
	
		η = θf(0.0, 1.0, glQuad1[n,1])
		λ = sqrt(3.0) * (1.0 - η) / sin(θ2)
	else
		error("Unrecognized case.")
	end
	return aE(rPoints, λ, η, θ2, θ1, glQuad1, case2, assemblyOpts)
end

function aS(rPoints::Array{Float64,2}, θ1::Float64, θ2::Float64, θ::Float64, 
	dir::Int64, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::ComplexF64

	xPoints = Array{Float64,2}(undef, 3, 2)
	aInt = 0.0 + 0.0im
	η1 = 0.0 
	η2 = 0.0 
	ξ1 = 0.0

	for n in 1:assemblyOpts.ordGLIntNear
	
		(η1, ξ1) = subTriangles(θ2, θ * sin(θ1), dir)
		(η2, ξ2) = subTriangles(θf(0.0, θ, glQuad1[n,1]) * cos(θ1) + θ2, 
			(θ - θf(0.0, θ, glQuad1[n,1])) * sin(θ1), dir)
		simplex!(xPoints, η1, η2, ξ1, ξ2)
		aInt += glQuad1[n,2] * θf(0.0, θ, glQuad1[n,1]) * 
		kernelS(rPoints, xPoints, assemblyOpts.freqPhase)
	end
	return θ * aInt / 2.0
end

function aE(rPoints::Array{Float64,2}, λ::Float64, η::Float64, θ2::Float64, 
	θ1::Float64, glQuad1::Array{Float64,2}, case::Int64, 
	assemblyOpts::MaxGAssemblyOpts)::ComplexF64

	xPoints = Array{Float64,2}(undef, 3, 2)
	intVal = 0.0 + 0.0im
	ζ = 0.0
	
	for n in 1 : assemblyOpts.ordGLIntNear
	
		ζ = θf(0.0, λ, glQuad1[n,1])
		simplexE!(xPoints, ζ, η, θ2, θ1, case)
		intVal += glQuad1[n,2] * ζ * ζ * kernelEV(rPoints, xPoints, 
			assemblyOpts.freqPhase)
	end

	return λ * intVal / 2.0
end

function subTriangles(λ1::Float64, λ2::Float64, 
	dir::Int64)::Tuple{Float64,Float64}

	if dir == 1
		
		return (λ1, λ2)
	
	elseif dir == 2
		
		return ((1.0 - λ1 - λ2 * sqrt(3)) / 2.0, 
			(sqrt(3.0) + λ1 * sqrt(3.0) - λ2) / 2.0)
	
	elseif dir == 3
		
		return (( - 1.0 - λ1 + λ2 * sqrt(3)) / 2.0, 
			(sqrt(3.0) - λ1 * sqrt(3.0) - λ2) / 2.0)

	else
		
		error("Unrecognized case.")
	end
end

function equiJacobianEV(rPoints::Array{Float64,2})::Float64

	
	return sqrt(dot(cross(rPoints[:,2] - rPoints[:,1], 
		rPoints[:,3] - rPoints[:,1]), cross(rPoints[:,2] - rPoints[:,1], 
		rPoints[:,3] - rPoints[:,1]))) * sqrt(dot(cross(rPoints[:,5] - 
		rPoints[:,4], rPoints[:,6] - rPoints[:,4]), cross(rPoints[:,5] - 
		rPoints[:,4], rPoints[:,6] - rPoints[:,4]))) / 12.0
end

function equiJacobianS(rPoints::Array{Float64,2})::Float64

	return dot(cross(rPoints[:,1] - rPoints[:,2], rPoints[:,3] - rPoints[:,1]), 
		cross(rPoints[:,1] - rPoints[:,2], rPoints[:,3] - rPoints[:,1])) / 12.0
end

@inline function θf(θa::Float64, θb::Float64, pos::Float64)::Float64	
	
	return ((θb - θa) * pos + θa + θb) / 2.0   
end

function simplexV!(xPoints::Array{Float64,2}, θ4::Float64, θ3::Float64, 
	θ2::Float64, θ1::Float64)

	simplex!(xPoints, θ4 * cos(θ3) * cos(θ1) - 1.0, θ4 * sin(θ3) * cos(θ2) - 
		1.0, θ4 * cos(θ3) * sin(θ1), θ4 * sin(θ3) * sin(θ2))
	
	return nothing
end

function simplexE!(xPoints::Array{Float64,2}, λ::Float64, η::Float64, 
	θ2::Float64, θ1::Float64, case::Int64)

	if case == 1
	
		simplex!(xPoints, η, λ * cos(θ2) * cos(θ1) - η , λ * sin(θ2), 
			λ * cos(θ2) * sin(θ1))
	
	elseif case ==  - 1
	
		simplex!(xPoints,  -η,  -(λ * cos(θ2) * cos(θ1) - η) , λ * sin(θ2), 
			λ * cos(θ2) * sin(θ1))

	else

		error("Unrecognized case.")
	end
	return nothing
end

function simplex!(xPoints::Array{Float64,2}, η1::Float64, η2::Float64, 
	ξ1::Float64, ξ2::Float64)

	xPoints[1,1] = (sqrt(3.0) * (1 - η1) - ξ1) / (2 * sqrt(3))
	xPoints[2,1] = (sqrt(3.0) * (1 + η1) - ξ1) / (2 * sqrt(3))
	xPoints[3,1] = ξ1 / sqrt(3.0)
	xPoints[1,2] = (sqrt(3.0) * (1 - η2) - ξ2) / (2 * sqrt(3))
	xPoints[2,2] = (sqrt(3.0) * (1 + η2) - ξ2) / (2 * sqrt(3))
	xPoints[3,2] = ξ2 / sqrt(3.0)
	
	return nothing
end

function kernelEV(rPoints::Array{Float64,2}, 
	xPoints::Array{Float64,2}, freqPhase::ComplexF64)::ComplexF64

	return	scaleGreen(distMag(xPoints[1,1] * rPoints[1,1] + xPoints[2,1] * 
		rPoints[1,2] + xPoints[3,1] * rPoints[1,3] - (xPoints[1,2] * 
			rPoints[1,4] + xPoints[2,2] * rPoints[1,5] + xPoints[3,2] * 
			rPoints[1,6]), xPoints[1,1] * rPoints[2,1] + xPoints[2,1] * 
		rPoints[2,2] + xPoints[3,1] * rPoints[2,3] - (xPoints[1,2] * 
			rPoints[2,4] + xPoints[2,2] * rPoints[2,5] + xPoints[3,2] * 
			rPoints[2,6]), xPoints[1,1] * rPoints[3,1] + xPoints[2,1] * 
		rPoints[3,2] + xPoints[3,1] * rPoints[3,3] - (xPoints[1,2] * 
			rPoints[3,4] + xPoints[2,2] * rPoints[3,5] + xPoints[3,2] * 
			rPoints[3,6])), freqPhase)
end

function kernelS(rPoints::Array{Float64,2}, 
	xPoints::Array{Float64,2}, freqPhase::ComplexF64)::ComplexF64
	
	return	scaleGreen(distMag(xPoints[1,1] * rPoints[1,1] + xPoints[2,1] * 
		rPoints[1,2] + xPoints[3,1] * rPoints[1,3] - (xPoints[1,2] * 
			rPoints[1,1] + xPoints[2,2] * rPoints[1,2] + xPoints[3,2] * 
			rPoints[1,3]), xPoints[1,1] * rPoints[2,1] + xPoints[2,1] * 
		rPoints[2,2] + xPoints[3,1] * rPoints[2,3] - (xPoints[1,2] * 
			rPoints[2,1] + xPoints[2,2] * rPoints[2,2] + xPoints[3,2] * 
			rPoints[2,3]), xPoints[1,1] * rPoints[3,1] + xPoints[2,1] * 
		rPoints[3,2] + xPoints[3,1] * rPoints[3,3] - (xPoints[1,2] * 
			rPoints[3,1] + xPoints[2,2] * rPoints[3,2] + xPoints[3,2] * 
			rPoints[3,3])), freqPhase)
end

# CPU computation of Green function
G_slf = genGreenSlf!(greenCircAA, volA, assemblyInfo) # A with A
print("G_slf: ", length(G_slf),"\n") # How to know the size of G if it's in 3d and size(G) gives (3,3,8,8,8)
# Does this mean: Z,Z, x=8, y,8,z=8? Is this a 3D matrix because it sure looks like it?
# It doesn't seem that the 3 means Z here since there are 3*3*8*8*=4608 elements in G...

G_ext = genGreenExt!(greenCircBA, volB, volA, assemblyInfo) # B with A
print("G_ext: ", length(G_ext), "\n")

# If G is a 3D matrix, then how do we define ji since I'm not sure how to do 3D matrix multiplication...



# 1. XX
# Pre-allocate memory for circulant XX green function vector. 
# greenCircAA_XX = Array{ComplexF64}(undef, 1, 1, 2 * cellsA[1], 2 * cellsA[1], 
# 	2 * cellsA[1])
# greenCircBA_XX = Array{ComplexF64}(undef, 1, 1, cellsB[1] + cellsA[1], cellsB[2] +
# 	cellsA[2], cellsB[3] + cellsA[3])

# G_slf = genGreenSlf!(greenCircAA_XX, volA, assemblyInfo) # A with A
# print("G_slf", length(G_slf),"\n") # How to know the size of G if it's in 3d and size(G) gives (3,3,8,8,8)
# # Does this mean: Z,Z, x=8, y,8,z=8? Is this a 3D matrix because it sure looks like it?
# # It doesn't seem that the 3 means Z here since there are 3*3*8*8*=4608 elements in G...

# G_ext = genGreenExt!(greenCircBA_XX, volB, volA, assemblyInfo) # B with A
# print("G_ext", length(G_ext), "\n")

# 2. XY
# Pre-allocate memory for circulant XY green function vector. 
# greenCircAA_XY = Array{ComplexF64}(undef, 1, 2, 2 * cellsA[1], 2 * cellsA[1], 
# 	2 * cellsA[1])
# greenCircBA_XY = Array{ComplexF64}(undef, 1, 2, cellsB[1] + cellsA[1], cellsB[2] +
# 	cellsA[2], cellsB[3] + cellsA[3])

# G_slf = genGreenSlf!(greenCircAA_XY, volA, assemblyInfo) # A with A
# print("G_slf", length(G_slf),"\n") # How to know the size of G if it's in 3d and size(G) gives (3,3,8,8,8)
# # Does this mean: Z,Z, x=8, y,8,z=8? Is this a 3D matrix because it sure looks like it?
# # It doesn't seem that the 3 means Z here since there are 3*3*8*8*=4608 elements in G...

# G_ext = genGreenExt!(greenCircBA_XY, volB, volA, assemblyInfo) # B with A
# print("G_ext", length(G_ext), "\n")

# 3. XZ
# Pre-allocate memory for circulant XZ green function vector. 
# greenCircAA_XZ = Array{ComplexF64}(undef, 1, 3, 2 * cellsA[1], 2 * cellsA[1], 
# 	2 * cellsA[1])
# greenCircBA_XZ = Array{ComplexF64}(undef, 1, 3, cellsB[1] + cellsA[1], cellsB[2] +
# 	cellsA[2], cellsB[3] + cellsA[3])

# G_slf = genGreenSlf!(greenCircAA_XZ, volA, assemblyInfo) # A with A
# print("G_slf", length(G_slf),"\n") # How to know the size of G if it's in 3d and size(G) gives (3,3,8,8,8)
# # Does this mean: Z,Z, x=8, y,8,z=8? Is this a 3D matrix because it sure looks like it?
# # It doesn't seem that the 3 means Z here since there are 3*3*8*8*=4608 elements in G...

# G_ext = genGreenExt!(greenCircBA_XZ, volB, volA, assemblyInfo) # B with A
# print("G_ext", length(G_ext), "\n")

# 4. YX
# Pre-allocate memory for circulant XY green function vector. 
# greenCircAA_YX = Array{ComplexF64}(undef, 2, 1, 2 * cellsA[1], 2 * cellsA[1], 
# 	2 * cellsA[1])
# greenCircBA_YX = Array{ComplexF64}(undef, 2, 1, cellsB[1] + cellsA[1], cellsB[2] +
# 	cellsA[2], cellsB[3] + cellsA[3])

# G_slf = genGreenSlf!(greenCircAA_YX, volA, assemblyInfo) # A with A
# print("G_slf", length(G_slf),"\n") # How to know the size of G if it's in 3d and size(G) gives (3,3,8,8,8)
# # Does this mean: Z,Z, x=8, y,8,z=8? Is this a 3D matrix because it sure looks like it?
# # It doesn't seem that the 3 means Z here since there are 3*3*8*8*=4608 elements in G...

# G_ext = genGreenExt!(greenCircBA_YX, volB, volA, assemblyInfo) # B with A
# print("G_ext", length(G_ext), "\n")

# 5. YY
# Pre-allocate memory for circulant YY green function vector. 
# greenCircAA_YY = Array{ComplexF64}(undef, 2, 2, 2 * cellsA[1], 2 * cellsA[1], 
# 	2 * cellsA[1])
# greenCircBA_YY = Array{ComplexF64}(undef, 2, 2, cellsB[1] + cellsA[1], cellsB[2] +
# 	cellsA[2], cellsB[3] + cellsA[3])

# G_slf = genGreenSlf!(greenCircAA_YY, volA, assemblyInfo) # A with A
# print("G_slf", length(G_slf),"\n") # How to know the size of G if it's in 3d and size(G) gives (3,3,8,8,8)
# # Does this mean: Z,Z, x=8, y,8,z=8? Is this a 3D matrix because it sure looks like it?
# # It doesn't seem that the 3 means Z here since there are 3*3*8*8*=4608 elements in G...

# G_ext = genGreenExt!(greenCircBA_YY, volB, volA, assemblyInfo) # B with A
# print("G_ext", length(G_ext), "\n")

# 6. YZ
# Pre-allocate memory for circulant YZ green function vector. 
# greenCircAA_YZ = Array{ComplexF64}(undef, 2, 3, 2 * cellsA[1], 2 * cellsA[1], 
# 	2 * cellsA[1])
# greenCircBA_YZ = Array{ComplexF64}(undef, 2, 3, cellsB[1] + cellsA[1], cellsB[2] +
# 	cellsA[2], cellsB[3] + cellsA[3])

# G_slf = genGreenSlf!(greenCircAA_YZ, volA, assemblyInfo) # A with A
# print("G_slf", length(G_slf),"\n") # How to know the size of G if it's in 3d and size(G) gives (3,3,8,8,8)
# # Does this mean: Z,Z, x=8, y,8,z=8? Is this a 3D matrix because it sure looks like it?
# # It doesn't seem that the 3 means Z here since there are 3*3*8*8*=4608 elements in G...

# G_ext = genGreenExt!(greenCircBA_YZ, volB, volA, assemblyInfo) # B with A
# print("G_ext", length(G_ext), "\n")

# 7. ZX
# Pre-allocate memory for circulant ZX green function vector. 
# greenCircAA_ZX = Array{ComplexF64}(undef, 3, 2, 2 * cellsA[1], 2 * cellsA[1], 
# 	2 * cellsA[1])
# greenCircBA_ZX = Array{ComplexF64}(undef, 3, 2, cellsB[1] + cellsA[1], cellsB[2] +
# 	cellsA[2], cellsB[3] + cellsA[3])

# G_slf = genGreenSlf!(greenCircAA_ZX, volA, assemblyInfo) # A with A
# print("G_slf", length(G_slf),"\n") # How to know the size of G if it's in 3d and size(G) gives (3,3,8,8,8)
# # Does this mean: Z,Z, x=8, y,8,z=8? Is this a 3D matrix because it sure looks like it?
# # It doesn't seem that the 3 means Z here since there are 3*3*8*8*=4608 elements in G...

# G_ext = genGreenExt!(greenCircBA_ZX, volB, volA, assemblyInfo) # B with A
# print("G_ext", length(G_ext), "\n")


# 8. ZY
# Pre-allocate memory for circulant ZY green function vector. 
# greenCircAA_ZY = Array{ComplexF64}(undef, 3, 2, 2 * cellsA[1], 2 * cellsA[1], 
# 	2 * cellsA[1])
# greenCircBA_ZY = Array{ComplexF64}(undef, 3, 2, cellsB[1] + cellsA[1], cellsB[2] +
# 	cellsA[2], cellsB[3] + cellsA[3])

# G_slf = genGreenSlf!(greenCircAA_ZY, volA, assemblyInfo) # A with A
# print("G_slf", length(G_slf),"\n") # How to know the size of G if it's in 3d and size(G) gives (3,3,8,8,8)
# # Does this mean: Z,Z, x=8, y,8,z=8? Is this a 3D matrix because it sure looks like it?
# # It doesn't seem that the 3 means Z here since there are 3*3*8*8*=4608 elements in G...

# G_ext = genGreenExt!(greenCircBA_ZY, volB, volA, assemblyInfo) # B with A
# print("G_ext", length(G_ext), "\n")

# 9. ZZ
# Pre-allocate memory for circulant ZZ green function vector. 
# greenCircAA_ZZ = Array{ComplexF64}(undef, 3, 3, 2 * cellsA[1], 2 * cellsA[1], 
# 	2 * cellsA[1])
# greenCircBA_ZZ = Array{ComplexF64}(undef, 3, 3, cellsB[1] + cellsA[1], cellsB[2] +
# 	cellsA[2], cellsB[3] + cellsA[3])

# G_slf = genGreenSlf!(greenCircAA_ZZ, volA, assemblyInfo) # A with A
# print("G_slf", length(G_slf),"\n") # How to know the size of G if it's in 3d and size(G) gives (3,3,8,8,8)
# # Does this mean: Z,Z, x=8, y,8,z=8? Is this a 3D matrix because it sure looks like it?
# # It doesn't seem that the 3 means Z here since there are 3*3*8*8*=4608 elements in G...

# G_ext = genGreenExt!(greenCircBA_ZZ, volB, volA, assemblyInfo) # B with A
# print("G_ext", length(G_ext), "\n")
