"""
The MaxGParallelUtilities provides support function for parallel computing in 
MaxG.The code is distributed under GNU LGPL.

Author: Sean Molesky 
"""
module MaxGParallelUtilities  
using Distributed, Base.Threads, LinearAlgebra
export ParallelConfig, nodeArrayW!, threadArrayW!, threadPtrW!, threadPtrR!, 
threadCpy!, genWorkBounds
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
end