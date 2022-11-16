"""
Utilities to interact with the maxG GPU library. 
"""
module MaxGCUDA
using Printf, Base.Threads, Libdl, MaxGParallelUtilities, MaxGStructs, MaxGCirc
export libInitCu!, libFinlCu!, devGlbInit!, devGlbFinl!, devCount, devSysInit!,devSysFinl!, devSysMemFinl!, devMem, sysMemCpy!, solve!, sysAdj!, fieldConvert!
"""
Open the libCu shared object that allows MaxG to perform GPU operations.
"""
function libInitCu!()::Ref{Ptr{Nothing}}

  	return  dlopen("./libs/libMGCu.so")
end
"""

    libFinlCu!(libHandle::Ref{Ptr{Nothing}})::Nothing

Close the  library connection specified by the library handle libHandle.
"""
function libFinlCu!(libHandle::Ref{Ptr{Nothing}})::Nothing

	dlclose(libHandle[])
	return nothing
end
"""

    deviceCount(libHandle::Ref{Ptr{Nothing}})::Int

Return the number of accessible devices.
"""
function devCount(libHandle::Ref{Ptr{Nothing}})::Int

	return convert(Int, ccall(dlsym(libHandle[], :devCount), Int32, ()))
end
"""

    devGlbInit!(libHandle::Ref{Ptr{Nothing}}, sysNum::Int, devMax::Int)::Nothing

Initialize global number of systems that will be handled by the Cu library.
"""
function devGlbInit!(libHandle::Ref{Ptr{Nothing}}, sysNum::Int, devMax::Int)::Nothing

	ccall(dlsym(libHandle[], :glbInit), Cvoid, (Int32, Int32), sysNum, devMax)
	return nothing
end
"""

    devMem(libHandle::Ref{Ptr{Nothing}}, numDevs::Int, devList::Array{Int, 1})::UInt

Returns free device memory in megabytes.
"""
function devMem(libHandle::Ref{Ptr{Nothing}}, numDevs::Int, devList::Array{Int, 1})::UInt

	return ccall(dlsym(libHandle[], :devMem), Csize_t, (Int32, Ref{Int32}), numDevs, convert(Array{Int32, 1}, devList))
end
"""

    devSysMemInit!(sysItr::Int, sysInfo::System, compOpts::MaxGCompOpts, solverOpts::MaxGSolverOpts)::Int

Open shared library, initialize global variables and partition green blocks over the 
number of devices specified by the device lists.
"""
function devSysMemInit!(sysItr::Int, sysInfo::MaxGSysInfo, compOpts::MaxGCompOpts, solverOpts::MaxGSolverOpts)::Int

	if convert(Int, ccall(dlsym(sysInfo.viCuLibHandle[], :sysInit), Int32, (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Ref{Int32}, Int32, Ref{Int32}), sysItr - 1,
		 compOpts.gBlocks, compOpts.gThreads, sysInfo.bodies, sysInfo.totalCells, 
		 solverOpts.basisDimension, solverOpts.deflateDimension, solverOpts.prefacMode, 
		 length(compOpts.deviceList), convert(Array{Int32, 1}, compOpts.deviceList), 
		 length(compOpts.deviceListDMR), convert(Array{Int32, 1}, compOpts.deviceListDMR))) != 0
		error("devSysMemInit! has failed to allocate GPU memory.\n")
		return 1
	end
	return 0
end
"""

    setDevice!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, devItr::Int)::Nothing
    
Set the current device to the number specified in position devItr of the  device list, 
devList[devItr].
"""
function devSet!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int,  devItr::Int)::Nothing

	ccall(dlsym(libHandle[], :setDevice), Cvoid, (Int32, Int32), sysItr - 1, devItr - 1)
	return nothing
end
"""

	devSync(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, devItr::Int)::Nothing

Wait for completion of all scheduled tasks on the device specified at position devItr of 
the  device list, devList[devItr].
"""
function devSync(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, devItr::Int)::Nothing

	ccall(dlsym(libHandle[], :devSync), Cvoid, (Int32, Int32), sysItr - 1, devItr - 1)
	return nothing
end
"""
	
	blcInitStream!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, cellsSrc::Array{Int,1}, cellsTrg::Array{Int,1}, blcItr::Int)::Int

Initialize memory for a  FFT (green block stream).
"""
function blcInitStream!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, cellsSrc::Array{Int,1}, cellsTrg::Array{Int,1}, blcItr::Int)::Int

	if convert(Int, ccall(dlsym(libHandle[], :blcInitStream), Int32, (Int32, Ref{Int32}, Ref{Int32}, Int32), sysItr - 1, convert(Array{Int32, 1}, cellsSrc), convert(Array{Int32, 1}, cellsTrg), blcItr - 1)) != 0
		error("blcInitStream! has failed to allocate GPU memory.\n")
		return 1
	end
	return 0
end
"""

    volInitStream!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, cellsBdy::Array{Int32,1}, volItr::Int)::Int

Initialize memory for a body stream.
"""
function volInitStream!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, cellsBdy::Array{Int,1}, volItr::Int)::Int

	if convert(Int, ccall(dlsym(libHandle[], :volInitStream), Int32, (Int32, Ref{Int32}, Int32), sysItr - 1, convert(Array{Int32, 1}, cellsBdy), volItr - 1)) != 0
		error("volInitStream! has failed to allocate GPU memory.\n")
		return 1
	end
	return 0
end
"""

    blcFinlStream!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, blcItr::Int)::Nothing

Close a FFT (green block) stream, clearing the GPU
memory.
"""
function blcFinlStream!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, blcItr::Int)::Nothing

	ccall(dlsym(libHandle[], :blcFinlStream), Cvoid, (Int32, Int32), sysItr - 1, blcItr - 1)
	return nothing
end
"""

    volFinlStream!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, volItr::Int)::Nothing

Close a body stream, clearing the GPU memory.
"""
function volFinlStream!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, volItr::Int)::Nothing

	ccall(dlsym(libHandle[], :volFinlStream), Cvoid, (Int32, Int32), sysItr - 1, volItr - 1)
	return nothing
end
"""

    devSysMemFinl!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int)::Nothing

Free global variables of the  shared library.
"""
function devSysMemFinl!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int)::Nothing

	ccall(dlsym(libHandle[], :sysFinl), Cvoid, (Int32,), sysItr - 1)
	return nothing
end
"""

    glbFinlCu!(libHandle::Ref{Ptr{Nothing}})::Nothing

Free global memory in the Cu library.
"""
function devGlbFinl!(libHandle::Ref{Ptr{Nothing}}, sysNum::Int)::Nothing

	ccall(dlsym(libHandle[], :glbFinl), Cvoid, (Int32,), sysNum)
	@printf(stdout, "Global gCu memory cleared.\n")
	return nothing
end
"""

    blcMemCpy!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, juArr::AbstractArray{ComplexF64}, memItr::Int, blcItr::Int)::Int

Copy memory from Julia to a FFT (green block) stream. 
# Arguments
memItr: == 1 access fftGreenSlf, green block entries for aligned Cartesian directions. == 2
access fftGreenExc, green block entries for xy, xz and yz Cartesian pairings.
"""
function blcMemCpy!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, juArr::AbstractArray{ComplexF64}, memItr::Int, blcItr::Int)::Int

	if convert(Int, ccall(dlsym(libHandle[], :blcMemCpy), Int32, (Int32, Ref{ComplexF64}, Int32, Int32), sysItr - 1, juArr, memItr - 1, blcItr - 1)) != 0
		error("blcMemCpy! has failed to copy memory to the GPU.\n")
		return 1
	end
	return 0
end
"""

    volMemCpy!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, juArr::AbstractArray{ComplexF64}, volItr::Int)::Int

Copy material response from Julia to a body stream.
"""
function volMemCpy!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, juArr::AbstractArray{ComplexF64}, volItr::Int)::Int

	if convert(Int, ccall(dlsym(libHandle[], :volMemCpy), Int32, (Int32, Ref{ComplexF64}, Int32), sysItr - 1, juArr, volItr - 1)) != 0
		error("volMemCpy! has failed to copy memory to the GPU.\n")
		return 1
	end
	return 0
end
"""

    sysMemCpy!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, juArr::AbstractArray{ComplexF64}, dir::Int, memItr::Int)::Int

Copy memory from Julia to the  system GPU stream 
# Arguments
memItr: == 1 access sysCurrSrc (totlCurr), == 2 access sysCurrTrg (initCurr).
dir: == 1 transfers from host to device, == 2 device to host.
"""
function sysMemCpy!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, juArr::AbstractArray{ComplexF64}, dir::Int, memItr::Int)::Int

	if convert(Int, ccall(dlsym(libHandle[], :sysMemCpy), Int32, (Int32, Ref{ComplexF64}, Int32, Int32), sysItr - 1, juArr, dir - 1, memItr - 1)) != 0
		error("sysMemCpy! has failed to perform GPU memory copy.\n")
		return 1
	end
	return 0
end
"""

    fftExeHost!(libHandle::Ref{Ptr{Nothing}}, sysIds::AbstractArray{Int, 1}, sysSize::Int, fftDir::Int, memItr::Int, blcItr::Int)::Int

Perform forward or reverse Fourier transform a FFT (green block) stream memory location.
# Arguments
fftDir: direction of Fourier transform == 1 FORWARD; == 2 REVERSE
memItr: memory index == 1 access fftGreenSlf; == 2 access fftGreenExc
"""
function fftExeHost!(libHandle::Ref{Ptr{Nothing}}, sysIds::AbstractArray{Int, 1}, sysSize::Int, fftDir::Int, memItr::Int, blcItr::Int)::Int

	sysNum = length(sysIds)
	sysIdsC = Array{Int32, 1}(undef, sysNum)

	for sysInd in 1:sysNum
		sysIdsC[sysInd] = convert(Int32, sysIds[sysInd] - 1)
	end

	if convert(Int, ccall(dlsym(libHandle[], :fftExeHost), Int32, (Ref{Int32}, Int32, Int32, Int32, Int32), sysIdsC, sysSize, fftDir - 1, memItr - 1, blcItr - 1)) != 0
		error("fftExeHost! has failed to execute Fourier transform.\n")
		return 1
	end
	return 0
end
"""

    fftInitHost!(libHandle::Ref{Ptr{Nothing}}, sysIds::AbstractArray{Int, 1}, sysSize::Int)::Int

Perform forward Fourier transforms on all Green function memory locations specified by sysIds.
"""
function fftInitHost!(libHandle::Ref{Ptr{Nothing}}, sysIds::AbstractArray{Int, 1}, sysSize::Int)::Int

	sysNum = length(sysIds)
	sysIdsC = Array{Int32, 1}(undef, sysNum)

	for sysInd in 1:sysNum
		sysIdsC[sysInd] = convert(Int32, sysIds[sysInd] - 1)
	end

	if convert(Int, ccall(dlsym(libHandle[], :fftInitHost), Int32, (Ref{Int32}, Int32), sysIdsC, sysSize)) != 0
		error("fftInitHost! has failed.\n")
		return 1
	end
	return 0
end
"""

    compInit(libHandle::Ref{Ptr{Nothing}}, sysIds::Array{Int, 1}, 
	system::Array{MaxGVol, 1}, sysInfo::MaxGSysInfo, assemblyInfo::MaxGAssemblyOpts, 
	solverOpts::MaxGSolverOpts)::Int

Load Julia calculated Green matrix entries onto CUDA devices.
"""
function compInit(libHandle::Ref{Ptr{Nothing}}, sysIds::Array{Int, 1}, 
	system::Array{MaxGVol, 1}, sysInfo::MaxGSysInfo, assemblyInfo::MaxGAssemblyOpts, 
	solverOpts::MaxGSolverOpts)::Int

	sysItr = 0
	blcItr = 0
	srcObj = system[1]
	trgObj = system[1]
	circCells = srcObj.cells .+ trgObj.cells

	for srcBdy in 1:sysInfo.bodies
		
		if srcBdy > 1
			srcObj = system[srcBdy]
		end
		
		for trgBdy in 1:sysInfo.bodies
			
			if trgBdy > 1 || srcBdy > 1
				trgObj = system[trgBdy]
			end
			circCells = srcObj.cells .+ trgObj.cells
			greenCirc = Array{ComplexF64}(undef, 3, 2, circCells[1], circCells[2], circCells[3])
			gCReorder = Array{ComplexF64}(undef, circCells[1], circCells[2], circCells[3], 3, 2)
			
			if(trgBdy == srcBdy)
				genGreenSlf!(greenCirc, srcObj, assemblyInfo)
			else
				genGreenExt!(greenCirc, trgObj, srcObj, assemblyInfo)
			end

			for ortB in 1:2 

				for ortA in 1:3

					for indZ in 1:circCells[3]

						for indY in 1:circCells[2]

							@threads for indX in 1:circCells[1]
								
								gCReorder[indX, indY, indZ, ortA, ortB] = 
								greenCirc[ortA, ortB, indX, indY, indZ]
							end
						end
					end
				end
			end
			blcItr = trgBdy + (srcBdy - 1) * sysInfo.bodies
			
			for sysItr in sysIds
				# Parallel 	
				if(blcMemCpy!(libHandle, sysItr, gCReorder[:, :, :, :, 1], 1, blcItr) != 0)
					return 1
				end
				# Perpendicular 
				if(blcMemCpy!(libHandle, sysItr, gCReorder[:, :, :, :, 2], 2, blcItr) != 0)
					return 1
				end
				@printf(stdout, "Block %d%d initialized on system %d.\n", trgBdy, srcBdy, sysItr)
			end
		end
		
		if(solverOpts.prefacMode == 0)

			for sysItr in sysIds
				
				if(volMemCpy!(libHandle, sysItr, srcObj.elecSus, srcBdy) != 0)
					return 1
				end
			end
		else
			volMatResp = Array{ComplexF64, 1}(undef, 3 * srcObj.totalCells)

			@threads for vecItr in 1:(3 * srcObj.totalCells)
				volMatResp[vecItr] = srcObj.elecSus[vecItr] / (1.0 + srcObj.elecSus[vecItr])
			end
			
			for sysItr in sysIds
				if(volMemCpy!(libHandle, sysItr, volMatResp, srcBdy) != 0)
					return 1
				end
			end
		end
	end
	fftInitHost!(libHandle, sysIds, length(sysIds))
	@printf(stdout, "\nForward Fourier transform of all Green function memory locations complete.\n")
	return 0
end
"""

    devSysInit!(sysIds::Array{Int,1}, system::Array{MaxGVol, 1}, 
    sysInfo::Array{MaxGSysInfo, 1})::Int

Fully initializes device (GPU) memory and streams in preparation for performing  computations.
"""
function devSysInit!(sysIds::Array{Int,1}, system::Array{MaxGVol, 1}, 
	sysInfo::Array{MaxGSysInfo, 1})::Int

	for sysItr in sysIds

		if(devSysMemInit!(sysItr, sysInfo[sysItr], sysInfo[sysItr].computeInfo, sysInfo[sysItr].solverInfo) 
			!= 0)
			return 1
		end
	end
	srcCells = sysInfo[1].cellList[1, :]
	trgCells = sysInfo[1].cellList[1, :]
	libHandle = sysInfo[1].viCuLibHandle

	for srcBdy in 1:sysInfo[1].bodies
	
		if srcBdy > 1
			srcCells = sysInfo[1].cellList[srcBdy, :]
		end
		
		for trgBdy in 1:sysInfo[1].bodies		
			
			if trgBdy > 1 || srcBdy > 1
				trgCells = sysInfo[1].cellList[trgBdy, :]
			end
			
			for sysItr in sysIds
				
				if(blcInitStream!(libHandle, sysItr, srcCells, trgCells, 
					trgBdy + (srcBdy - 1) * sysInfo[1].bodies) != 0)
					return 1
				end
				
			end
		end
	
		for sysItr in sysIds
		
			if(volInitStream!(libHandle, sysItr, srcCells, srcBdy) != 0)
				return 1
			end
		end
	end
	
	if(compInit(libHandle, sysIds, system, sysInfo[1], sysInfo[1].assemblyInfo, sysInfo[1].solverInfo) != 0)
		return 1
	end
	@printf(stdout, " initialization complete, specified systems are ready.\n\n")
	return 0
end
"""
    devSysFinl!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, sysInfo::MaxGSysInfo)::Nothing

Clear all device computation memory, prepare for program termination.
"""
function devSysFinl!(libHandle::Ref{Ptr{Nothing}}, sysItr::Int, sysInfo::MaxGSysInfo)::Nothing

	for srcBdy in 1:sysInfo.bodies
		
		for trgBdy in 1:sysInfo.bodies		

			blcFinlStream!(libHandle, sysItr, trgBdy + (srcBdy - 1) * sysInfo.bodies)
			@printf(stdout, "Block %d%d cleared on system %d.\n", trgBdy, srcBdy, sysItr)
		end
		volFinlStream!(libHandle, sysItr, srcBdy)
	end
	@printf(stdout, "Device system memory cleared.\n")
	return nothing
end
"""

    sysAdj!(libHandle::Ref{Ptr{Nothing}}, sysIds::AbstractArray{Int,1}, sysSize::Int)::Nothing

Conjugates the material response in all bodies for specified systems. 
Used in solving adjoint systems.
"""
function sysAdj!(libHandle::Ref{Ptr{Nothing}}, sysIds::AbstractArray{Int,1})::Nothing

	sysNum = length(sysIds)
	sysIdsC = Array{Int32, 1}(undef, sysNum)

	for sysInd in 1:sysNum
		sysIdsC[sysInd] = convert(Int32, sysIds[sysInd] - 1)
	end

	if(ccall(dlsym(libHandle[], :sysAdj), Int32, (Ref{Int32}, Int32), sysIdsC, 
		sysNum) != 0)
		error("Failed to switch to adjoint system.")
	end
	return nothing
end
"""

    invSolve!(libHandle::Ref{Ptr{Nothing}}, sysIds::Array{Int32,1}, sysSize::Int, deflateMode::Array{Int32,1}, solTol::Array{Float64,1}, numIts::Array{Int32,1}, relRes::Array{Float64,1})::Int

Perform inverse solve for in current sysCurrTrg and return the relative magnitude of the 
residual. Returns 0 on successful exit, 1 otherwise.
# Arguments
sysIds: Systems to solve.
sysSize: Number of systems.
deflateMode: A value of !0 indicates that the solver has been previously called on a similar
 linear 
system and that this existing deflation space should be used in the first Arnoldi iteration.	
solTol: Relative allowable magnitude of the residual, difference between true and approximate
images.
numIts: Number of iterations used by the solver.
relRes: Relative residuals of the solutions.
"""
function invSolve!(libHandle::Ref{Ptr{Nothing}}, sysIds::Array{Int32,1}, sysSize::Int, deflateMode::Array{Int32,1}, solTol::Array{Float64,1}, numIts::Array{Int32,1}, relRes::Array{Float64,1})::Int

	return convert(Int, ccall(dlsym(libHandle[], :invSolve), Int32, (Ref{Int32}, Int32, Ref{Int32}, Ref{Float64}, Ref{Int32}, Ref{Float64}), sysIds, sysSize, deflateMode, solTol, numIts, relRes))
end
"""

    solve!(libHandle::Ref{Ptr{Nothing}}, sysIds::AbstractArray{Int, 1}, deflateMode::AbstractArray{Int, 1}, solTol::AbstractArray{Float64, 1}, initCurr::AbstractArray{ComplexF64, 2}, totlCurr::AbstractArray{ComplexF64, 2})::Tuple{Array{Float64, 1}, Array{Int32, 1}}

Perform inverse solve for in current sysCurrTrg and return the relative magnitude of the 
residual.
# Arguments
deflateMode: value of !0 indicates that the solver has been previously called on a similar linear 
system and that this existing deflation space should be used in the first Arnoldi iteration.
solTol: Relative allowable magnitude of the residual, difference between true and approximate
images.
totlCurr: Guess (approximation) of the total system current. On successful termination mutated to 
true solution.
initCurr: Target current. Right hand side of  equation
"""
function solve!(libHandle::Ref{Ptr{Nothing}}, sysIds::AbstractArray{Int, 1}, deflateMode::AbstractArray{Int, 1}, solTol::AbstractArray{Float64, 1}, initCurr::AbstractArray{ComplexF64, 2}, totlCurr::AbstractArray{ComplexF64, 2})::Tuple{Array{Float64, 1}, Array{Int32, 1}}

	sysItr = 0
	sysSize = length(sysIds)
	numIts = Array{Int32, 1}(undef, sysSize)
	relRes = Array{Float64, 1}(undef, sysSize)
	sysIdsC = Array{Int32, 1}(undef, sysSize)
	defModeC = Array{Int32, 1}(undef, sysSize)

	for sysInd in 1:sysSize
		sysItr = sysIds[sysInd]
		sysIdsC[sysInd] = convert(Int32, sysIds[sysInd] - 1)
		defModeC[sysInd] = convert(Int32, deflateMode[sysInd])

		if sysMemCpy!(libHandle, sysItr, view(totlCurr, :, sysInd), 1, 1) != 0
			error("Failed to load source current to GPU solver.")
			numIts[1] = -1
			return (relRes, numIts)
		end

		if sysMemCpy!(libHandle, sysItr, view(initCurr, :, sysInd), 1, 2) != 0
			error("Failed to load target current to GPU solver.")
			numIts[1] = -1
			return (relRes, numIts)
		end
	end

	if(invSolve!(libHandle, sysIdsC, sysSize, defModeC, solTol, numIts, relRes) != 0)
		error("Inverse solve failure.")
		numIts[1] = -1
		return (relRes, numIts)
	end

	for sysInd in 1:sysSize
		sysItr = sysIds[sysInd]

		if sysMemCpy!(libHandle, sysItr, view(totlCurr, :, sysInd), 2, 1) != 0
			error("Failed to copy solution from GPU solver.")
			numIts[1] = -1
			return (relRes, numIts)
		end
	end
	return (relRes, numIts)
end
"""

    fieldConvert!(libHandle::Ref{Ptr{Nothing}}, sysIds::AbstractArray{Int, 1}, field::AbstractArray{ComplexF64,2})::Nothing

Convert the total current density found by the solver into an electric field multiplied by 
free space wave vector.
"""
function fieldConvert!(libHandle::Ref{Ptr{Nothing}}, sysIds::AbstractArray{Int, 1}, field::AbstractArray{ComplexF64,2})::Nothing

	sysItr = 0
	sysSize = length(sysIds)
	sysIdsC = Array{Int32, 1}(undef, sysSize)

	for sysInd in 1:sysSize
		sysIdsC[sysInd] = convert(Int32, sysIds[sysInd] - 1)
	end

	if(ccall(dlsym(libHandle[], :fieldConvert), Int32, (Ref{Int32}, Int32), sysIdsC, sysSize) 
		!= 0)
		error("Failed to convert total system current density into electric field.")
		return nothing
	end

	for sysInd in 1:sysSize
		sysItr = sysIds[sysInd]

		if(sysMemCpy!(libHandle, sysItr, view(field, :, sysInd), 2, 1) != 0)
			error("Failed to copy solution from GPU solver.")
			return nothing
		end
	end
	return nothing
end
end