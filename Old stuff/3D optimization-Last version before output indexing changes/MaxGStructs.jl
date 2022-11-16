"""
The MaxGStructs module defines the layout of the various computational 
structures used elsewhere in MaxG. The code distributed under GNU LGPL.

Author: Sean Molesky 

Reference: Sean Molesky, MaxG documentation section ...
"""
module MaxGStructs
using Random, MaxGParallelUtilities
export MaxGDom, MaxGVol, genMaxGVol, MaxGAssemblyOpts
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
Valid MaxG operation settings
"""
@enum MaxGOptMode begin

	singleSlv
	greenFunc
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

end