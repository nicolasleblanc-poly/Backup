"""
The MaxGCirc module furnishes the unique elements of the electromagnetic 
Green functions, embedded in a circulant form. The code is distributed under 
GNU LGPL.

Author: Sean Molesky 

Reference: Sean Molesky, MaxG documentation sections II and IV.
"""
module MaxGCirc
using MaxGParallelUtilities, MaxGStructs, MaxGBasisIntegrals  
export genGreenExt!, genGreenSlf!, facePairs, cubeFaces, separationGrid, greenSlfFunc!
"""

	genGreenExt!(greenCirc::Array{ComplexF64}, srcVol::MaxGVol, 
	trgVol::MaxGVol, assemblyInfo::MaxGAssemblyOpts)::Nothing

Calculate circulant form for the discrete Green function between a target 
volume, trgVol, and a source domain, srcVol.
"""
function genGreenExt!(greenCirc::Array{ComplexF64}, trgVol::MaxGVol, 
	srcVol::MaxGVol, assemblyInfo::MaxGAssemblyOpts)::Nothing

	fPairs = facePairs()
	trgFaces = cubeFaces(trgVol.scale)
	srcFaces = cubeFaces(srcVol.scale)
	sGridT = separationGrid(trgVol, srcVol, 0)
	sGridS = separationGrid(trgVol, srcVol, 1)
	glQuadFar = gaussQuad2(assemblyInfo.ordGLIntFar)
	glQuadMed = gaussQuad2(assemblyInfo.ordGLIntMed)
	glQuadNear = gaussQuad2(assemblyInfo.ordGLIntNear)
	assembleGreenCircExt!(greenCirc, trgVol, srcVol, sGridT, sGridS, glQuadFar, 
		glQuadMed, glQuadNear, trgFaces, srcFaces, fPairs, assemblyInfo)
	
	return nothing
end
"""
	
	genGreenSlf!(greenCirc::Array{ComplexF64}, slfVol::MaxGVol, 
	assemblyInfo::MaxGAssemblyOpts)::Nothing

Calculate circulant form for the discrete Green function on a single domain.
"""
function genGreenSlf!(greenCirc::Array{ComplexF64}, slfVol::MaxGVol, 
	assemblyInfo::MaxGAssemblyOpts)::Nothing

	fPairs = facePairs()
	trgFaces = cubeFaces(slfVol.scale)
	srcFaces = cubeFaces(slfVol.scale)
	glQuadFar = gaussQuad2(assemblyInfo.ordGLIntFar)
	glQuadMed = gaussQuad2(assemblyInfo.ordGLIntMed)
	glQuadNear = gaussQuad2(assemblyInfo.ordGLIntNear)
	sGrid = separationGrid(slfVol, slfVol, 0)
	assembleGreenCircSelf!(greenCirc, slfVol, sGrid, glQuadFar, glQuadMed, 
		glQuadNear, trgFaces, srcFaces, fPairs, assemblyInfo)
	
	return nothing
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
	fPairs::Array{Int64,2}, assemblyInfo::MaxGAssemblyOpts)::Nothing

	indSplit1 = trgVol.cells[1]
	indSplit2 = trgVol.cells[2]
	indSplit3 = trgVol.cells[3]

	greenExt! = (greenMat::SubArray{ComplexF64,2}, ind1::Int64, ind2::Int64, 
		ind3::Int64) -> 
	greenExtFunc!(greenMat, ind1, ind2, ind3, indSplit1, indSplit2, indSplit3, 
		sGridT, sGridS, glQuadFar, glQuadMed, glQuadNear, trgVol.scale, 
		srcVol.scale, trgFaces, srcFaces, fPairs, assemblyInfo)
	threadArrayW!(greenCirc, 3, size(greenCirc), greenExt!)
	
	return nothing
end
"""
Generate Green function self interaction circulant vector.
"""
function assembleGreenCircSelf!(greenCirc::Array{ComplexF64}, slfVol::MaxGVol, 
	sGrid::Array{<:StepRangeLen,1}, glQuadFar::Array{Float64,2}, 
	glQuadMed::Array{Float64,2}, glQuadNear::Array{Float64,2}, 
	trgFaces::Array{Float64,3}, srcFaces::Array{Float64,3}, 
	fPairs::Array{Int64,2}, assemblyInfo::MaxGAssemblyOpts)::Nothing

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
	# Return order of normal faces xx yy zz
	wS = (^(prod(slfVol.scale), -1) .* 
		weakS(slfVol.scale, glQuadNear1, assemblyInfo))
	# Return order of normal faces xxY xxZ yyX yyZ zzX zzY xy xz yz
	wE = (^(prod(slfVol.scale), -1) .* 
		weakE(slfVol.scale, glQuadNear1, assemblyInfo))
	# Return order of normal faces xx yy zz xy xz yz 
	wV = (^(prod(slfVol.scale), -1) .* 
		weakV(slfVol.scale, glQuadNear1, assemblyInfo))
	# Correct singular integrals for coincident and adjacent cells.
	greenFunc! = (greenMat::SubArray{ComplexF64,2}, ind1::Int64, ind2::Int64, 
		ind3::Int64) -> greenSingFunc!(greenMat, ind1, ind2, ind3, wS, wE, wV, 
		sGrid, glQuadNear1, glQuadNear, slfVol.scale, trgFaces, srcFaces, 
		fPairs, assemblyInfo)
	threadArrayW!(greenToe, 3, (3, 3, min(slfVol.cells[1], 2), 
		min(slfVol.cells[2], 2), min(slfVol.cells[3], 2)), greenFunc!)
	# Embed self Green function into a circulant form
	indSplit1 = div(size(greenCirc)[3], 2)
	indSplit2 = div(size(greenCirc)[4], 2)
	indSplit3 = div(size(greenCirc)[5], 2)
	embedFunc = (greenMat::SubArray{ComplexF64,2}, ind1::Int64, ind2::Int64, 
		ind3::Int64) -> greenToeToCirc!(greenMat, greenToe, ind1, ind2, ind3, 
		indSplit1, indSplit2, indSplit3)
	threadArrayW!(greenCirc, 3, size(greenCirc), embedFunc)
	return nothing
end
"""
Generate circulant self Green function from Toeplitz self Green function. The 
implemented mask takes into account the relative flip in the assumed dipole 
direction under a coordinate reflection. 
"""
function greenToeToCirc!(greenCirc::SubArray{ComplexF64,2}, 
	greenToe::Array{ComplexF64}, ind1::Int64, ind2::Int64, ind3::Int64, 
	indSplit1::Int64, indSplit2::Int64, indSplit3::Int64)::Nothing
	
	fi = indFlip(ind1, indSplit1)
	fj = indFlip(ind2, indSplit2)
	fk = indFlip(ind3, indSplit3)

	greenCirc .= (greenToe[:, :, indSelect(ind1, indSplit1), 
	indSelect(ind2, indSplit2), indSelect(ind3, indSplit3)] 
	# .* 
	# (convert(Array{ComplexF64}, [
	# 	1.0       (fi * fj) (fi * fk) 
	# 	(fj * fi) 1.0       (fj * fk)
	# 	(fk * fi) (fk * fj) 1.0]))
	)

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
	glQuadMed::Array{Float64,2}, glQuadNear::Array{Float64,2}, 
	scaleT::NTuple{3,Float64}, scaleS::NTuple{3,Float64}, 
	trgFaces::Array{Float64,3}, srcFaces::Array{Float64,3}, 
	fPairs::Array{Int64,2}, assemblyInfo::MaxGAssemblyOpts)::Nothing
	
	greenFunc!(greenMat, gridSelect(ind1, indSplit1, 1, sGridT, sGridS), 
		gridSelect(ind2, indSplit2, 2, sGridT, sGridS), 
		gridSelect(ind3, indSplit3, 3, sGridT, sGridS), 
		quadSelect(gridSelect(ind1, indSplit1, 1, sGridT, sGridS), 
			gridSelect(ind2, indSplit2, 2, sGridT, sGridS), 
			gridSelect(ind3, indSplit3, 3, sGridT, sGridS), 
			min(minimum(scaleT), minimum(scaleS)), glQuadFar, glQuadMed, 
			glQuadNear, assemblyInfo), 
		scaleT, scaleS, trgFaces, srcFaces, fPairs, assemblyInfo)
end
"""
Write Green element for a pair of cubes in a common domain. 
"""
@inline function greenSlfFunc!(greenMat::SubArray{ComplexF64}, ind1::Int64, 
	ind2::Int64, ind3::Int64, sGrid::Array{<:StepRangeLen,1}, 
	glQuadFar::Array{Float64,2}, glQuadMed::Array{Float64,2}, 
	glQuadNear::Array{Float64,2}, scale::NTuple{3,Float64}, 
	trgFaces::Array{Float64,3}, srcFaces::Array{Float64,3}, 
	fPairs::Array{Int64,2}, assemblyInfo::MaxGAssemblyOpts)::Nothing
	
	greenFunc!(greenMat, sGrid[1][ind1], sGrid[2][ind2], sGrid[3][ind3], 
		quadSelect(sGrid[1][ind1], sGrid[2][ind2], sGrid[3][ind3], 
			minimum(scale), glQuadFar, glQuadMed, glQuadNear, assemblyInfo), 
		scale, scale, trgFaces, srcFaces, fPairs, assemblyInfo)
end
"""
Write a general Green function element to a shared memory array. 
"""
function greenFunc!(greenMat::SubArray{ComplexF64,2}, gridX::Float64, 
	gridY::Float64, gridZ::Float64, glQuad2::Array{Float64,2}, 
	scaleT::NTuple{3,Float64}, scaleS::NTuple{3,Float64}, 
	trgFaces::Array{Float64,3}, srcFaces::Array{Float64,3}, 
	fPairs::Array{Int64,2}, assemblyInfo::MaxGAssemblyOpts)::Nothing

	surfMat = zeros(ComplexF64, 36)
	# Green function between all cube faces.
	greenSurfs!(gridX, gridY, gridZ, surfMat, glQuad2, trgFaces, srcFaces, 
		fPairs, surfScale(scaleS, scaleT), assemblyInfo)
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
	ind2::Int64, ind3::Int64, wS::Array{ComplexF64,1}, wE::Array{ComplexF64,1},
	wV::Array{ComplexF64,1}, sGrid::Array{<:StepRangeLen,1}, 
	glQuad1::Array{Float64,2}, glQuad2::Array{Float64,2}, 
	scale::NTuple{3,Float64}, trgFaces::Array{Float64,3}, 
	srcFaces::Array{Float64,3}, fPairs::Array{Int64,2}, 
	assemblyInfo::MaxGAssemblyOpts)

	# Uncorrected surface integrals.
	surfMat = zeros(ComplexF64, 36)
	greenSurfs!(sGrid[1][ind1], sGrid[2][ind2], sGrid[3][ind3], surfMat, 
		glQuad2, trgFaces, srcFaces, fPairs, surfScale(scale, scale),  
		assemblyInfo)
	# Face convention yzL yzU (x-faces) xzL xzU (y-faces) xyL xyU (z-faces)
	# Index based corrections.
	if (ind1, ind2, ind3) == (1, 1, 1) 
		
		correctionVal = [
		wS[1]  0.0    wE[7]  wE[7]  wE[8]  wE[8]
		0.0    wS[1]  wE[7]  wE[7]  wE[8]  wE[8]
		wE[7]  wE[7]  wS[2]   0.0   wE[9]  wE[9]
		wE[7]  wE[7]  0.0    wS[2]  wE[9]  wE[9]
		wE[8]  wE[8]  wE[9]  wE[9]  wS[3]  0.0
		wE[8]  wE[8]  wE[9]  wE[9]  0.0    wS[3]]
		
		mask =[
		1 0 1 1 1 1
		0 1 1 1 1 1
		1 1 1 0 1 1
		1 1 0 1 1 1
		1 1 1 1 1 0
		1 1 1 1 0 1]
	
	elseif (ind1, ind2, ind3) == (2, 1, 1) 
		
		correctionVal = [
		0.0  wS[1]  wE[7]  wE[7]  wE[8]  wE[8]
		0.0  0.0    0.0    0.0    0.0    0.0
		0.0  wE[7]  wE[3]  0.0    wV[6]  wV[6]
		0.0  wE[7]  0.0    wE[3]  wV[6]  wV[6]
		0.0  wE[8]  wV[6]  wV[6]  wE[5]  0.0
		0.0  wE[8]  wV[6]  wV[6]  0.0    wE[5]]
		
		mask = [
		0 1 1 1 1 1
		0 0 0 0 0 0
		0 1 1 0 1 1
		0 1 0 1 1 1
		0 1 1 1 1 0
		0 1 1 1 0 1]

	elseif (ind1, ind2, ind3) == (2, 1, 2)
		
		correctionVal = [
		0.0  wE[2]  wV[4]  wV[4]  0.0  wE[8]
		0.0  0.0    0.0    0.0    0.0  0.0
		0.0  wV[4]  wV[2]  0.0    0.0  wV[6]
		0.0  wV[4]  0.0    wV[2]  0.0  wV[6]
		0.0  wE[8]  wV[6]  wV[6]  0.0  wE[5]
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
		wE[2]  0.0    wV[4]  wV[4]  0.0  wE[8]
		0.0    wE[2]  wV[4]  wV[4]  0.0  wE[8]
		wV[4]  wV[4]  wE[4]  0.0    0.0  wE[9]
		wV[4]  wV[4]  0.0    wE[4]  0.0  wE[9]
		wE[8]  wE[8]  wE[9]  wE[9]  0.0  wS[3]
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
		wE[1]  0.0    0.0  wE[7]  wV[5]  wV[5]
		0.0    wE[1]  0.0  wE[7]  wV[5]  wV[5]
		wE[7]  wE[7]  0.0  wS[2]  wE[9]  wE[9]
		0.0    0.0    0.0  0.0    0.0    0.0
		wV[5]  wV[5]  0.0  wE[9]  wE[6]  0.0
		wV[5]  wV[5]  0.0  wE[9]  0.0    wE[6]]
		
		mask = [
		1 0 0 1 1 1
		0 1 0 1 1 1
		1 1 0 1 1 1
		0 0 0 0 0 0
		1 1 0 1 1 0
		1 1 0 1 0 1]

	elseif (ind1, ind2, ind3) == (2, 2, 1) 

		correctionVal = [
		0.0  wE[1]  0.0  wE[7]  wV[5]  wV[5]
		0.0  0.0    0.0  0.0    0.0    0.0
		0.0  wE[7]  0.0  wE[3]  wV[6]  wV[6]
		0.0  0.0    0.0  0.0    0.0    0.0
		0.0  wV[5]  0.0  wV[6]  wV[3]  0.0
		0.0  wV[5]  0.0  wV[6]  0.0    wV[3]]

		mask = [
		0 1 0 1 1 1
		0 0 0 0 0 0
		0 1 0 1 1 1
		0 0 0 0 0 0
		0 1 0 1 1 0
		0 1 0 1 0 1]  

	elseif (ind1, ind2, ind3) == (1, 2, 2) 
		
		correctionVal = [
		wV[1]  0.0    0.0  wV[4]  0.0  wV[5]
		0.0    wV[1]  0.0  wV[4]  0.0  wV[5]
		wV[4]  wV[4]  0.0  wE[4]  0.0  wE[9]
		0.0    0.0    0.0  0.0    0.0  0.0
		wV[5]  wV[5]  0.0  wE[9]  0.0  wE[6]
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
		0.0  wV[1]  0.0  wV[4]  0.0  wV[5]
		0.0  0.0    0.0  0.0    0.0  0.0
		0.0  wV[4]  0.0  wV[2]  0.0  wV[6]
		0.0  0.0    0.0  0.0    0.0  0.0
		0.0  wV[5]  0.0  wV[6]  0.0  wV[3]
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
	surfMat::Array{ComplexF64,1}, glQuad2::Array{Float64,2}, 
	trgFaces::Array{Float64,3}, srcFaces::Array{Float64,3}, 
	fPairs::Array{Int64,2}, srfScales::Array{Float64,1}, 
	assemblyInfo::MaxGAssemblyOpts)::Nothing

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

			surfMat[fp] += glQuad2[iP1,3] * glQuad2[iP2,3] * 
			scaleGreen(distMag(cVX(iP1, iP2, fp) + gridX, cVY(iP1, iP2, fp) + 
				gridY, cVZ(iP1, iP2, fp) + gridZ), assemblyInfo.freqPhase)
		end

		surfMat[fp] *= srfScales[fp]
	end

return nothing
end
"""
Generate all unique pairs of cube faces. 
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
Determine appropriate scaling for surface integrals
"""
function surfScale(scaleS::NTuple{3,Float64}, 
	scaleT::NTuple{3,Float64})::Array{Float64,1}

	srcScaling = 1.0
	trgScaling = 1.0
	srfScales = Array{Float64,1}(undef, 36)

	for srcFId in 1 : 6

		if srcFId == 1 || srcFId == 2

			srcScaling = scaleS[2] * scaleS[3]

		elseif srcFId == 3 || srcFId == 4

			srcScaling = scaleS[1] * scaleS[3]

		else

			srcScaling = scaleS[1] * scaleS[2]
		end

		for trgFId in 1 : 6
			
			if trgFId == 1 || trgFId == 2

				trgScaling = scaleT[1]

			elseif trgFId == 3 || trgFId == 4

				trgScaling = scaleT[2]

			else

				trgScaling = scaleT[3]
			end			

			srfScales[(srcFId - 1) * 6 + trgFId] = srcScaling / trgScaling
		end
	end

	return srfScales
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
Generate array of cuboid faces based from a characteristic size, l[]. 
L and U reference relative positions on the corresponding normal axis.
Points are number in a counter-clockwise convention when viewing the 
face from the exterior of the cube. 
"""
function cubeFaces(size::NTuple{3,Float64})::Array{Float64,3}
	
	yzL = hcat([0.0, 0.0, 0.0], [0.0, size[2], 0.0], [0.0, size[2], size[3]], 
		[0.0, 0.0, size[3]])
	yzU = hcat([size[1], 0.0, 0.0], [size[1], 0.0, size[3]], 
		[size[1], size[2], size[3]], [size[1], size[2], 0.0])
	xzL = hcat([0.0, 0.0, 0.0], [0.0, 0.0, size[3]], [size[1], 0.0, size[3]], 
		[size[1], 0.0, 0.0])
	xzU = hcat([0.0, size[2], 0.0], [size[1], size[2], 0.0], 
		[size[1], size[2], size[3]], [0.0, size[2], size[3]])
	xyL = hcat([0.0, 0.0, 0.0], [size[1], 0.0, 0.0], [size[1], size[2], 0.0], 
		[0.0, size[2], 0.0])
	xyU = hcat([0.0, 0.0, size[3]], [0.0, size[2], size[3]], 
		[size[1], size[2], size[3]], [size[1], 0.0, size[3]])
	
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

		sep = round.(srcVol.scale, digits = 4)

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
		
		sep = round.(trgVol.scale, digits = 4)
		
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
	minScale::Float64, glQuadFar::Array{Float64,2}, glQuadMed::Array{Float64,2}, 
	glQuadNear::Array{Float64,2}, 
	assemblyInfo::MaxGAssemblyOpts)::Array{Float64,2}

	if (distMag(vX, vY, vZ) / minScale) > assemblyInfo.crossMedFar
		
		return glQuadFar

	elseif (distMag(vX, vY, vZ) / minScale) > assemblyInfo.crossNearMed
		
		return glQuadMed
	# The value of 1.733 is slightly larger than sqrt(3). Any smaller separation 
	# between cell centers indicates the presence of a common vertex, edge, 
	# face, or volume. Such cases are treated independently by direct evaluation 
	# methods.
	elseif (distMag(vX, vY, vZ) / minScale) < 1.733

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
end