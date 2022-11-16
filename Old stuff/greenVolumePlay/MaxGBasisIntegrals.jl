"""
Conventions for the values returned by the weak functions. Small letters 
correspond to normal face directions; capital letters correspond to grid 
increment directions. 

Self 	xx  yy  zz
     	1   2   3


Edge 	xxY xxZ yyX yyZ zzX zzY xy xz yz
     	1   2   3   4   5   6   7  8  9     


Vertex 	xx  yy  zz  xy  xz  yz 
       	1   2   3   4   5   6
"""

"""
MaxGBasisIntegrals contains all necessary support functions for  numerical 
integration of the electromagnetic Green function. This code is translated from 
DIRECTFN_E by Athanasios Polimeridis, and is distributed under the GNU LGPL.

Author: Sean Molesky

Reference: Polimeridis AG, Vipiana F, Mosig JR, Wilton DR. 
DIRECTFN: Fully numerical algorithms for high precision computation of singular 
integrals in Galerkin SIE methods. 
IEEE Transactions on Antennas and Propagation. 2013; 61(6):3112-22.

In what follows the word weak is used in reference to the fact the form of the 
scalar Green function is weakly singular: the integrand exhibits a singularity 
proportional to the inverse of the separation distance. The letters S, E and 
V refer, respectively, to integration over self-adjacent triangles, 
edge-adjacent triangles, and vertex-adjacent triangles. 

The article cited above contains useful error comparison plots for the number 
evaluation points considered. 
"""
module MaxGBasisIntegrals
using LinearAlgebra, MaxGStructs
export weakS, weakE, weakV, gaussQuad2, gaussQuad1, scaleGreen, scaleGreen2, distMag

const π = 3.14159265358979323846264338328
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

weakS(scale::NTuple{3,Float64}, glQuad1::Array{Float64,2}, 
assemblyOpts::MaxGAssemblyOpts)::Array{ComplexF64,1}

Head function for integration over coincident square panels. The scale 
vector contains the characteristic lengths of a cuboid voxel relative to the 
wavelength. glQuad1 is an array of Gauss-Legendre quadrature weights and 
positions. The assemblyOps parameter determines the level of precision used for 
integral calculations. Namely, assemblyOpts.ordGLIntNear is used internally in 
all weakly singular integral computations. 
"""
function weakS(scale::NTuple{3,Float64}, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::Array{ComplexF64,1}

	grdPts = Array{Float64}(undef, 3, 18)
	# Weak self integrals for the three characteristic faces of a cuboid. 
	# dir = 1 -> xy face (z-nrm)   dir = 2 -> xz face (y-nrm) 
	# dir = 3 -> yz face (x-nrm)
	return [weakSDir(3, scale, grdPts, glQuad1, assemblyOpts);
	weakSDir(2, scale, grdPts, glQuad1, assemblyOpts); 
	weakSDir(1, scale, grdPts, glQuad1, assemblyOpts)]
end
# Weak self-integral of a particular face.
function weakSDir(dir::Int, scale::NTuple{3,Float64}, grdPts::Array{Float64,2},
	glQuad1::Array{Float64,2}, assemblyOpts::MaxGAssemblyOpts)::ComplexF64

	weakGridPts!(dir, scale, grdPts)

	return (((
	weakSInt(hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5]), glQuad1, 
		assemblyOpts) +
	weakSInt(hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4]), glQuad1, 
		assemblyOpts) +
	weakEInt(hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
	 	grdPts[:,1], grdPts[:,5], grdPts[:,4]), glQuad1, 
	assemblyOpts) +
	weakEInt(hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
		grdPts[:,1], grdPts[:,2], grdPts[:,5]), glQuad1, 
	assemblyOpts)) + (
	weakSInt(hcat(grdPts[:,4], grdPts[:,1], grdPts[:,2]), glQuad1, 
		assemblyOpts) +
	weakSInt(hcat(grdPts[:,4], grdPts[:,2], grdPts[:,5]), glQuad1, 
		assemblyOpts) +
	weakEInt(hcat(grdPts[:,4], grdPts[:,1], grdPts[:,2], 
	 	grdPts[:,4], grdPts[:,2], grdPts[:,5]), glQuad1, 
	assemblyOpts) +
	weakEInt(hcat(grdPts[:,4], grdPts[:,2], grdPts[:,5], 
		grdPts[:,4], grdPts[:,1], grdPts[:,2]), glQuad1, 
	assemblyOpts))) / 2.0)

end
"""

weakE(scale::NTuple{3,Float64}, glQuad1::Array{Float64,2}, 
assemblyOpts::MaxGAssemblyOpts)::Array{ComplexF64,1}	

Head function for integration over edge adjacent square panels. See weakS for 
input parameter descriptions. 
"""
function weakE(scale::NTuple{3,Float64}, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::Array{ComplexF64,1}
	
	grdPts = Array{Float64}(undef, 3, 18)
	# Labels are panelDir-panelDir-gridIncrement
	vals = weakEDir(3, scale, grdPts, glQuad1, assemblyOpts)
	# The case letters reference the normal directions of the rectangles.
	# The upper case letter references increasing axis direction. 
	xxY = vals[1]
	xxZ = vals[3]
	xyA = vals[2]
	xzA = vals[4]

	vals = weakEDir(2, scale, grdPts, glQuad1, assemblyOpts)

	yyZ = vals[1]
	yyX = vals[3]
	yzA = vals[2]
	xyB = vals[4]

	vals = weakEDir(1, scale, grdPts, glQuad1, assemblyOpts)

	zzX = vals[1]
	zzY = vals[3]
	xzB = vals[2]
	yzB = vals[4]

	return [xxY; xxZ; yyX; yyZ; zzX; zzY; (xyA + xyB) / 2.0; (xzA + xzB) / 2.0;
	 (yzA + yzB) / 2.0]
end
#= Weak edge integrals for a given face as specified by dir.

	dir = 1 -> z face -> [y-edge (++ gridX): zz(x), xz(x); 
						  x-edge (++ gridY) zz(y) yz(y)]

	dir = 2 -> y face -> [x-edge (++ gridZ): yy(z), yz(z); 
						  z-edge (++ gridX) yy(x) xy(x)]

	dir = 3 -> x face -> [z-edge (++ gridY): xx(y), xy(y); 
						  y-edge (++ gridZ) xx(z) xz(z)]
=#
function weakEDir(dir::Int, scale::NTuple{3,Float64}, grdPts::Array{Float64,2},
	glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::Array{ComplexF64,1}

	grdPts = Array{Float64}(undef, 3, 18)
	weakGridPts!(dir, scale, grdPts) 

	return [weakEInt(hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5],
	grdPts[:, 2], grdPts[:, 3], grdPts[:, 5]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 3], grdPts[:, 6], grdPts[:, 5]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4],
	grdPts[:, 2], grdPts[:, 3], grdPts[:, 5]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 3], grdPts[:, 6], grdPts[:, 5]), glQuad1, assemblyOpts);
	weakEInt(hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5],
	grdPts[:, 2], grdPts[:, 11], grdPts[:, 5]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 11], grdPts[:, 14], grdPts[:, 5]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4],
	grdPts[:, 2], grdPts[:, 11], grdPts[:, 5]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 11], grdPts[:, 14], grdPts[:, 5]), glQuad1, assemblyOpts);
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 4], grdPts[:, 5], grdPts[:, 7]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 5], grdPts[:, 8], grdPts[:, 7]), glQuad1, assemblyOpts) +
	weakEInt(hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 4], grdPts[:, 5], grdPts[:, 7]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 5], grdPts[:, 8], grdPts[:, 7]), glQuad1, assemblyOpts);
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 4], grdPts[:, 5], grdPts[:, 13]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 5], grdPts[:, 14], grdPts[:, 13]), glQuad1, assemblyOpts) +
	weakEInt(hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 4], grdPts[:, 5], grdPts[:, 13]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 5], grdPts[:, 14], grdPts[:, 13]), glQuad1, assemblyOpts)]
end
"""

weakV(scale::NTuple{3,Float64}, glQuad1::Array{Float64,2}, 
assemblyOpts::MaxGAssemblyOpts)::Tuple{ComplexF64,ComplexF64}	

Head function returning integral values for the Green function over vertex 
adjacent square panels. See weakS for input parameter descriptions. 
"""
function weakV(scale::NTuple{3,Float64}, glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::Array{ComplexF64,1}

	grdPts = Array{Float64}(undef, 3, 18)
	# Vertex integrals for x-normal face.
	vals = weakVDir(3, scale, grdPts, glQuad1, assemblyOpts)
	xxO = vals[1]
	xyA = vals[2]
	xzA = vals[3]
	# Vertex integrals for y-normal face.
	vals = weakVDir(2, scale, grdPts, glQuad1, assemblyOpts)
	yyO = vals[1]
	yzA = vals[2]
	xyB = vals[3]
	# Vertex integrals for z-normal face.
	vals = weakVDir(1, scale, grdPts, glQuad1, assemblyOpts)
	zzO = vals[1]
	xzB = vals[2]
	yzB = vals[3]
	return[xxO; yyO; zzO; (xyA + xyB) / 2.0; (xzA + xzB) / 2.0; 
	(yzA + yzB) / 2.0]
	
end
#= Weak edge integrals for a given face as specified by dir.
	dir = 1 -> z face -> [zz zx zy]
	dir = 2 -> y face -> [yy yz yx]
	dir = 3 -> x face -> [xx xy xz]
=#
function weakVDir(dir::Int, scale::NTuple{3,Float64}, grdPts::Array{Float64,2},
	glQuad1::Array{Float64,2}, 
	assemblyOpts::MaxGAssemblyOpts)::Array{ComplexF64,1}

	weakGridPts!(dir, scale, grdPts) 

	return [weakVInt(hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 5], grdPts[:, 6], grdPts[:, 9]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 5], grdPts[:, 9], grdPts[:, 8]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 5], grdPts[:, 6], grdPts[:, 9]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 5], grdPts[:, 9], grdPts[:, 8]), glQuad1, assemblyOpts);
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 5], grdPts[:, 17], grdPts[:, 14]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 5], grdPts[:, 8], grdPts[:, 17]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 5], grdPts[:, 17], grdPts[:, 14]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 5], grdPts[:, 8], grdPts[:, 17]), glQuad1, assemblyOpts);
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 5], grdPts[:, 15], grdPts[:, 14]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 2], grdPts[:, 5], 
	grdPts[:, 5], grdPts[:, 6], grdPts[:, 15]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 5], grdPts[:, 15], grdPts[:, 14]), glQuad1, assemblyOpts) +
	weakVInt(hcat(grdPts[:, 1], grdPts[:, 5], grdPts[:, 4], 
	grdPts[:, 5], grdPts[:, 6], grdPts[:, 15]), glQuad1, assemblyOpts)]

end
"""
Create grid point system for calculation for calculation of weakly singular 
integrals. 
"""
function weakGridPts!(dir::Int, scale::NTuple{3,Float64}, 
	grdPts::Array{Float64,2})::Nothing

	if dir == 1

		gridX = scale[1] 
		gridY = scale[2]
		gridZ = scale[3]
	
	elseif dir == 2

		gridX = scale[3] 
		gridY = scale[1]
		gridZ = scale[2]

	elseif dir == 3

		gridX = scale[2] 
		gridY = scale[3]
		gridZ = scale[1]
	else

		error("Invalid direction selection.")

	end

	grdPts[:, 1] = [0.0; 	   	0.0;     	   0.0]
	grdPts[:, 2] = [gridX; 	   	0.0; 	   0.0]
	grdPts[:, 3] = [2.0 * gridX;	0.0; 	   0.0]

	grdPts[:, 4] = [0.0; 		gridY; 	   0.0]
	grdPts[:, 5] = [gridX; 		gridY; 	   0.0]
	grdPts[:, 6] = [2.0 * gridX; 	gridY; 	   0.0]

	grdPts[:, 7] = [0.0; 		2.0 * gridY; 0.0]
	grdPts[:, 8] = [gridX; 		2.0 * gridY; 0.0]
	grdPts[:, 9] = [2.0 * gridX; 	2.0 * gridY; 0.0]

	grdPts[:, 10] = [0.0; 	 	0.0; 	   gridZ]
	grdPts[:, 11] = [gridX; 	 	0.0; 	   gridZ]
	grdPts[:, 12] = [2.0 * gridX; 0.0; 	   gridZ]

	grdPts[:, 13] = [0.0; 		gridY; 	   gridZ]
	grdPts[:, 14] = [gridX; 		gridY; 	   gridZ]
	grdPts[:, 15] = [2.0 * gridX; gridY; 	   gridZ]

	grdPts[:, 16] = [0.0; 		2.0 * gridY; gridZ]
	grdPts[:, 17] = [gridX; 		2.0 * gridY; gridZ]
	grdPts[:, 18] = [2.0 * gridX; 2.0 * gridY; gridZ]

	return nothing
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
end