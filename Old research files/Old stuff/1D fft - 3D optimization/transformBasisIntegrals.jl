"""
transformBasisIntegrals evaluates the integrands called by the weakS, weakE, 
and weakV head functions using a series of variable transformations and 
analytic integral evaluations---reducing the four dimensional surface integrals 
performed for ``standard'' cells to one dimensional integrals. No comments are 
included in this low level code, which is simply a julia translation of 
DIRECTFN_E by Athanasios Polimeridis. For a complete description of the steps 
being performed see the article cited above and references included therein. 
"""
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