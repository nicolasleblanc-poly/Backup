# When considering A as a source and B as a target, we have circCells = cellsA + cellsB 
# and the sourceCells is those of A. We will also have targetCells is those of B. 
# When considering A as a source and A as a target, we have circCells = 2*cellsA and 
# the sourceCells is those of A. We will also have targetCells is those of A. 
# For AB, circcells A + B
# source: A

# For AA
# circCells 2A
module Embedding
export embedVec
# Don't forget to add Threads.@threads
# embedding function using 3D (still 1D for now) direct fft and inverse fft 
function embedVec(ji, sourceCellsX, sourceCellsY, sourceCellsZ, circCellsX, circCellsY, circCellsZ)
    numElecCrc = circCellsX*circCellsY*circCellsZ 
    # print("numElecCrc ", numElecCrc, "\n")
    # ji_embedded = zeros(ComplexF64, 9, 1)
    # For the test the code below gives a vector with length 8
    # but the Green function has 9 elements, so let's try with the code above
    ji_embedded = zeros(ComplexF64, numElecCrc, 1)
    # print("ji_embedded ", size(ji_embedded), "\n")
    # print("ji ", size(ji), "\n")
    # Threads.@threads 
    for i in range(start = 0, stop=numElecCrc-1, step=Int64(1)) 
        cellX = mod(i,circCellsX)
        # cellY = div(mod(i-cellX, (circCellsX*circCellsY)),circCellsX)
        # cellZ = div((i-cellX-cellY*circCellsX),(circCellsX*circCellsY))
        cellY = div(mod(i - cellX, circCellsX * circCellsY), circCellsX)
	    cellZ = div(i - cellX - cellY * circCellsX, circCellsX * circCellsY)
        # print("cellX ", cellX, "\n")
        # print("cellY ", cellY, "\n")
        # print("cellZ ", cellZ, "\n")

        if cellX < sourceCellsX && cellY < sourceCellsY && cellZ < sourceCellsZ
            celllndSrc = cellX + (cellY*sourceCellsX)+(cellZ*sourceCellsX*sourceCellsY)
            ji_embedded[i+1] = ji[celllndSrc+1] 
        # else
        #     ji_embedded[Int64(i+1)] = 0.0
        #     ji_embedded[Int64(i+1)] = 0.0
        end
    end
    print("ji_embedded ", ji_embedded, "\n")
    return ji_embedded 
end
end

# https://juliamath.github.io/AbstractFFTs.jl/stable/api/#Public-Interface-1

# embedding function using 1D direct fft and inverse fft 
# function embedVec(ji, sourceCellsX, sourceCellsY, sourceCellsZ, circCellsX, circCellsY, circCellsZ)
#     numElecCrc = circCellsX*circCellsY*circCellsZ 
#     # print("numElecCrc ", numElecCrc, "\n")
#     # ji_embedded = zeros(ComplexF64, 9, 1)
#     # For the test the code below gives a vector with length 8
#     # but the Green function has 9 elements, so let's try with the code above
#     ji_embedded = zeros(ComplexF64, numElecCrc, 1)
#     # print("ji_embedded ", size(ji_embedded), "\n")
#     # print("ji ", size(ji), "\n")
#     # Threads.@threads 
#     for i in range(start = 0, stop=numElecCrc-1, step=Int64(1)) 
#         cellX = mod(i,circCellsX)
#         # cellY = div(mod(i-cellX, (circCellsX*circCellsY)),circCellsX)
#         # cellZ = div((i-cellX-cellY*circCellsX),(circCellsX*circCellsY))
#         cellY = div(mod(i - cellX, circCellsX * circCellsY), circCellsX)
# 	    cellZ = div(i - cellX - cellY * circCellsX, circCellsX * circCellsY)
#         # print("cellX ", cellX, "\n")
#         # print("cellY ", cellY, "\n")
#         # print("cellZ ", cellZ, "\n")

#         if cellX < sourceCellsX && cellY < sourceCellsY && cellZ < sourceCellsZ
#             celllndSrc = cellX + (cellY*sourceCellsX)+(cellZ*sourceCellsX*sourceCellsY)
#             ji_embedded[i+1] = ji[celllndSrc+1] 
#         # else
#         #     ji_embedded[Int64(i+1)] = 0.0
#         #     ji_embedded[Int64(i+1)] = 0.0
#         end
#     end
#     return ji_embedded 
# end
# end
# check if the length of ji_embedded is equal to numElecCrc, since we will do 
# an element-wise multiplication between G and ji_embedded later. 

# Number of cells in the volume. 
# cellsA = [4, 4, 4]
# cellsB = [4, 4, 4]
# # Let's consider the source in the x (ji_x)
# ji_z = rand(ComplexF64, (2*cellsA[1])*(2*cellsA[2])*(2*cellsA[3]), 1) 
# # For the AA case, we have:
# # XX case 
# # 1. Embed ji
# ind1 = 3
# ind2 = 3
# ji_embedded_zz = embedVec(ji_z, cellsA[1], cellsA[2], cellsA[3], 2*cellsA[1], 2*cellsA[2], 2*cellsA[3])