# When considering A as a source and B as a target, we have circCells = cellsA + cellsB 
# and the sourceCells is those of A. We will also have targetCells is those of B. 
# When considering A as a source and A as a target, we have circCells = 2*cellsA and 
# the sourceCells is those of A. We will also have targetCells is those of A. 
# For AB, circcells A + B
# source: A

# For AA
# circCells 2A
module JiEmbedding
export embedVec
function embedVec(ji, sourceCellsX, sourceCellsY, sourceCellsZ, circCellsX, circCellsY, circCellsZ)
    numElecCrc = Int64(circCellsX*circCellsY*circCellsZ)
    # 1. What are the two first indexes (ind1, ind2) if they aren't x, y or z?
    # 2. How to multiply the embedded ji with the Green function? 
    # 3. 6 words for FRNQT grant
    ji_embedded = zeros(ComplexF64, numElecCrc, 1)
    Threads.@threads for i in range(start = 0, stop=numElecCrc-1, step=Int64(1)) 
        cellX = mod(i,circCellsX)
        # cellY = div(mod(i-cellX, (circCellsX*circCellsY)),circCellsX)
        # cellZ = div((i-cellX-cellY*circCellsX),(circCellsX*circCellsY))
        cellY = div(mod(i - cellX, circCellsX * circCellsY), circCellsX)
	    cellZ = div(i - cellX - cellY * circCellsX, circCellsX * circCellsY)
        if cellX < sourceCellsX && cellY < sourceCellsY && cellZ < sourceCellsZ
            celllndSrc = cellX + (cellY*sourceCellsX)+(cellZ*sourceCellsX*sourceCellsY)
            ji_embedded[i+1] = ji[celllndSrc+1] 
        # else
        #     ji_embedded[Int64(i+1)] = 0.0
        #     ji_embedded[Int64(i+1)] = 0.0
        end
    end
    return ji_embedded 
end
end
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