# When considering A as a source and B as a target (BA situation), we have circCells = cellsA + cellsB 
# and the sourceCells is those of A. We will also have targetCells is those of B. 
# When considering A as a source and A as a target (AA situation), we have circCells = 2*cellsA and 
# the sourceCells is those of A. We will also have targetCells is those of A. 
# circcells A + B
# target B
# fft_inv: inverse of the multiplication of G and ji_embedded
# project: the projection of fft_inv
module Projection
export proj
# Don't forget to add Threads.@threads
# embedding function using 3D direct fft and inverse fft 
function proj(fft_inv, circCellsX, circCellsY, circCellsZ, targetCellsX, targetCellsY, targetCellsZ)
    numCellsCrc = circCellsX*circCellsY*circCellsZ 
    project= zeros(ComplexF64, targetCellsX*targetCellsY*targetCellsZ, 1) 
    Threads.@threads for i in range(start = 0,stop = numCellsCrc-1, step = 1) 
        cellX = mod(i,circCellsX)
        cellY = div(mod(i - cellX, circCellsX * circCellsY), circCellsX)
	    cellZ = div(i - cellX - cellY * circCellsX, circCellsX * circCellsY)
        # cellY = mod(i-cellX, (circCellsX*circCellsY)/circCellsX)
        # cellZ = (i-cellX-cellY*circCellsX)/(circCellsX*circCellsY)
        if cellX < targetCellsX && cellY < targetCellsY && cellZ < targetCellsZ
            # print("In projection \n")
            celllndTrg = cellX + (cellY*targetCellsX)+(cellZ*targetCellsX*targetCellsY)
            project[celllndTrg+1] = fft_inv[i+1]
        end
    end
    return project 
end
end


# embedding function using 1D direct fft and inverse fft 
# function proj(fft_inv, circCellsX, circCellsY, circCellsZ, targetCellsX, targetCellsY, targetCellsZ)
#     numCellsCrc = circCellsX*circCellsY*circCellsZ 
#     project= zeros(ComplexF64, targetCellsX*targetCellsY*targetCellsZ, 1) 
#     Threads.@threads for i in range(start = 0,stop = numCellsCrc-1, step = 1) 
#         cellX = mod(i,circCellsX)
#         cellY = div(mod(i - cellX, circCellsX * circCellsY), circCellsX)
# 	    cellZ = div(i - cellX - cellY * circCellsX, circCellsX * circCellsY)
#         # cellY = mod(i-cellX, (circCellsX*circCellsY)/circCellsX)
#         # cellZ = (i-cellX-cellY*circCellsX)/(circCellsX*circCellsY)
#         if cellX < targetCellsX && cellY < targetCellsY && cellZ < targetCellsZ
#             # print("In projection \n")
#             celllndTrg = cellX + (cellY*targetCellsX)+(cellZ*targetCellsX*targetCellsY)
#             project[celllndTrg+1] = fft_inv[i+1]
#         end
#     end
#     return project 
# end
# end