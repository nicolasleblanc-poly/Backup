# import Pkg
# Pkg.add("FFTW")

# add FFTW
using FFTW
# print("fft: ", fft([0; 1; 2; 1], "\n")
A=[0+0im; 1+1im; 2+2im; 1+1im]
# print("fft(A)", size(A), "\n")
# #FFTW.dct(A[:,:])
# print("fft(A)", fft(A), "\n")
# # print(plan_fft(A; flags=FFTW.ESTIMATE, timelimit=Inf))
# # What type of FFT plan do we want to use?
# # Hartley transform, a real-input DFT with halfcomplex-format output, a discrete sine transform or a discrete cosine transform
print("A",A, "\n")
x=rand(ComplexF64,4)
p = plan_fft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
fftA = p*A
print("p*A",fftA, "\n")
p_inv = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
fft_invA = p_inv*fftA
print("p_inv",fft_invA, "\n")



# print([1 2;3 4], "\n")
# print([1;2;3;4], "\n")

# Steps
# 1. Embed ji
# 2. Take the fft of the embedded ji
# 3. Multiply G by ji 
# 4. Inverse the result of the previous multiplication
# 5. Project the inverse we just calculated

# srcCellX = 4
# srcCellY = 4
# srcCellZ = 4

# crcCellsX = 2 * srcCellX
# crcCellsY = 2 * srcCellY
# crcCellsZ = 2 * srcCellZ

# numCells = crcCellsX * crcCellsY * crcCellsZ

# cellX = 0
# cellY = 0
# cellZ = 0

# for linInd in 0 : (numCells - 1)

# 	global cellX = mod(linInd, crcCellsX) 
# 	global cellY = div(mod(linInd - cellX, crcCellsX * crcCellsY), crcCellsX)
# 	global cellZ = div(linInd - cellX - cellY * crcCellsX, crcCellsX * crcCellsY)

#     print("cellPre ", cellY, "\n")
#     print("cellZ ", cellZ, "\n")

# 	# global cellY = mod(linInd - cellX, crcCellsX * crcCellsY) / crcCellsX
# 	# global cellZ = (linInd - cellX - cellY * crcCellsX) / (crcCellsX * crcCellsY)	

# 	print('<', ' ', cellX, ' ', cellY, ' ', cellZ, ' ', '>')

# 	if ((cellX < srcCellX) && (cellY < srcCellY) && (cellZ < srcCellZ))

# 		print(' ', "Embedded!", "\n")

# 	else

# 		print(' ', "Null!", "\n")
# 	end
# end