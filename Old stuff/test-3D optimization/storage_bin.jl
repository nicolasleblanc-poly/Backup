# Green function BA
# print("Green ", greenCircBA[3,3,:,:,:], "\n")
# g_BA = greenCircBA[:,3,:,:,:] # this will get the entries for the z column of the Green's function
# print("g_BA ", g_BA, "\n")
# # 2. Embed ji_x
# ji_embedded_test = embedVec(ji_test, cellsA_test[1], cellsA_test[2], cellsA_test[3], cellsA_test[1]+cellsB_test[1], cellsA_test[2]+cellsB_test[2], cellsA_test[3]+cellsB_test[3])
# print("ji ", size(ji_embedded_test), "\n")
# # 3. Get gxx as a vector
# g_test = zeros(ComplexF64, length(ji_embedded_test), 1)
# index = 1
# # while index <= length(ji_embedded_test)
# # 	g_test[index] = g_BA[index]
# # 	global index += 1
# # end
# for element in g_BA
# 	if index < length(ji_embedded_test)
# 		g_test[index] = element
# 		global index += 1
# 	end
# end
# print("g_test ", g_test, "\n")
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used)
# x=rand(ComplexF64, length(ji_embedded_test), 1) 
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# p_BA = plan_fft(x) # ; flags=FFTW.ESTIMATE, timelimit=Inf
# fft_ji_embedded_test = p_BA*ji_embedded_test
# fft_g_test = p_BA*g_test
# # 5. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_test = fft_g_test .* fft_ji_embedded_test
# # 6. Inverse the result of the previous multiplication
# p_inv_test = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
# fft_inv_test = p_inv_test*mult_test
# # 7. Project the previous result 
# projection_BA = proj(fft_inv_test, cellsA_test[1]+cellsB_test[1], cellsA_test[2]+cellsB_test[2], cellsA_test[3]+cellsB_test[3], cellsB_test[1], cellsB_test[2], cellsB_test[3])
# print("projection_BA ", projection_BA, "\n")
# # print("length(projection_BA) ", length(projection_BA), "\n")
# # Looking at Novotny, the resulting field should be radia (aka in x-y plane) but 
# # it doesn't seem to be. Could it be because A is 1x1x1 with 1 as it's only entry, 
# # since it's 1x1 matrix? genMaxGVol(MaxGDom(cellsB_test, scale_test, coordB_test))
# # Information for Green function construction. 
# # Complex frequency ratio. 
# freqPhase = 1.0 + im * 0.0
# # Gauss-Legendre approximation orders. 
# ordGLIntFar = 2
# ordGLIntMed = 4
# ordGLIntNear = 16
# # Cross over points for Gauss-Legendre approximation.
# crossMedFar = 16
# crossNearMed = 8
# assemblyInfo = MaxGAssemblyOpts(freqPhase, ordGLIntFar, ordGLIntMed, 
# 	ordGLIntNear, crossMedFar, crossNearMed)
# # Pre-allocate memory for circulant green function vector. 
# greenCircAA = Array{ComplexF64}(undef, 3, 3, 2 * cellsA_test[1], 2 * cellsA_test[2], 
# 	2 * cellsA_test[3])
# greenCircBA = Array{ComplexF64}(undef, 3, 3, cellsB_test[1] + cellsA_test[1], cellsB_test[2] +
# 	cellsA_test[2], cellsB_test[3] + cellsA_test[3])
# # CPU computation of Green function
# genGreenExt!(greenCircBA, volB, volA, assemblyInfo)
# # print("test ", greenCircBA[:,3,:,:,:],"\n")
# # Incident field
# # For the test we are only considering the z-direction, so we get a current density 
# # vector that is a column vector of the form (0, 0, 1).
# ji_test = ones(ComplexF64, (cellsA_test[1])*(cellsA_test[2])*(cellsA_test[3]), 1) #double check if this is the right size of the initial ji
# # Green function BA
# print("Green ", greenCircBA[3,3,:,:,:], "\n")
# g_BA = greenCircBA[:,3,:,:,:] # this will get the entries for the z column of the Green's function
# print("g_BA ", g_BA, "\n")
# # 2. Embed ji_x
# ji_embedded_test = embedVec(ji_test, cellsA_test[1], cellsA_test[2], cellsA_test[3], cellsA_test[1]+cellsB_test[1], cellsA_test[2]+cellsB_test[2], cellsA_test[3]+cellsB_test[3])
# print("ji ", size(ji_embedded_test), "\n")
# # 3. Get gxx as a vector
# g_test = zeros(ComplexF64, length(ji_embedded_test), 1)
# index = 1
# # while index <= length(ji_embedded_test)
# # 	g_test[index] = g_BA[index]
# # 	global index += 1
# # end
# for element in g_BA
# 	if index < length(ji_embedded_test)
# 		g_test[index] = element
# 		global index += 1
# 	end
# end
# print("g_test ", g_test, "\n")
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used)
# x=rand(ComplexF64, length(ji_embedded_test), 1) 
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# p_BA = plan_fft(x) # ; flags=FFTW.ESTIMATE, timelimit=Inf
# fft_ji_embedded_test = p_BA*ji_embedded_test
# fft_g_test = p_BA*g_test
# # 5. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_test = fft_g_test .* fft_ji_embedded_test
# # 6. Inverse the result of the previous multiplication
# p_inv_test = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
# fft_inv_test = p_inv_test*mult_test
# # 7. Project the previous result 
# projection_BA = proj(fft_inv_test, cellsA_test[1]+cellsB_test[1], cellsA_test[2]+cellsB_test[2], cellsA_test[3]+cellsB_test[3], cellsB_test[1], cellsB_test[2], cellsB_test[3])
# print("projection_BA ", projection_BA, "\n")
# print("length(projection_BA) ", length(projection_BA), "\n")
# Looking at Novotny, the resulting field should be radia (aka in x-y plane) but 
# it doesn't seem to be. Could it be because A is 1x1x1 with 1 as it's only entry, 
# since it's 1x1 matrix?

# The x^ field vector made up from Gxx, Gyx and Gzx?



# The Green function calls for the AA and BA cases are below. 

# # For the AA case, we have:
# # Let's consider the source in the x (ji_x)
# ji_x = rand(ComplexF64, (cellsA[1])*(cellsA[2])*(cellsA[3]), 1) #double check if this is the right size of the initial ji
# # XX case 
# # 1. Get the xx Green function 
# gxx_m = greenCircAA[1,1,:,:,:]
# # 2. Embed ji_x
# ji_embedded_x = embedVec(ji_x, cellsA[1], cellsA[2], cellsA[3], 2*cellsA[1], 2*cellsA[2], 2*cellsA[3])
# # 3. Get gxx as a vector
# gxx = zeros(ComplexF64, length(ji_embedded_x), 1)
# index = 1
# while index < length(ji_embedded_x)
# 	gxx[index] = gxx_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used)
# x=rand(ComplexF64, length(ji_embedded_x), 1) 
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# p_x = plan_fft(x) # ; flags=FFTW.ESTIMATE, timelimit=Inf
# fft_ji_embedded_x = p_x*ji_embedded_x
# fft_gxx = p_x*gxx
# # 5. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_xx = fft_gxx .* fft_ji_embedded_x
# # 6. Inverse the result of the previous multiplication
# p_inv_x = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
# fft_inv_xx = p_inv_x*mult_xx
# # 7. Project the previous result 
# projection_xx = proj(fft_inv_xx, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_xx", projection_xx, "\n")

# # XY case 
# # 1. Get the xy Green function
# gxy_m = greenCircAA[1,2,:,:,:]
# # 2. Embed ji_x
# # Use the embedded ji found for the xx case 
# # 3. Get gxy as a vector
# gxy = zeros(ComplexF64, length(ji_embedded_x), 1)
# index = 1
# while index < length(ji_embedded_x)
# 	gxy[index] = gxy_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used)
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# # Use the plan and the fft_ji_embedded_x found for the xx case 
# fft_gxy = p_x*gxy
# # 5. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_xy = fft_gxy .* fft_ji_embedded_x
# # 6. Inverse the result of the previous multiplication
# # It's better to take the complex conjugate of the transpose of the plan apparently, so need to 
# # check if it gives the samething as the inv of the plan. 
# # Use the fft plan found for the xx case 
# fft_inv_xy = p_inv_x*mult_xy
# # 7. Project the previous result 
# projection_xy = proj(fft_inv_xy, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_xy", projection_xy, "\n")

# # XZ case 
# # 1. Get the xy Green function
# gxz_m = greenCircAA[1,3,:,:,:]
# # 2. Embed ji_x
# # Use the embedded ji found for the xx case 
# # 3. Get gxz as a vector
# gxz = zeros(ComplexF64, length(ji_embedded_x), 1)
# index = 1
# while index < length(ji_embedded_x)
# 	gxz[index] = gxz_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used)
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# # Use the plan and the fft_ji_embedded_x found for the xx case 
# fft_gxz = p_x*gxz
# # 4. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_xz = fft_gxz .* fft_ji_embedded_x
# # 4. Inverse the result of the previous multiplication
# # It's better to take the complex conjugate of the transpose of the plan apparently, so need to 
# # check if it gives the samething as the inv of the plan. 
# # Use the fft plan found for the xx case 
# fft_inv_xz = p_inv_x*mult_xz
# # 5. Project the previous result 
# projection_xz = proj(fft_inv_xz, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_xz", projection_xz, "\n")

# # Now we can generate the total for x by adding the xx, xy and xz projections 
# x_proj = projection_xx + projection_xy + projection_xz



# # Let's consider the source in the y (ji_y)
# ji_y = rand(ComplexF64, (2*cellsA[1])*(2*cellsA[2])*(2*cellsA[3]), 1) 
# # YX case 
# # 1. Get the xx Green function 
# gyx_m = greenCircAA[2,1,:,:,:]
# # 2. Embed ji_y
# ji_embedded_y = embedVec(ji_y, cellsA[1], cellsA[2], cellsA[3], 2*cellsA[1], 2*cellsA[2], 2*cellsA[3])
# # 3. Get gyx as a vector
# gyx = zeros(ComplexF64, length(ji_embedded_x), 1)
# index = 1
# while index < length(ji_embedded_x)
# 	gyx[index] = gyx_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used)
# x=rand(ComplexF64, length(ji_embedded_y), 1) 
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# p_y = plan_fft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
# fft_ji_embedded_y = p_x*ji_embedded_y
# fft_gyx = p_x*gyx
# # 4. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_yx = fft_gyx .* fft_ji_embedded_y
# # 4. Inverse the result of the previous multiplication
# # It's better to take the complex conjugate of the transpose of the plan apparently, so need to 
# # check if it gives the samething as the inv of the plan. 
# p_inv_y = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf) # inverse fft plan
# fft_inv_yx = p_inv_y*mult_yx
# # 5. Project the previous result 
# projection_yx = proj(fft_inv_yx, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_yx", projection_yx, "\n")

# # YY case 
# # 1. Get the xy Green function
# gyy_m = greenCircAA[2,2,:,:,:]
# # 2. Embed ji_y
# # Use the embedded ji found for the yx case 
# # 3. Get gyy as a vector
# gyy = zeros(ComplexF64, length(ji_embedded_y), 1)
# index = 1
# while index < length(ji_embedded_y)
# 	gyy[index] = gyy_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used) 
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# # Use the plan and the fft_ji_embedded_x found for the xx case 
# fft_gyy = p_y*gyy
# # 4. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_yy = fft_gyy .* fft_ji_embedded_y
# # 4. Inverse the result of the previous multiplication
# # It's better to take the complex conjugate of the transpose of the plan apparently, so need to 
# # check if it gives the samething as the inv of the plan. 
# # Use the fft plan found for the yx case 
# fft_inv_yy = p_inv_y*mult_yy
# # 5. Project the previous result 
# projection_yy = proj(fft_inv_yy, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_yy", projection_yy, "\n")


# # YZ case 
# # 1. Get the xy Green function
# gyz_m = greenCircAA[2,3,:,:,:]
# # 2. Embed ji_y
# # Use the embedded ji found for the xx case 
# # 3. Get gyz as a vector
# gyz = zeros(ComplexF64, length(ji_embedded_y), 1)
# index = 1
# while index < length(ji_embedded_y)
# 	gyz[index] = gyz_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used) 
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# # Use the plan and the fft_ji_embedded_x found for the xx case 
# fft_gyz = p_x*gyz
# # 4. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_yz = fft_gyz .* fft_ji_embedded_y
# # 4. Inverse the result of the previous multiplication
# # Use the fft plan found for the yx case 
# fft_inv_yz = p_inv_y*mult_yz
# # 5. Project the previous result 
# projection_yz = proj(fft_inv_yz, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_yz", projection_yz, "\n")

# # Now we can generate the total for y by adding the yx, yy and yz projections 
# y_proj = projection_yx + projection_yy + projection_yz



# # Let's consider the source in the z (ji_z)
# ji_z = rand(ComplexF64, (2*cellsA[1])*(2*cellsA[2])*(2*cellsA[3]), 1) 
# # ZX case 
# # 1. Get the zx Green function 
# gzx_m = greenCircAA[3,1,:,:,:]
# # 2. Embed ji_z
# ji_embedded_z = embedVec(ji_z, cellsA[1], cellsA[2], cellsA[3], 2*cellsA[1], 2*cellsA[2], 2*cellsA[3])
# # 3. Get gyx as a vector
# gzx = zeros(ComplexF64, length(ji_embedded_z), 1)
# index = 1
# while index < length(ji_embedded_z)
# 	gzx[index] = gzx_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used)
# x=rand(ComplexF64, length(ji_embedded_z), 1) 
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# p_z = plan_fft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
# fft_ji_embedded_z = p_x*ji_embedded_z
# fft_gzx = p_x*gzx
# # 4. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_zx = fft_gzx .* fft_ji_embedded_z
# # 4. Inverse the result of the previous multiplication
# # It's better to take the complex conjugate of the transpose of the plan apparently, so need to 
# # check if it gives the samething as the inv of the plan. 
# p_inv_z = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf) # inverse fft plan
# fft_inv_zx = p_inv_z*mult_zx
# # 5. Project the previous result 
# projection_zx = proj(fft_inv_zx, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_zx", projection_zx, "\n")

# # ZY case 
# # 1. Get the zy Green function
# gzy_m = greenCircAA[3,2,:,:,:]
# # 2. Embed ji_y
# # Use the embedded ji found for the yx case 
# # 3. Get gyy as a vector
# gzy = zeros(ComplexF64, length(ji_embedded_z), 1)
# index = 1
# while index < length(ji_embedded_y)
# 	gzy[index] = gzy_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used) 
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# # Use the plan and the fft_ji_embedded_x found for the xx case 
# fft_gzy = p_z*gzy
# # 4. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_zy = fft_gzy .* fft_ji_embedded_z
# # 4. Inverse the result of the previous multiplication
# # It's better to take the complex conjugate of the transpose of the plan apparently, so need to 
# # check if it gives the samething as the inv of the plan. 
# # Use the fft plan found for the xx case 
# fft_inv_zy = p_inv_z*mult_zy
# # 5. Project the previous result 
# projection_zy = proj(fft_inv_zy, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_zy", projection_zy, "\n")


# # ZZ case 
# # 1. Get the zz Green function
# gzz_m = greenCircAA[3,3,:,:,:]
# # 2. Embed ji_z
# # Use the embedded ji found for the zz case 
# # 3. Get gyz as a vector
# gzz = zeros(ComplexF64, length(ji_embedded_z), 1)
# index = 1
# while index < length(ji_embedded_z)
# 	gzz[index] = gzz_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used) 
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# # Use the plan and the fft_ji_embedded_x found for the xx case 
# fft_gzz = p_x*gzz
# # 4. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_zz = fft_gzz .* fft_ji_embedded_z
# # 4. Inverse the result of the previous multiplication
# # It's better to take the complex conjugate of the transpose of the plan apparently, so need to 
# # check if it gives the samething as the inv of the plan. 
# # Use the fft plan found for the zx case 
# fft_inv_zz = p_inv_z*mult_zz
# # 5. Project the previous result 
# projection_zz = proj(fft_inv_zz, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_zz", projection_zz, "\n")

# # Now we can generate the total for z by adding the zx, zy and zz projections 
# z_proj = projection_zx + projection_zy + projection_zz


# # For the BA case, we have: 
# # Let's consider the source in the x (ji_x)
# ji_x = rand(ComplexF64, (cellsA[1])*(cellsA[2])*(cellsA[3]), 1) #double check if this is the right size of the initial ji
# # XX case 
# # 1. Get the xx Green function 
# gxx_m = greenCircBA[1,1,:,:,:]
# # 2. Embed ji_x
# ji_embedded_x = embedVec(ji_x, cellsA[1], cellsA[2], cellsA[3], 2*cellsA[1], 2*cellsA[2], 2*cellsA[3])
# # 3. Get gxx as a vector
# gxx = zeros(ComplexF64, length(ji_embedded_x), 1)
# index = 1
# while index < length(ji_embedded_x)
# 	gxx[index] = gxx_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used)
# x=rand(ComplexF64, length(ji_embedded_x), 1) 
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# p_x = plan_fft(x) # ; flags=FFTW.ESTIMATE, timelimit=Inf
# fft_ji_embedded_x = p_x*ji_embedded_x
# fft_gxx = p_x*gxx
# # 5. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_xx = fft_gxx .* fft_ji_embedded_x
# # 6. Inverse the result of the previous multiplication
# p_inv_x = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
# fft_inv_xx = p_inv_x*mult_xx
# # 7. Project the previous result 
# projection_xx = proj(fft_inv_xx, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_xx", projection_xx, "\n")

# # XY case 
# # 1. Get the xy Green function
# gxy_m = greenCircBA[1,2,:,:,:]
# # 2. Embed ji_x
# # Use the embedded ji found for the xx case 
# # 3. Get gxy as a vector
# gxy = zeros(ComplexF64, length(ji_embedded_x), 1)
# index = 1
# while index < length(ji_embedded_x)
# 	gxy[index] = gxy_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used)
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# # Use the plan and the fft_ji_embedded_x found for the xx case 
# fft_gxy = p_x*gxy
# # 5. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_xy = fft_gxy .* fft_ji_embedded_x
# # 6. Inverse the result of the previous multiplication
# # It's better to take the complex conjugate of the transpose of the plan apparently, so need to 
# # check if it gives the samething as the inv of the plan. 
# # Use the fft plan found for the xx case 
# fft_inv_xy = p_inv_x*mult_xy
# # 7. Project the previous result 
# projection_xy = proj(fft_inv_xy, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_xy", projection_xy, "\n")

# # XZ case 
# # 1. Get the xy Green function
# gxz_m = greenCircBA[1,3,:,:,:]
# # 2. Embed ji_x
# # Use the embedded ji found for the xx case 
# # 3. Get gxz as a vector
# gxz = zeros(ComplexF64, length(ji_embedded_x), 1)
# index = 1
# while index < length(ji_embedded_x)
# 	gxz[index] = gxz_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used)
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# # Use the plan and the fft_ji_embedded_x found for the xx case 
# fft_gxz = p_x*gxz
# # 4. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_xz = fft_gxz .* fft_ji_embedded_x
# # 4. Inverse the result of the previous multiplication
# # It's better to take the complex conjugate of the transpose of the plan apparently, so need to 
# # check if it gives the samething as the inv of the plan. 
# # Use the fft plan found for the xx case 
# fft_inv_xz = p_inv_x*mult_xz
# # 5. Project the previous result 
# projection_xz = proj(fft_inv_xz, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_xz", projection_xz, "\n")

# # Now we can generate the total for x by adding the xx, xy and xz projections 
# x_proj = projection_xx + projection_xy + projection_xz



# # Let's consider the source in the y (ji_y)
# ji_y = rand(ComplexF64, (2*cellsA[1])*(2*cellsA[2])*(2*cellsA[3]), 1) 
# # YX case 
# # 1. Get the xx Green function 
# gyx_m = greenCircAA[2,1,:,:,:]
# # 2. Embed ji_y
# ji_embedded_y = embedVec(ji_y, cellsA[1], cellsA[2], cellsA[3], 2*cellsA[1], 2*cellsA[2], 2*cellsA[3])
# # 3. Get gyx as a vector
# gyx = zeros(ComplexF64, length(ji_embedded_x), 1)
# index = 1
# while index < length(ji_embedded_x)
# 	gyx[index] = gyx_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used)
# x=rand(ComplexF64, length(ji_embedded_y), 1) 
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# p_y = plan_fft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
# fft_ji_embedded_y = p_x*ji_embedded_y
# fft_gyx = p_x*gyx
# # 4. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_yx = fft_gyx .* fft_ji_embedded_y
# # 4. Inverse the result of the previous multiplication
# # It's better to take the complex conjugate of the transpose of the plan apparently, so need to 
# # check if it gives the samething as the inv of the plan. 
# p_inv_y = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf) # inverse fft plan
# fft_inv_yx = p_inv_y*mult_yx
# # 5. Project the previous result 
# projection_yx = proj(fft_inv_yx, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_yx", projection_yx, "\n")

# # YY case 
# # 1. Get the xy Green function
# gyy_m = greenCircAA[2,2,:,:,:]
# # 2. Embed ji_y
# # Use the embedded ji found for the yx case 
# # 3. Get gyy as a vector
# gyy = zeros(ComplexF64, length(ji_embedded_y), 1)
# index = 1
# while index < length(ji_embedded_y)
# 	gyy[index] = gyy_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used) 
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# # Use the plan and the fft_ji_embedded_x found for the xx case 
# fft_gyy = p_y*gyy
# # 4. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_yy = fft_gyy .* fft_ji_embedded_y
# # 4. Inverse the result of the previous multiplication
# # It's better to take the complex conjugate of the transpose of the plan apparently, so need to 
# # check if it gives the samething as the inv of the plan. 
# # Use the fft plan found for the yx case 
# fft_inv_yy = p_inv_y*mult_yy
# # 5. Project the previous result 
# projection_yy = proj(fft_inv_yy, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_yy", projection_yy, "\n")


# # YZ case 
# # 1. Get the xy Green function
# gyz_m = greenCircAA[2,3,:,:,:]
# # 2. Embed ji_y
# # Use the embedded ji found for the xx case 
# # 3. Get gyz as a vector
# gyz = zeros(ComplexF64, length(ji_embedded_y), 1)
# index = 1
# while index < length(ji_embedded_y)
# 	gyz[index] = gyz_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used) 
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# # Use the plan and the fft_ji_embedded_x found for the xx case 
# fft_gyz = p_x*gyz
# # 4. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_yz = fft_gyz .* fft_ji_embedded_y
# # 4. Inverse the result of the previous multiplication
# # Use the fft plan found for the yx case 
# fft_inv_yz = p_inv_y*mult_yz
# # 5. Project the previous result 
# projection_yz = proj(fft_inv_yz, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_yz", projection_yz, "\n")

# # Now we can generate the total for y by adding the yx, yy and yz projections 
# y_proj = projection_yx + projection_yy + projection_yz



# # Let's consider the source in the z (ji_z)
# ji_z = rand(ComplexF64, (2*cellsA[1])*(2*cellsA[2])*(2*cellsA[3]), 1) 
# # ZX case 
# # 1. Get the zx Green function 
# gzx_m = greenCircAA[3,1,:,:,:]
# # 2. Embed ji_z
# ji_embedded_z = embedVec(ji_z, cellsA[1], cellsA[2], cellsA[3], 2*cellsA[1], 2*cellsA[2], 2*cellsA[3])
# # 3. Get gyx as a vector
# gzx = zeros(ComplexF64, length(ji_embedded_z), 1)
# index = 1
# while index < length(ji_embedded_z)
# 	gzx[index] = gzx_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used)
# x=rand(ComplexF64, length(ji_embedded_z), 1) 
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# p_z = plan_fft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
# fft_ji_embedded_z = p_x*ji_embedded_z
# fft_gzx = p_x*gzx
# # 4. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_zx = fft_gzx .* fft_ji_embedded_z
# # 4. Inverse the result of the previous multiplication
# # It's better to take the complex conjugate of the transpose of the plan apparently, so need to 
# # check if it gives the samething as the inv of the plan. 
# p_inv_z = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf) # inverse fft plan
# fft_inv_zx = p_inv_z*mult_zx
# # 5. Project the previous result 
# projection_zx = proj(fft_inv_zx, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_zx", projection_zx, "\n")

# # ZY case 
# # 1. Get the zy Green function
# gzy_m = greenCircAA[3,2,:,:,:]
# # 2. Embed ji_y
# # Use the embedded ji found for the yx case 
# # 3. Get gyy as a vector
# gzy = zeros(ComplexF64, length(ji_embedded_z), 1)
# index = 1
# while index < length(ji_embedded_y)
# 	gzy[index] = gzy_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used) 
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# # Use the plan and the fft_ji_embedded_x found for the xx case 
# fft_gzy = p_z*gzy
# # 4. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_zy = fft_gzy .* fft_ji_embedded_z
# # 4. Inverse the result of the previous multiplication
# # It's better to take the complex conjugate of the transpose of the plan apparently, so need to 
# # check if it gives the samething as the inv of the plan. 
# # Use the fft plan found for the xx case 
# fft_inv_zy = p_inv_z*mult_zy
# # 5. Project the previous result 
# projection_zy = proj(fft_inv_zy, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_zy", projection_zy, "\n")


# # ZZ case 
# # 1. Get the zz Green function
# gzz_m = greenCircAA[3,3,:,:,:]
# # 2. Embed ji_z
# # Use the embedded ji found for the zz case 
# # 3. Get gyz as a vector
# gzz = zeros(ComplexF64, length(ji_embedded_z), 1)
# index = 1
# while index < length(ji_embedded_z)
# 	gzz[index] = gzz_m[index]
# 	global index += 1
# end
# # 4. Take the fft of ji_embedded (TBD what type of fft should be used) 
# # I think we could just use ji_embedded instead of x (since we just need it for the shape of the
# # shape of the linear operator computed by the fft) but to be safe I'll use x 
# # Use the plan and the fft_ji_embedded_x found for the xx case 
# fft_gzz = p_x*gzz
# # 4. Generate the circulent Green function and multiply it  
# # by the fft of the embeddent source current 
# mult_zz = fft_gzz .* fft_ji_embedded_z
# # 4. Inverse the result of the previous multiplication
# # It's better to take the complex conjugate of the transpose of the plan apparently, so need to 
# # check if it gives the samething as the inv of the plan. 
# # Use the fft plan found for the zx case 
# fft_inv_zz = p_inv_z*mult_zz
# # 5. Project the previous result 
# projection_zz = proj(fft_inv_zz, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
# print("projection_zz", projection_zz, "\n")

# # Now we can generate the total for z by adding the zx, zy and zz projections 
# z_proj = projection_zx + projection_zy + projection_zz