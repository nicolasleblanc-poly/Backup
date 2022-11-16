using LinearAlgebra
using SparseArrays
using IterativeSolvers
function get_pml_x(omega, Nx, Ny, Npmlx, dx, m=3, lnR=-20)
    pml_x_Hz = ones(ComplexF64, Nx, Ny)
    pml_x_Ey = ones(ComplexF64, Nx, Ny)
    # print("pml_x_Hz",pml_x_Hz,"\n")
    # print("pml_x_Hz[:Npmlx,:]",pml_x_Hz[1:Npmlx,:],"\n")
    if Npmlx==0
        return pml_x_Hz, pml_x_Ey
    end
    x = range(0,Nx-1,step=1)
    y = range(0,Ny-1,step=1)
    X = first.(Iterators.product(x,y))
    Y = last.(Iterators.product(x,y))
    w_pml = Npmlx * dx
    sigma_max = -(m+1)*lnR / (2*w_pml) / omega
    #define left PML
    x = X[1:Npmlx, :]
    # print("x",x,"\n")
    pml_x_Hz[1:Npmlx,:] = 1.0 ./ (1.0 .+ 1im * sigma_max * ((Npmlx.-x) ./ Npmlx).^m)
     # print("pml_x_Hz[1:Npmlx,:] 3",pml_x_Hz,"\n")
    # print("pml_x_Hz[1:Npmlx,:]",pml_x_Hz[1:Npmlx,:])
    pml_x_Ey[1:Npmlx,:] = 1.0 ./ (1.0 .+ 1im * sigma_max * ((Npmlx.-x.+0.5) / Npmlx).^m)
     # print("pml_x_Hz[1:Npmlx,:] 4",pml_x_Ey[1:Npmlx,:],"\n")
    #define right PML
    x = X[end-Npmlx+1:end,1:end]
    # print("x2",x)
    pml_x_Hz[end-Npmlx+1:end,1:end] = 1.0 ./ (1.0 .+ 1im * sigma_max * ((x.-Nx.+1 .+Npmlx)/Npmlx).^m)
     # print("pml_x_Hz[1:Npmlx,:] 5",pml_x_Hz[end+1-Npmlx:end,1:end],"\n")
    pml_x_Ey[end-Npmlx+1:end,1:end] = 1.0 ./ (1.0 .+ 1im * sigma_max * ((x.-Nx.+1 .+Npmlx.-0.5)/Npmlx).^m)
    # print("pml_x_Hz[1:Npmlx,:] 6",pml_x_Ey[end+1-Npmlx:end,1:end],"\n")
     # print("pml_x_Hz",size(pml_x_Hz),"\n")
     # print("pml_x_Ey",size(pml_x_Ey),"\n")
    return pml_x_Hz, pml_x_Ey
end
function get_pml_y(omega, Nx, Ny, Npmly, dy, m=3, lnR=-20)
    pml_y_Hz = ones(ComplexF64, Nx, Ny)
    pml_y_Ex = ones(ComplexF64, Nx, Ny)
    if Npmly==0
        return pml_y_Hz, pml_y_Ex
    end
    x = range(0,Nx-1,step=1)
    y = range(0,Ny-1,step=1)
    X = first.(Iterators.product(x,y))
    Y = last.(Iterators.product(x,y))
    
    w_pml = Npmly * dy
    sigma_max = -(m+1)*lnR / (2*w_pml) / omega
    #define bottom PML
    y = Y[1:end,1:Npmly] # -1 since the second index is included
    
    pml_y_Hz[1:end,1:Npmly] = 1.0 ./ (1.0 .+ 1im * sigma_max .* ((Npmly.-y)./Npmly).^m)
    pml_y_Ex[1:end,1:Npmly] = 1.0 ./ (1.0 .+ 1im * sigma_max * ((Npmly.-y.-0.5)./Npmly).^m)
    #define top PML
    y = Y[1:end,end+1-Npmly:end]
    pml_y_Hz[1:end,end+1-Npmly:end] = 1.0 ./ (1.0 .+ 1im * sigma_max * ((y.-Ny.+1 .+Npmly)./Npmly).^m)
    pml_y_Ex[1:end,end+1-Npmly:end] = 1.0 ./ (1.0 .+ 1im * sigma_max * ((y.-Ny.+1 .+Npmly.+0.5)./Npmly).^m)
    return pml_y_Hz, pml_y_Ex
end
function build_TE_vac_A(omega, Nx, Ny, Npmlx, Npmly, dx, dy)
    """
    construct TE FDFD system matrix for vacuum
    the ordering of the indices goes (x,y,Hz), (x,y,Ex), (x,y,Ey), (x,y+1,Hz), (x,y+1,Ex) , ...
    for now default periodic boundary conditions with no phase shift
    """
    pml_x_Hz, pml_x_Ey = get_pml_x(omega, Nx, Ny, Npmlx, dx)
    pml_y_Hz, pml_y_Ex = get_pml_y(omega, Nx, Ny, Npmly, dy)
    A_data = []
    A_i = []
    A_j = []  #prepare to construct A matrix in COO format
    for cx in range(0,Nx-1,step=1)
        for cy in range(0,Ny-1,step=1)
            xyind = cx*Ny + cy
            # print("xyind",xyind)
            if cx<Nx-1
                xp1yind = (cx+1)*Ny + cy
            else
                xp1yind = cy
            end
            if cx>0
                xm1yind = (cx-1)*Ny + cy
            else
                xm1yind = (Nx-1)*Ny + cy
            end
            if cy<Ny-1
                xyp1ind = cx*Ny + cy + 1
            else
                xyp1ind = cx*Ny
            end
            if cy>0
                xym1ind = cx*Ny + cy - 1
            else
                xym1ind = cx*Ny + Ny - 1
            end
            Hzind = 3*xyind
            #construct Hz row
            i = Hzind
            # print("A_i  ",A_i,"\n")
            #1
            append!(A_i,i+1)
            append!(A_j,i+1)
            append!(A_data,-1im*omega) #diagonal
            #2
            jEx0 = 3*xym1ind + 1
            jEx1 = i + 1
            append!(A_i,i+1)
            append!(A_j,jEx0+1)
            append!(A_data,pml_y_Hz[cx+1,cy+1]/dy)
            #3 
            append!(A_i,i+1)
            append!(A_j,jEx1+1)
            append!(A_data,-pml_y_Hz[cx+1,cy+1]/dy) #Ex part of curl E term
            #4
            jEy0 = i + 2
            jEy1 = 3*xp1yind + 2
            append!(A_i,i+1)
            append!(A_j,jEy0+1)
            append!(A_data,-pml_x_Hz[cx+1,cy+1]/dx) #Ey part of curl E term
            #5 problem
            append!(A_i,i+1)
            append!(A_j,jEy1+1)
            append!(A_data,pml_x_Hz[cx+1,cy+1]/dx)
            #6   
            #construct Ex row
            i = i+1 #Ex comes after Hz
            append!(A_i,i+1)
            append!(A_j,i+1)
            append!(A_data,1im*omega)
            #7  
            jHz0 = Hzind
            jHz1 = 3*xyp1ind
            append!(A_i,i+1)
            append!(A_j,jHz0+1)
            append!(A_data,-pml_y_Ex[cx+1,cy+1]/dy)
            #8 problem
            append!(A_i,i+1)
            append!(A_j,jHz1+1)
            append!(A_data,pml_y_Ex[cx+1,cy+1]/dy) #Hz curl
            #9 
            #constraint Ey row
            i = i+1 #Ey comes after Ex
            append!(A_i,i+1)
            append!(A_j,i+1)
            append!(A_data,1im*omega)
            #10
            jHz0 = 3*xm1yind
            jHz1 = Hzind
            append!(A_i,i+1)
            append!(A_j,jHz0+1)
            append!(A_data,pml_x_Ey[cx+1,cy+1]/dx)
            #11
            append!(A_i,i+1)
            append!(A_j,jHz1+1)
            append!(A_data,-pml_x_Ey[cx+1,cy+1]/dx) #Hz curl
        end  
    end
    # print("length(A_data)",length(A_data),"\n")
    A = spzeros(ComplexF64, 3*Nx*Ny, 3*Nx*Ny)
    # print("length(A_i)",length(A_data))
    k=1
    # print("A_i  ",A_i,"\n")
    # print("A_j  ",A_j,"\n")
    # print("A_data ",A_data,"\n")
    while k<length(A_data)
        A[A_i[k],A_j[k]]=A_data[k]
        # print("k: ",k," A[A_i[k],A_j[k]] ",A[A_i[k],A_j[k]],"\n")
        k+=1
    end
    # print("A ",A[1,8],"\n")
    return sparse(A)
    # Is the above line the equivalent of the line below in julia?
    # code in python: A = A.tocsr()
end
function get_diagA_from_chigrid(omega, chi_x, chi_y)
    Nx,Ny=shape(chi_x)
    diagA=zeros(ComplexF64, 3*Nx*Ny, 3*Nx*Ny)
    diagA[2:end:3]=1im*omega*Iterators.flatten(chi_x)
    diagA[3:end:3]=1im*omega*Iterators.flatten(chi_y)
    return Diagonal(diagA)
end
function get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx,cy, pol, Qabs, amp=1.0, chigrid=nothing)
    omega = 2*pi/wvlgth * (1 + 1im/2/Qabs)
    A = build_TE_vac_A(omega, Nx, Ny, Npmlx, Npmly, dx, dy)
    # print("A",A,"\n")
    # Is csr just the transpose of the csc?
    # code in python: A = A.tocsr()
    # The line below is my attemp at the line above but it doesn't
    # seem to be correct.
    if  !(chigrid === nothing)
        A += get_diagA_from_chigrid(omega, chigrid, chigrid)
    end
    dim=size(A)
    # print("dim",dim,"\n")
    dimx=dim[1]
    b = zeros(ComplexF64, dimx, 1)
    # print("b",b,"\n")
    # print("size(b)",size(b),"\n")
    xyind = cx*Ny + cy
    b[3*xyind+pol+1] = amp
    x = A\b # The problem here is A is not a square matrix
    # print("x",x,"\n")
    # print("Good until here \n")
    # print("The problem is in the lines below with the reshapes! \n")
    Hzfield = reshape(x[1:3:end],(Nx,Ny))
    Exfield = reshape(x[2:3:end], (Nx,Ny))
    Eyfield = reshape(x[3:3:end], (Nx,Ny))
    # print("Done \n")
    return Hzfield, Exfield, Eyfield
end
function get_Yee_TE_GreenFcn(wvlgth, Gx,Gy, Npmlx, Npmly, dx,dy, Qabs)
    """
    generate Green's function of a domain with shape (Gx,Gy), with a periodicY unit cell of Y height Ny
    Green's function basis ordering: ...(x,y,Ex)...(0,0,Ey),(0,1,Ey),...,(x,y,Ey),...
    Qabs=np.inf 
    """
    gpwx = floor(Int, 1.0/dx)
    gpwy = floor(Int, 1.0/dy)
    Nx = 2*Gx-1 + floor(Int,gpwx/2)+ 2*Npmlx
    Ny = 2*Gy-1 + floor(Int,gpwy/2)+ 2*Npmly
    cx = floor(Int,Nx/2)
    # print("Cx",cx,"\n")
    cy = floor(Int,Ny/2)
    # print("Cy",cy,"\n")
    _, x_Exfield, x_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 1, Qabs)
    _, y_Exfield, y_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 2, Qabs)
    # print("x_Exfield",x_Exfield,"\n")
    
    # The 1 and 2 is the value of the variable pol 
    numCoord = Gx*Gy
    G = zeros(ComplexF64, 2*numCoord, 2*numCoord)
    for ix in range(0,Gx-1,step=1)
        for iy in range(0,Gy-1,step=1)
            x_Ex = x_Exfield[cx-ix+1:cx-ix+Gx,cy-iy+1:cy-iy+Gy]
            x_Ey = x_Eyfield[cx-ix+1:cx-ix+Gx,cy-iy+1:cy-iy+Gy]
            y_Ex = y_Exfield[cx-ix+1:cx-ix+Gx,cy-iy+1:cy-iy+Gy]
            y_Ey = y_Eyfield[cx-ix+1:cx-ix+Gx,cy-iy+1:cy-iy+Gy]
            xyind = ix*Gy + iy+1
            # print("ix",ix,"\n")
            # print("iy",iy,"\n")
            # print("xyind",xyind,"\n")
            
            # print("Iterators.flatten(x_Ex) ", collect(Iterators.flatten(x_Ex)),"\n")
            # print("G ",G,"\n")
            # print("G[1:numCoord,xyind]",size(G[1:numCoord,xyind]),"\n")
            # print("numCoord ",numCoord,"\n")
            # print("xyind ",xyind,"\n")
            
            G[1:numCoord,xyind] .= Iterators.flatten(x_Ex)
            # print("G ",G,"\n")
            G[numCoord+1:end,xyind] .= Iterators.flatten(x_Ey)
            G[1:numCoord,xyind+numCoord] .= Iterators.flatten(y_Ex)
            G[numCoord+1:end,xyind+numCoord] .= Iterators.flatten(y_Ey)
        end
    end
    eta = 1.0 #dimensionless units
    k0 = 2*pi/wvlgth * (1+1im/2/Qabs) / 1
    Gfac = -1im*k0/eta
    G *= Gfac
    #print('check G reciprocity', np.linalg.norm(G-G.T))
    return G
end

# Code to check if the Green's function is good
# wvlgth = 1.0
# Qabs = floor(Int, 1e4) 
# omega = (2*pi/wvlgth) * (1+1im/2/Qabs)
# k0 = omega / 1
# dL = 0.05
# Nx = 10
# Ny = 10
# Npml = 4
# Mx = 2
# My = 2
# Mx0 = floor(Int,(Nx-Mx)/2)
# My0 = floor(Int,(Ny-My)/2)
#check Gdd definiteness
# G = get_Yee_TE_GreenFcn(wvlgth, Mx, My, Npml, Npml, dL, dL, Qabs)
# print("G done",G)
# The check is done and we get the same Green's function as in python :)

function DipoleOverSlab_setup(dx,dy,addx,liste,dL)
    Mx = dx+addx # We will have 3 more pixels on the left side of the region
    # used to generate the Green's function. The dipole will be 
    # situated in this region of length 3 pixels.
    My = dy
    Npml = 15
    dL=0.05
    Qabs = 1e4
    # Obtain the Green's function
    Gbig = get_Yee_TE_GreenFcn(wvlgth, Mx, My, Npml, Npml, dL, dL, Qabs)
    dim=2*dx*dy # 2*dx*dy 
    M=zeros(ComplexF64, dim, dim)
    i=addx*dy+1 # +1 on the index, since julia starts indexing at 1 and not 0.
    icopy=deepcopy(i)
    half_lenGbig = floor(Int, Mx*My)
    j=floor(Int,half_lenGbig)+addx*dy+1
    jcopy=deepcopy(j)
    position=floor(Int, dim/2)
    print("position",position,"\n")
    lenGbig = floor(Int, 2*Mx*My)
    while icopy<half_lenGbig
        while jcopy<lenGbig
            M[1:position, 1:position] .= Gbig[i,i] # xx
            M[position+1:end, 1:position] .= Gbig[i,j] # xy
            M[1:position, position+1:end] .= Gbig[j,i] # yx
            M[position+1:end, position+1:end] .= Gbig[j,j] # yy
            jcopy+=1
        icopy+=1
        end
    end
    # print(size(M))
    dipx=floor(Int,i/2)
    dipy=half_lenGbig+dipx
    u=zeros(ComplexF64, dim, 1)
    v=zeros(ComplexF64, dim, 1)
    u[1:position,1]=Gbig[i:half_lenGbig,dipx]
    u[position+1:end,1]=Gbig[j:lenGbig,dipx]
    v[1:position,1]=Gbig[i:half_lenGbig,dipy]
    v[position+1:end,1]=Gbig[j:lenGbig,dipy]
    alpha=1/(sqrt(2))
    beta=1/(sqrt(2))
    ei=alpha*u+beta*v # this gives a dipole at 45 degrees
    # Pv=[]
    # for element in liste
    #     P=Diagonal(element)
    #     append!(Pv,P)
    # end
    return M,ei #,Pv
end
function Pv(element)
    return Diagonal(element)
end
function C1(T, ei_tr, chi_invdag, Gdag, P) 
    # Left term
    PT=P*T
    # E_tc = np.matrix.conjugate(np.transpose(ei))
    EPT=ei_tr*PT
    I_EPT = imag(EPT) # (1/2)*
    # Right term
    chiGP = P*(chi_invdag - Gdag)
    chiG_A=(chiGP-conj!(transpose(chiGP)))/(2im)
    # M^A = (M+M^dagg)/2 -> j is i in python
    chiGA_T = chiG_A*T
    print("transpose(T)",transpose(T),"\n")
    T_chiGA_T = conj!(transpose(T))*chiGA_T
    print("I_EPT - T_chiGA_T",I_EPT - T_chiGA_T,"\n")
    return I_EPT - T_chiGA_T
end
# Second constraint
function C2(T, ei_tr, chi_invdag, Gdag, P) 
    # Left term
    PT = P*T
    # E_tc = np.matrix.conjugate(np.transpose(ei))
    EPT=ei_tr*PT
    I_EPT = real(EPT) # (1/2)*
    # Right term
    chiGP = P*(chi_invdag - Gdag)
    # M^S = (M+M^dagg)/2 
    chiG_S=(chiGP+conj!(transpose(chiGP)))/2
    chiGS_T = chiG_S*T
    T_chiGS_T = conj!(transpose(T))*chiGS_T
    print("I_EPT - T_chiGS_T",I_EPT - T_chiGS_T,"\n")
    return I_EPT - T_chiGS_T
end
function Am(x,chi_invdag,Gdag,liste) # the A matrix
    # print("floor(Int, sqrt(length(x)))",floor(Int, length(x)),"\n")
    A=zeros(ComplexF64, floor(Int, sqrt(length(chi_invdag))), floor(Int,sqrt(length(chi_invdag))))
    for i in range(1,length(x),step=1)
        P = Pv(liste[ceil(Int,i/2)])
        # print("size(P) ",size(P),"\n")
        # print("chi_invdag - Gdag: ",chi_invdag - Gdag,"\n")
        if mod(i,2) == 1 # Anti-sym
            chiGP = P*(chi_invdag - Gdag)
            # print("*")
            chiG_A=(chiGP-conj!(transpose(chiGP)))/(2im)
            #print(":) \n")
            # print("size(chiG_A)",size(chiG_A),"\n")
            A .+= x[i]*chiG_A
            # print("Out")
        else # Sym -> if mod(i,2) == 0
            chiGP = P*(chi_invdag - Gdag)
            chiG_S=(chiGP+conj!(transpose(chiGP)))/(2)
            A .+= x[i]*chiG_S
            # print("In")
        end
    end
    # print("Exited A \n")
    # print(A)
    return A
end
function bv(x, ei, liste) # the b vector
    # print("In b \n")
    term1 = k0*1.0im/(8*Z)
    b = term1*ei
    for i in range(1,length(x),step=1) 
        # print("In loop")
    #Start at 1 instead of 0 and keep the same end/stop point 
        # print("ceil(Int,i/2)",ceil(Int,i/2),"\n")
        # print("liste", liste, "\n")
        # print("liste[ceil(Int,i/2)]",liste[ceil(Int,i/2)],"\n")
        # P = Pv(liste[ceil(Int,i/2)])
        #term = (1/4)*x[i]*P
        #b .+= term*ei
    end
    # print("Out b \n")
    return b
end
# Code for the different tsolvers
function th_solve(A,b)
    T=A\b 
    return T
end
function cg_solve(A,b)
    T=cg(A,b) 
    return T
end
function bicgstab_solve(A,b)
    T= bicgstabl(A, b)
    return T
end
function gmres_solve(A,b)
    T= gmres(A, b)
    return T
end
tsolvers=Dict("th"=>th_solve, "cg"=>cg_solve, "bicgstab"=>bicgstab_solve,"gmres"=>gmres_solve)
function tsolverpick(name,A,b) 
    return tsolvers[name](A,b)
end
# val = dgfunc(dof, grad, fSlist, tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, get_grad=true)  
function Dual(x,g,fSlist, tsolver, ei, ei_tr, chi_invdag, Gdag, liste, get_grad)
    # print("Entered the dual")
    A=Am(x, chi_invdag, Gdag, liste)
    b=bv(x, ei, liste)
    # print("size(A)", size(A),"\n")
    # print("size(b)", size(b),"\n")
    # print("typeof(A) here",typeof(A),"\n")
    # print("typeof(b) here 2",typeof(b),"\n")
    solve=tsolverpick(tsolver,A,b)
    T=solve
    # print("T",T,"\n")
    g=ones(ComplexF64, length(x), 1)
    for i in range(1,length(x), step=1)
        P = Pv(liste[ceil(Int,i/2)])
        print("i",i,"\n")
        if mod(i,2) == 1 # Asym
            g[i] = C1(T, ei_tr, chi_invdag, Gdag, P)[1]
        else   #Sym
            g[i] = C2(T, ei_tr, chi_invdag, Gdag, P)[1]
        end
    end
    D=0.0+0.0im
    if length(fSlist)==0
        ei_T=ei_tr*T
        D = ((-1/2)*real(ei_T*k0*1.0im/(Z))) .+ 0.0im
        print("D",D,"\n")
        for i in range(1,length(x), step=1)
            print("x[i]*g[i]",x[i]*g[i],"\n")
            D .+= x[i]*g[i]
        end
    else
        if isinstance(fSlist, list)
            f=fSlist[0]
        else # fSlist is an array
            f=fSlist
        end
        A_f = A*f
        f_tr = conj!(transpose(f)) 
        fAf=f_tr*A_f
        ei_T=ei_tr*T
        D = ((-1/2)*real(ei_T*k0*1.0im/(Z)))+fAf
        for i in range(1, length(x), step=1)
            D .+= x[i]*g[i]
        end
    end
    if get_grad == true
        return D, g # D.real
    elseif get_grad == false
        return D # D.real
    end
end
    
function mineigfunc(x, chi_invdag, Gdag, Pv)
    A=Am(x, chi_invdag, Gdag, liste)
    # print(A)
    val=eigen(A)
    w_val=val.values
    v=val.vectors
    # w: eigen values
    # v: eigen vectors
    w=real(w_val)
    w_min=10
    i=1
    for element in w
        if 0<= element <w_min
            w_min=element
            i+=1
        end
        
    end
    # This code is to find the minimum eigenvalue
    # min_eval=w[end:end]
    print("w",w,"\n")
    print("w_min",w_min,"\n")
    # print(min_eval)
    # This code is to find the eigenvector
    # associated with the mimimum eigenvalue
    print("v",v,"\n")
    
    min_evec=reshape(v[i:i,i:i],(sqrt(length(chi_invdag)),1))
    return min_eval,min_evec
end
function validityfunc(x, chi_invdag, Gdag, liste)
    val=mineigfunc(x, chi_invdag, Gdag, liste)
    min_eval=val[0]
    min_evec=val[1]
    if min_eval>0
        return 1
    else
        return -1
    end
end
function BFGS_fakeS_with_restart(initdof, dgfunc, validityfunc, mineigfunc, tsolver, ei, ei_tr, chi_invdag, Gdag, liste, gradConverge=false, opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, min_iter=6)
    print("Entered BFGS")
    dofnum = length(initdof)
    grad = zeros(ComplexF64, dofnum, 1)
    tmp_grad = zeros(ComplexF64, dofnum, 1)
    Hinv = I #approximate inverse Hessian
    prev_dualval = Inf
    dof = initdof
    function justfunc(d)
        return dgfunc(d, np.array([]), fSlist, tsolver, ei, ei_tr, chi_invdag, Gdag, liste, false)
    end
    # justfunc = lambda d: dgfunc(d, np.array([]), fSlist, get_grad=False)
    olddualval = Inf
    reductCount = 0
    while true #gradual reduction of modified source amplitude, outer loop
        fSlist = [] #remove old fake sources and restart with each outer iteration
        alpha_last = 1.0 #reset step size
        Hinv = I #reset Hinv
        val = dgfunc(dof, grad, fSlist, tsolver, ei, ei_tr, chi_invdag, Gdag, liste, true)  
        dualval = val[1]
        grad = val[2]
#             print("grad",grad,"\n")
        # This gets the value of the dual function and the dual's gradient

#             print('\n', 'Outer iteration #', reductCount, 'the starting dual value is', dualval, 'fakeSratio', fakeSratio)
        iternum = 0
        while true
            iternum += 1
#                 print('Outer iteration #', reductCount, 'Inner iteration #', iternum, flush=True)
            Ndir = - Hinv * grad
            pdir = Ndir / norm(Ndir)
            #backtracking line search, impose feasibility and Armijo condition
            p_dot_grad = dot(pdir, grad)
#                 print('pdir dot grad is', p_dot_grad)
            c_reduct = 0.7
            c_A = 1e-4
            c_W = 0.9
            alpha = alpha_start = alpha_last
#                 print('starting alpha', alpha_start)
            while validityfunc(dof+alpha*pdir, chi_invdag, Gdag, liste)<=0 #move back into feasibility region
                alpha *= c_reduct
            end 
            alpha_feas = alpha
#                 print('alpha before backtracking is', alpha_feas)
            alphaopt = alpha
            Dopt = inf
            while True
                tmp_dual = justfunc(dof+alpha*pdir)
                if tmp_dual<Dopt #the dual is still decreasing as we backtrack, continue
                    Dopt = tmp_dual; alphaopt=alpha
                else
                    alphaopt=alpha ###ISSUE!!!!
                    break
                end
                if tmp_dual<=dualval + c_A*alpha*p_dot_grad #Armijo backtracking condition
                    alphaopt = alpha
                    break
                end                      
                alpha *= c_reduct
            end
            added_fakeS = False
            if alphaopt/alpha_start>(c_reduct+1)/2 #in this case can start with bigger step
                alpha_last = alphaopt*2
            else
                alpha_last = alphaopt
                if alpha_feas/alpha_start<(c_reduct+1)/2 && alphaopt/alpha_feas>(c_reduct+1)/2 #this means we encountered feasibility wall and backtracking linesearch didn't reduce step size, we should add fake source
                    added_fakeS = True
#                         print('encountered feasibility wall, adding a fake source term')
                    singular_dof = dof + alpha_feas*pdir #dof that is roughly on duality boundary
                    eig_fct_eval = mineigfunc(singular_dof, chi_invdag, Gdag, liste) #find the subspace of ZTT closest to being singular to target with fake source
                    mineigw=eig_fct_eval[0]
                    mineigv=eig_fct_eval[1]
#                         print("min_evec shape",mineigv.shape,"\n")
#                         print("min_evec type",type(mineigv),"\n")
                    # fakeSval = dgfunc(dof, np.matrix([]), matrix, get_grad=False)
                    fakeSval = dgfunc(dof, np.array([]), [mineigv], tsolver, ei, ei_tr, chi_invdag, Gdag, liste, false)
                    # This only gets us the value of the dual function
                    epsS = sqrt(fakeSratio*abs(dualval/fakeSval))
#                         print('epsS', epsS, '\n')
                    #fSlist.append(np.matmul(epsS,mineigv))
                    append!(fSlist,epsS*mineigv) #add new fakeS to fSlist                                  
                    #fSlist.append(epsS*mineigv) 
#                         print('length of fSlist', len(fSlist))
                 end
              end
#                 print('stepsize alphaopt is', alphaopt, '\n')
            delta = alphaopt * pdir

            #########decide how to update Hinv############
            tmp_val = dgfunc(dof+delta, tmp_grad, fSlist, tsolver, ei, ei_tr, chi_invdag, Gdag, liste, true)
            tmp_dual = tmp_val[0]
            tmp_grad = tmp_val[1]
            # Not sure what the @ does.
            p_dot_tmp_grad = pdir * tmp_grad
            if added_fakeS
                Hinv = I #the objective has been modified; restart Hinv from identity
            elseif p_dot_tmp_grad > c_W*p_dot_grad #satisfy Wolfe condition, update Hinv
#                     print('updating Hinv')
                gamma = tmp_grad - grad
                gamma_dot_delta = alpha * (p_dot_tmp_grad-p_dot_grad)
                Hinv_dot_gamma = Hinv * tmp_grad + Ndir
#                     print("np.outer(delta, Hinv_dot_gamma)",( np.outer(Hinv_dot_gamma, delta) + np.outer(delta, Hinv_dot_gamma) - (1+np.dot(gamma, Hinv_dot_gamma)/gamma_dot_delta)*np.outer(delta, delta) ) / gamma_dot_delta,"\n")
#                     print("Hinv",Hinv,"\n")

                Hinv -= ( np.outer(Hinv_dot_gamma, delta) + np.outer(delta, Hinv_dot_gamma) - (1+np.dot(gamma, Hinv_dot_gamma)/gamma_dot_delta)*np.outer(delta, delta) ) / gamma_dot_delta
            end
            dualval = tmp_dual
            grad[1:end] = tmp_grad[1:end]
            dof += delta

            objval = dualval - dot(dof,grad)
            eqcstval = abs(dof) * abs(grad)
#                 print('at iteration #', iternum, 'the dual, objective, eqconstraint value is', dualval, objval, eqcstval)
#                 print('normgrad is', la.norm(grad))

            if gradConverge && iternum>min_iter && abs(dualval-objval)<opttol*abs(objval) && abs(eqcstval)<opttol*abs(objval) && norm(grad)<opttol * abs(dualval) #objective and gradient norm convergence termination
                break
            end
            if gradConverge==true && iternum>min_iter && abs(dualval-objval)<opttol*abs(objval) && abs(eqcstval)<opttol*abs(objval) #just objective convergence termination, in this case require minimum iternum to allow for adding new constraints with 0 multiplier
                break
            end
            if mod(iternum,iter_period)==0
#                     print('prev_dualval is', prev_dualval)
                if abs(prev_dualval-dualval)<abs(dualval)*opttol #dual convergence / stuck optimization termination
                    break
                end
                prev_dualval = dualval
            end
        end
        if abs(olddualval-dualval)<opttol*abs(dualval)
            break
        end
        """
        #if len(fSlist)<=1 and np.abs(olddualval-dualval)<opttol*np.abs(dualval): #converged w.r.t. reducing fake sources
            #break
        if len(fSlist)==0: #converged without needing fake sources, suggests strong duality
            break
        """
        olddualval = dualval
        reductCount += 1
        fakeSratio *= reductFactor #reduce the magnitude of the fake sources for the next iteration
    end
    return dof, grad, dualval, objval
end
function item(x)
    return x # TBD with Sean
end
function nomad(x)
    return x # TBD with Sean
end
lsolvers=Dict("bfgs"=>BFGS_fakeS_with_restart, "item"=>item, "nomad"=>nomad)
function lsolverpick(name, x, tsolver, ei, ei_tr, chi_invdag, Gdag, liste) 
    return lsolvers[name](x, Dual, validityfunc, mineigfunc, tsolver, ei, ei_tr, chi_invdag, Gdag, liste)
end
function extracted_power_optmization(M, ei, x0, tsolver, lsolver, liste)
    s=floor(Int,sqrt(length(M)))
    chi_coeff = 3.0+2.0im
    chi = chi_coeff*Matrix(I, s, s)
    # print(typeof(I))
    # print("conj!(chi)",conj!(chi),"\n")
    chi_invdag = inv(conj!(transpose(chi)))
    ei_tr = conj!(transpose(ei)) 
    Gdag = conj!(transpose(M))
    print("Solved using the ", tsolver, " method and the ", lsolver, " method")
    # @time 
    x_th=lsolverpick(lsolver, x0, tsolver, ei, ei_tr, chi_invdag, Gdag, liste)
    gradient=x_th[end:end]
    Dual=x_th[end-1:end-1]
    obj=x_th[end-2:end-2]
    print("Gradient value: ", gradient, "\n")
    print("Objective value: ", objective, "\n")

end                                                                                                                                              
Z=376.730313668
wvlgth=1.0
Qabs=1e4
omega=(2*pi/wvlgth)*(1+1im/2/Qabs)
k0=omega/1
dL=0.05
dx=2
dy=2
addx=1
liste=[[1;2;3;4;5;6;7;8],[5;6;7;8;1;2;3;4]]
# liste=[[2,2,2,2,2,2,2,2],[3,3,3,3,3,3,3,3],[2,2,2,2,2,2,2,2],[2,2,2,2,2,2,2,2]]
setup=DipoleOverSlab_setup(dx,dy,addx,liste, dL)
print("setup",setup)
M=setup[1]
ei=setup[2]
x0=ones(ComplexF64, 2*length(liste), 1)
# different tsolvers and bfgs as the lsolver
lsolver="bfgs"
tsolver="th"
opt=extracted_power_optmization(M, ei, x0, tsolver, lsolver, liste)

tsolver="cg"
opt=extracted_power_optmization(M, ei, x0, tsolver, lsolver, liste)

tsolver="bicgstab"
opt=extracted_power_optmization(M, ei, x0, tsolver, lsolver, liste)

tsolver="gmres"
opt=extracted_power_optmization(M, ei, x0, tsolver, lsolver, liste)






function DipoleOverSlab_setup(dx,dy,addx,liste,dL)
    Mx = dx+addx # We will have 3 more pixels on the left side of the region
    # used to generate the Green's function. The dipole will be 
    # situated in this region of length 3 pixels.
    My = dy
    Npml = 2*dx
    # Obtain the Green's function
    Gbig = get_Yee_TE_GreenFcn(wvlgth, Mx, My, Npml, Npml, dL, dL, Qabs)
    dim=2*dx*dy # 2*dx*dy 
    M=zeros(ComplexF64, dim, dim)
    i=addx*dy+1 # +1 on the index, since julia starts indexing at 1 and not 0.
    icopy=deepcopy(i)
    j=floor(Int,len(Gbig)/2)+addx*dy+1
    jcopy=deepcopy(j)
    position=floor(Int,lenght(M)/2)
    half_lenGbig=floor(Int,length(Gbig)/2)
    lenGbig=floor(Int,length(Gbig))
    while icopy<half_lenGbig
        while jcopy<lenGbig
            M[1:position,1:position]=Gbig[i,i] # xx
            M[position:end,1:position]=Gbig[i,j] # xy
            M[1:position,position:end]=Gbig[j,i] # yx
            M[position:end,position:end]=Gbig[j,j] # yy
            jcopy+=1
        icopy+=1
        end
    end
    # print(size(M))
    dipx=floor(Int,i/2)
    dipy=half_lenGbig+dipx
    u=zeros(ComplexF64, dim, 1)
    v=zeros(ComplexF64, dim, 1)
    u[1:position,1]=Gbig[i:half_lenGbig,dipx]
    u[position:end,1]=Gbig[j:lenGbig,dipx]
    v[1:position,1]=Gbig[i:half_lenGbig,dipy]
    v[position:end,1]=Gbig[j:lenGbig,dipy]
    alpha=1/(sqrt(2))
    beta=1/(sqrt(2))
    ei=alpha*u+beta*v # this gives a dipole at 45 degrees
    Pv=[]
    for element in liste
        P=Diagonal(element)
        append!(P_v,P)
    end
    return M,ei,Pv
end

function C1(T,i,ei_tr,chi_invdag,Gdag,Pv)
    # Left term
    P=Pv[i]
    PT=P*T
    # E_tc = np.matrix.conjugate(np.transpose(ei))
    EPT=ei_tr*PT
    I_EPT = imag(EPT) # (1/2)*
    # Right term
    chiGP = P*(chi_invdag - Gdag)
    chiG_A=(chiGP-!conj(transpose(chiGP)))/(2im)
    # M^A = (M+M^dagg)/2 -> j is i in python
    chiGA_T = chiG_A*T
    T_chiGA_T = !conj(transpose(T))*chiGA_T
    return I_EPT - T_chiGA_T
end
# Second constraint
function C2(T, i, ei_tr, chi_invdag, Gdag, Pv)
    # Left term
    P=Pv[i]
    PT=P*T
    # E_tc = np.matrix.conjugate(np.transpose(ei))
    EPT=ei_tr*PT
    I_EPT = real(EPT) # (1/2)*
    # Right term
    chiGP = P*(chi_invdag - Gdag)
    # M^S = (M+M^dagg)/2 
    chiG_S=(chiGP+!conj(transpose(chiGP)))/2
    chiGS_T = chiG_S*T
    T_chiGS_T = !conj(transpose(T))*chiGS_T
    return I_EPT - T_chiGS_T
end
function Am(x,chi_invdag,Gdag,Pv) # the A matrix
    A=0.0+0.0im
    for i in range(1,length(x))
        P=Pv[math.ceil(Int,i/2)]
        if mod(i,2) ==0 # Anti-sym
            chiGP = P*(chi_invdag - Gdag)
            chiG_A=(chiGP-!conjugate(transpose(chiGP)))/(2im)
            A += x[i]*chiG_A
        else # Sym
            chiGP = P*(chi_invdag - Gdag)
            chiG_S=(chiGP+!conj(transpose(chiGP)))/(2)
            A += x[i]*chiG_S
        end
    end
    # print(A)
    return A
end
function bv(x, ei, Pv) # the b vector
    term1 = k0*1.0im/(8*Z)
    b = term1*ei
    for i in range(1,length(x)) 
    #Start at 1 instead of 0 and keep the same end/stop point 
        P=Pv[floor(Int,i/2)]
        term = (1/4)*x[i]*P
        b += term*ei
    end
    return b
end
# Code for the different tsolvers
function th_solve(A,b)
    T=A\b 
end
function cg_solve(A,b)
    T=A\b 
end
function bicgstab_solve(A,b)
    T= bicgstabl(A, b)
end
function gmres_solve(A,b)
    T= gmres(A, b)
end
tsolvers=Dict("th"=>th_solve, "cg"=>cg_solve, "bicgstab"=>bicgstab_solve,"gmres"=>gmres_solve)
function tsolverpick(name,A,b) 
    return tsolvers[name](A,b)
end
function Dual_(x,g,fSlist, tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, get_grad)
    A=Am(x, chi_invdag, Gdag, Pv)
    b=bv(x, ei, Pv)
    solve=tsolverpick[tsolver](A,b)
    T=solve[1]
    g=ones(ComplexF64, length(x), 1)
    for i in range(length(x))
        if mod(i,2) ==0
            g[i] = C1(T,ceil(Int,i/2), ei_tr, chi_invdag, Gdag, Pv) 
        else
            g[i] = C2(T,ceil(Int,i/2), ei_tr, chi_invdag, Gdag, Pv) 
    end
    D=0.0+0.0im
    if len(fSlist)==0
        ei_T=ei_tr*T
        D = ((-1/2)*real(ei_T*k0*1.0im/(Z)))+0.0im
        for i in range(1,length(x))
            D+=x[i]*g[i]
        end
    else
        if isinstance(fSlist, list)
            f=fSlist[0]
        else # fSlist is an array
            f=fSlist
        end
        A_f = A*f
        f_tr = !conj(transpose(f)) 
        fAf=f_tr*A_f
        ei_T=ei_tr*T
        D = ((-1/2)*real(ei_T*k0*1.0im/(Z)))+fAf
        for i in range(length(x))
            D+=x[i]*g[i]
        end
    end
    if get_grad == True
        return D, g # D.real
    elseif get_grad == False
        return D # D.real
    end
end
function mineigfunc(x, chi_invdag, Gdag, Pv)
    A=Am(x, chi_invdag, Gdag, Pv)
    # print(A)
    val=eigen(A)
    w_val=val.values
    v=val.vectors
    # w: eigen values
    # v: eigen vectors
    w=real(w_val)
    # This code is to find the minimum eigenvalue
    min_eval=w[end:end]
    # print(min_eval)
    # This code is to find the eigenvector
    # associated with the mimimum eigenvalue
    min_evec=reshape(v[end:end,end:end],(length(ei),1))
    return min_eval,min_evec
end
function validityfunc(x, chi_invdag, Gdag, Pv)
    val=mineigfunc(x, chi_invdag, Gdag, Pv)
    min_eval=val[0]
    min_evec=val[1]
    if min_eval>0
        return 1
    else
        return -1
    end
end
function BFGS_fakeS_with_restart(initdof, dgfunc, validityfunc, mineigfunc, tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, gradConverge=false, opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, min_iter=6)
    dofnum = length(initdof)
    grad = zeros(ComplexF64, dofnum, 1)
    tmp_grad = zeros(ComplexF64, dofnum, 1)
    Hinv = I #approximate inverse Hessian
    prev_dualval = Inf
    dof = initdof
    function justfund(d)
        return dgfunc(d, np.array([]), fSlist, tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, get_grad=false)
    end
    # justfunc = lambda d: dgfunc(d, np.array([]), fSlist, get_grad=False)
    olddualval = Inf
    reductCount = 0
    while True #gradual reduction of modified source amplitude, outer loop
        fSlist = [] #remove old fake sources and restart with each outer iteration
        alpha_last = 1.0 #reset step size
        Hinv = I #reset Hinv
        val = dgfunc(dof, grad, fSlist, tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, get_grad=true)  
        dualval = val[1]
        grad = val[2]
#             print("grad",grad,"\n")
        # This gets the value of the dual function and the dual's gradient

#             print('\n', 'Outer iteration #', reductCount, 'the starting dual value is', dualval, 'fakeSratio', fakeSratio)
        iternum = 0
    end
        while true
            iternum += 1
#                 print('Outer iteration #', reductCount, 'Inner iteration #', iternum, flush=True)

            Ndir = - Hinv * grad
            pdir = Ndir / norm(Ndir)
            #backtracking line search, impose feasibility and Armijo condition
            p_dot_grad = pdir * grad
#                 print('pdir dot grad is', p_dot_grad)
            c_reduct = 0.7
            c_A = 1e-4
            c_W = 0.9
            alpha = alpha_start = alpha_last
#                 print('starting alpha', alpha_start)
            while validityfunc(dof+alpha*pdir, chi_invdag, Gdag, Pv)<=0 #move back into feasibility region
                alpha *= c_reduct
            end 
            alpha_feas = alpha
#                 print('alpha before backtracking is', alpha_feas)
            alphaopt = alpha
            Dopt = inf
            while True
                tmp_dual = justfunc(dof+alpha*pdir)
                if tmp_dual<Dopt #the dual is still decreasing as we backtrack, continue
                    Dopt = tmp_dual; alphaopt=alpha
                else
                    alphaopt=alpha ###ISSUE!!!!
                    break
                end
                if tmp_dual<=dualval + c_A*alpha*p_dot_grad #Armijo backtracking condition
                    alphaopt = alpha
                    break
                end                      
                alpha *= c_reduct
            end
            added_fakeS = False
            if alphaopt/alpha_start>(c_reduct+1)/2 #in this case can start with bigger step
                alpha_last = alphaopt*2
            else
                alpha_last = alphaopt
                if alpha_feas/alpha_start<(c_reduct+1)/2 && alphaopt/alpha_feas>(c_reduct+1)/2 #this means we encountered feasibility wall and backtracking linesearch didn't reduce step size, we should add fake source
                    added_fakeS = True
#                         print('encountered feasibility wall, adding a fake source term')
                    singular_dof = dof + alpha_feas*pdir #dof that is roughly on duality boundary
                    eig_fct_eval = mineigfunc(singular_dof, chi_invdag, Gdag, Pv) #find the subspace of ZTT closest to being singular to target with fake source
                    mineigw=eig_fct_eval[0]
                    mineigv=eig_fct_eval[1]
#                         print("min_evec shape",mineigv.shape,"\n")
#                         print("min_evec type",type(mineigv),"\n")

                    # fakeSval = dgfunc(dof, np.matrix([]), matrix, get_grad=False)
                    
                    fakeSval = dgfunc(dof, np.array([]), [mineigv], tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, get_grad=talse)
                    # This only gets us the value of the dual function

                    epsS = sqrt(fakeSratio*abs(dualval/fakeSval))
#                         print('epsS', epsS, '\n')
                    #fSlist.append(np.matmul(epsS,mineigv))
                    !append(fSlist,epsS*mineigv) #add new fakeS to fSlist                                  
                    #fSlist.append(epsS*mineigv) 
#                         print('length of fSlist', len(fSlist))
                 end
              end
#                 print('stepsize alphaopt is', alphaopt, '\n')
            delta = alphaopt * pdir

            #########decide how to update Hinv############
            tmp_val = dgfunc(dof+delta, tmp_grad, fSlist, tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, get_grad=true)
            tmp_dual = tmp_val[0]
            tmp_grad = tmp_val[1]
            # Not sure what the @ does.
            p_dot_tmp_grad = pdir * tmp_grad
            if added_fakeS
                Hinv = I #the objective has been modified; restart Hinv from identity
            elseif p_dot_tmp_grad > c_W*p_dot_grad #satisfy Wolfe condition, update Hinv
#                     print('updating Hinv')
                gamma = tmp_grad - grad
                gamma_dot_delta = alpha * (p_dot_tmp_grad-p_dot_grad)
                Hinv_dot_gamma = Hinv * tmp_grad + Ndir
#                     print("np.outer(delta, Hinv_dot_gamma)",( np.outer(Hinv_dot_gamma, delta) + np.outer(delta, Hinv_dot_gamma) - (1+np.dot(gamma, Hinv_dot_gamma)/gamma_dot_delta)*np.outer(delta, delta) ) / gamma_dot_delta,"\n")
#                     print("Hinv",Hinv,"\n")

                Hinv -= ( np.outer(Hinv_dot_gamma, delta) + np.outer(delta, Hinv_dot_gamma) - (1+np.dot(gamma, Hinv_dot_gamma)/gamma_dot_delta)*np.outer(delta, delta) ) / gamma_dot_delta
            end
            dualval = tmp_dual
            grad[1:end] = tmp_grad[1:end]
            dof += delta

            objval = dualval - dot(dof,grad)
            eqcstval = abs(dof) * abs(grad)
#                 print('at iteration #', iternum, 'the dual, objective, eqconstraint value is', dualval, objval, eqcstval)
#                 print('normgrad is', la.norm(grad))

            if gradConverge && iternum>min_iter && abs(dualval-objval)<opttol*abs(objval) && abs(eqcstval)<opttol*abs(objval) && norm(grad)<opttol * abs(dualval) #objective and gradient norm convergence termination
                break
            end
            if gradConverge==true && iternum>min_iter && abs(dualval-objval)<opttol*abs(objval) && abs(eqcstval)<opttol*abs(objval) #just objective convergence termination, in this case require minimum iternum to allow for adding new constraints with 0 multiplier
                break
            end
            if mod(iternum,iter_period)==0
#                     print('prev_dualval is', prev_dualval)
                if abs(prev_dualval-dualval)<abs(dualval)*opttol #dual convergence / stuck optimization termination
                    break
                end
                prev_dualval = dualval
            end
        end
        if abs(olddualval-dualval)<opttol*abs(dualval)
            break
        end
        """
        #if len(fSlist)<=1 and np.abs(olddualval-dualval)<opttol*np.abs(dualval): #converged w.r.t. reducing fake sources
            #break
        if len(fSlist)==0: #converged without needing fake sources, suggests strong duality
            break
        """
        olddualval = dualval
        reductCount += 1
        fakeSratio *= reductFactor #reduce the magnitude of the fake sources for the next iteration
    return dof, grad, dualval, objval
end
function item(x)
    return x # TBD with Sean
end
function nomad(x)
    return x # TBD with Sean
end
lsolvers=Dict("bfgs"=>th_solve, "item"=>cg_solve, "nomad"=>bicgstab_solve)
function lsolverpick(name, x, tsolver, ei, ei_tr, chi_invdag, Gdag, Pv) 
    return lsolvers[name](x, Dual, validityfunc, mineigfunc, tsolver, ei, ei_tr, chi_invdag, Gdag, Pv)
end
function extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)
    s=len(M)
    chi_coeff=3.0+2.0im
    chi=chi_coeff*np.identity(s)
    chi_invdag = inv(!conj(transpose(chi)))
    ei_tr = !conj(transpose(ei)) 
    Gdag=!conj(transpose(M))
    print("Solved using the ", tsolver, " method and the ", lsolver, " method")
    @time x_th=lsolverpick(lsolver, x0, tsolver, ei, ei_tr, chi_invdag, Gdag, Pv)
    gradient=x_th[end:end]
    Dual=x_th[end-1:end-1]
    obj=x_th[end-2:end-2]
    print("Gradient value: ", gradient, "\n")
    print("Objective value: ", objective, "\n")

end                                                                                                                                              
Z=376.730313668
wvlgth=1.0
Qabs=1e4
omega=(2*pi/wvlgth)*(1+1im/2/Qabs)
k0=omega/1
dx=2
dy=2
addx=1
liste=[[2,2,2,2,2,2,2,2],[3,3,3,3,3,3,3,3],[2,2,2,2,2,2,2,2],[2,2,2,2,2,2,2,2]]
setup=DipoleOverSlab_setup(dx,dy,addx,liste, dL)
print("setup",setup)
M=setup[1]
ei=setup[2]
Pv=setup[3]
x0=ones(ComplexF64, 2*length(liste), 1)
# different tsolvers and bfgs as the lsolver
lsolver="bfgs"
tsolver="cg"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)

tsolver="bicgstab"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)

tsolver="gmres"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)

# different tsolvers and item as the lsolver
lsolver="item"
tsolver="cg"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)

tsolver="bicgstab"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)

tsolver="gmres"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)

# different tsolvers and nomad as the lsolver
lsolver="nomad"
tsolver="cg"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)

tsolver="bicgstab"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)

tsolver="gmres"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)

A=zeros(Int8, 2, 3)
size(A)[2]

print("mod(i,2)",mod(4,2),"\n")


