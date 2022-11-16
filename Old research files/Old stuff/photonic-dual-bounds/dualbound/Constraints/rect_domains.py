import numpy as np


def get_rect_proj_in_rect_region(Rulco, Rbrco, Pulco, Pbrco):
    """
    get 1D boolean array mask for rectangular projection region in a larger rectangular region
    ulco stands for upper left corner coordinate; brco stands for bottom right corner coordinate; all input arguments are tuples
    """
    Rulx, Ruly = Rulco
    Rbrx, Rbry = Rbrco
    Pulx, Puly = Pulco
    Pbrx, Pbry = Pbrco

    if Pulx<Rulx or Puly<Ruly or Pbrx>Rbrx or Pbry>Rbry:
        raise ValueError('projection region outside of domain')

    Proj2D = np.zeros((Rbrx-Rulx,Rbry-Ruly),dtype=np.bool)
    Proj2D[Pulx-Rulx:Pbrx-Rulx,Puly-Ruly:Pbry-Ruly] = True
    return Proj2D.flatten()

def divide_rect_region(Rulco, Rbrco, Dx, Dy):
    #Dx, Dy are the row and column # of the rectangular subdivisions
    Nx = Rbrco[0] - Rulco[0]
    Ny = Rbrco[1] - Rulco[1]
    lxcoords = Rulco[0] + int(np.floor(Nx/Dx))*np.arange(Dx)
    rxcoords = np.zeros_like(lxcoords, dtype=int)
    rxcoords[:-1] = lxcoords[1:]
    rxcoords[-1] = Rbrco[0]

    uycoords = Rulco[1] + int(np.floor(Ny/Dy))*np.arange(Dy)
    bycoords = np.zeros_like(uycoords, dtype=int)
    bycoords[:-1] = uycoords[1:]
    bycoords[-1] = Rbrco[1]

    ulco_list = []
    brco_list = []
    for i in range(Dx):
        for j in range(Dy):
            ulco = (lxcoords[i],uycoords[j])
            brco = (rxcoords[i],bycoords[j])
            ulco_list.append(ulco)
            brco_list.append(brco)

    return ulco_list, brco_list
