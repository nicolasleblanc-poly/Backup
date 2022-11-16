import numpy as np
import scipy.sparse as sp

def get_sparseProj_from_masks(designMask, subMask, format=None):
    """
    return a sparse diagonal projection operator acting on spatial point basis
    selected by 1D boolean array designMask that projects further 
    into subregion selected by 1D boolean array subMask
    """
    if np.sum(np.logical_and(np.logical_not(designMask), subMask))>0:
        raise ValueError('subMask contains points outside of design specified by designMask')
    return sp.diags(subMask[designMask].astype(np.complex), format=format)
