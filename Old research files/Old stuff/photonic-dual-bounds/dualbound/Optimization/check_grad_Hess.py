import numpy as np
import scipy.linalg as la

def check_grad_Hess(func, dof, step=1e-3, checkHess=True, checkHessPD=True):
    """
    func(dof, grad, Hess) returns val=func(dof), and stores gradient and Hessian info in place
    func uses NLOPT convention: don't evaluate grad/Hess if len(grad)/len(Hess)==0
    len(grad) = len(dof), Hess.shape = (len(dof),len(dof))
    prints comparisons of finite different grad and Hess compared with calculated grad and Hess at dof
    """
    ndof = len(dof)
    grad = np.zeros(ndof)
    Hess = np.zeros((ndof,ndof))
    
    if checkHess:
        justfunc = lambda D: func(D, np.array([]), np.array([]))
        P0 = func(dof, grad, Hess)
    else:
        justfunc = lambda D: func(D, np.array([]))
        P0 = func(dof, grad)
        
    print('P0 is', P0)
    
    for i in range(ndof):
        deltai = step*np.abs(dof[i])
        if deltai<1e-12:
            deltai = step
            
        dof[i] += deltai
        P1 = justfunc(dof)
        dof[i] -= deltai
        fdgrad = (P1-P0)/deltai
        print('for dof',i,'the calculated gradient and fd estimate are', grad[i], fdgrad)

    if not checkHess:
        return
    
    for i in range(ndof):
        deltai = step*np.abs(dof[i])
        if deltai<1e-12:
            deltai = step
            
        pidof = dof.copy(); pidof[i] += deltai
        midof = dof.copy(); midof[i] -= deltai
        
        fdsqri = (justfunc(pidof)-2*P0+justfunc(midof))/deltai**2
        print('for dof',i, 'the calculated 2nd derivative and fd estimate are', Hess[i,i], fdsqri)
        
        for j in range(i+1, ndof):
            deltaj = step*np.abs(dof[j])
            if deltaj<1e-12:
                deltaj = step
            pipjdof = dof.copy(); pipjdof[i] += deltai; pipjdof[j] += deltaj
            pimjdof = dof.copy(); pimjdof[i] += deltai; pimjdof[j] -= deltaj
            mipjdof = dof.copy(); mipjdof[i] -= deltai; mipjdof[j] += deltaj
            mimjdof = dof.copy(); mimjdof[i] -= deltai; mimjdof[j] -= deltaj
            fd2ij = (justfunc(pipjdof)-justfunc(pimjdof)-justfunc(mipjdof)+justfunc(mimjdof))/(4*deltai*deltaj)
            print('for dofs',i,j,'the calculated cross derivative and fd estimate are', Hess[i,j], fd2ij)
    
    testSym = Hess - Hess.T
    print('test Hessian symmetry', la.norm(testSym))
    
    Heigw = la.eigvalsh(Hess)
    print('test Hess PD', Heigw)
