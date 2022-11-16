import numpy as np
import nlopt


def alternating_NLOPT_opt(altFunc, initdof1, initdof2, alg=nlopt.LD_MMA, ftol_rel=1e-3):
    """
    altFunc takes in two sets of dofs and grads: altFunc(dof1,grad1, dof2,grad2)
    altFunc modifies grad1 and grad2 in place; if grad1/grad2 is np.array([]) then no gradient is evaluated for the corresponding dof
    we optimize over both dof1 and dof2 by keeping one set of dof fixed while using NLOPT to optimize over the other, alternating as we reach convergence over one set of dof
    terminates when after one full cycle, the relative change in altFunc is less than ftol_rel
    """

    outer_i = 1
    prev_val = None
    while True:

        opt1func = lambda DOF, GRAD: altFunc(DOF,GRAD, initdof2, np.array([]))
        opt1 = nlopt.opt(alg, len(initdof1))
        opt1.set_lower_bounds(-np.inf * np.ones_like(initdof1))
        opt1.set_upper_bounds(np.inf * np.ones_like(initdof1))
        opt1.set_ftol_rel(ftol_rel)
        opt1.set_min_objective(opt1func)

        initdof1 = opt1.optimize(initdof1)
        print('at outer iteration #', outer_i, 'dof1 optimization')
        print('current function val at', opt1.last_optimum_value())
        print('result code = ', opt1.last_optimize_result())

        grad1 = np.zeros_like(initdof1)
        val = altFunc(initdof1, grad1, initdof2, np.array([]))
        print('grad1 is', grad1)

        opt2func = lambda DOF, GRAD: altFunc(initdof1,np.array([]), DOF,GRAD)
        opt2 = nlopt.opt(alg, len(initdof2))
        opt2.set_lower_bounds(-np.inf * np.ones_like(initdof2))
        opt2.set_upper_bounds(np.inf * np.ones_like(initdof2))
        opt2.set_ftol_rel(ftol_rel)
        opt2.set_min_objective(opt2func)

        initdof2 = opt2.optimize(initdof2)
        val = opt2.last_optimum_value()
        print('at outer iteration #', outer_i, 'dof2 optimization')
        print('current function val at', val)
        print('result code = ', opt2.last_optimize_result())

        if (not (prev_val is None)) and np.abs(val-prev_val) < ftol_rel*np.abs(val):
            break
        
        prev_val = val
        outer_i += 1

    return val, initdof1, initdof2

