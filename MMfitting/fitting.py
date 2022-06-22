import numpy as np
from scipy.optimize import nnls

def get_affine_matrix(X3d, x2d):
    '''
    Parameters
    ----------
        X3d: [n, 3]
        x2d: [n, 2]
        x2d = P @ X3d
    Returns
    -------
        P: [3, 4]
    '''
    x2d = x2d.T
    X3d = X3d.T
    assert(x2d.shape[1] == X3d.shape[1])
    n = x2d.shape[1]
    assert(n >= 4)

    #--- 1. normalization
    # 2d points
    mean = np.mean(x2d, 1) # (2,)
    x2d = x2d - np.tile(mean[:, np.newaxis], [1, n])
    average_norm = np.mean(np.sqrt(np.sum(x2d**2, 0)))
    scale = np.sqrt(2) / average_norm
    x2d = scale * x2d

    T = np.zeros((3,3), dtype = np.float32)
    T[0, 0] = T[1, 1] = scale
    T[:2, 2] = -mean*scale
    T[2, 2] = 1

    # 3d points
    X_homo = np.vstack((X3d, np.ones((1, n))))
    mean = np.mean(X3d, 1) # (3,)
    X3d = X3d - np.tile(mean[:, np.newaxis], [1, n])
    m = X_homo[:3,:] - X3d
    average_norm = np.mean(np.sqrt(np.sum(X3d**2, 0)))
    scale = np.sqrt(3) / average_norm
    X3d = scale * X3d

    U = np.zeros((4,4), dtype = np.float32)
    U[0, 0] = U[1, 1] = U[2, 2] = scale
    U[:3, 3] = -mean*scale
    U[3, 3] = 1

    # --- 2. equations
    A = np.zeros((n*2, 8), dtype = np.float32);
    X_homo = np.vstack((X3d, np.ones((1, n)))).T
    A[:n, :4] = X_homo
    A[n:, 4:] = X_homo
    b = np.reshape(x2d, [-1, 1])
 
    # --- 3. solution
    p_8 = np.linalg.pinv(A).dot(b)
    P = np.zeros((3, 4), dtype = np.float32)
    P[0, :] = p_8[:4, 0]
    P[1, :] = p_8[4:, 0]
    P[-1, -1] = 1

    # --- 4. denormalization
    # P_Affine = np.linalg.inv(T).dot(P.dot(U))
    P_Affine = np.linalg.solve(T, P@U)
    return P_Affine

def P2Rft(P):
    ''' decompositing camera matrix P
    Args: 
        P: (3, 4). Affine Camera Matrix.
    Returns:
        f: scale factor.
        R: (3, 3). rotation matrix.
        t: (3,). translation. 
    '''
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    f = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return f, R, t

def optim_regression(x, fullMU, argPC, argEV, arg, f, R, t2d, intact_arg, intact_argPC, laplacian_matrix, intact_fullMU, lamb=100, mu=1e-10, type='shape'):
    '''
    Args:
        x: (2, n). image points (to be fitted)
        
        fullMU: (3n, 1)
        argPC: (3n, n_arg) (arg:shape/exp)
        argEV: (n_arg, 1)
        arg: (3, n)
        f: scale
        R: (3, 3). rotation matrix
        t2d: (2,). 2d translation
        
        intact_arg: (3, n_full)
        intact_argPC: (3n_full, n_arg)
        lambda: regulation coefficient
    
    Returns:
        para: (n_arg, 1) principal parameters
    '''
    x = x.copy()
    assert(fullMU.shape[0] == argPC.shape[0])
    assert(fullMU.shape[0] == x.shape[1]*3)
    
    ## the regularization of the laplacian 
    n_full = intact_arg.shape[1]
    n_arg = intact_argPC.shape[1]
    intact_shapeMU = intact_fullMU.reshape((-1, 3))
    intact_pc_3d = intact_argPC.reshape((3, n_full, n_arg))
    intact_pc_3d = intact_pc_3d.transpose((1, 2, 0)) #(n_full, n_arg, 3)
    intact_pc_3d = intact_pc_3d.reshape((n_full, -1)) #(n_full, (n_arg*3))
    weight = laplacian_matrix @ intact_pc_3d #(n_full, (n_arg*3))
    weight = weight.reshape((n_full, 3, n_arg))
    weight = weight.reshape((3*n_full, n_arg)).T #(n_arg, (3*n_full))
    b = (laplacian_matrix @(intact_shapeMU + intact_arg.T)).T #(3, n_full)
    b = b.reshape((-1,1)) #(3n_full, 1)
    equation_right = mu * weight @ b #(n_arg, 1)
    equation_left = mu * weight @ weight.T
    
    
    n = x.shape[1]
    sigma = argEV
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    A = f*P@R
    # --- calc pc
    pc_3d = np.resize(argPC.T, [n_arg, n, 3]) 
    pc_3d = np.reshape(pc_3d, [n_arg*n, 3]) 
    pc_2d = pc_3d @ A.T
    pc = np.reshape(pc_2d, [n_arg, -1]).T # 2n x 29
    # --- calc b
    # fullMU
    mu_3d = np.resize(fullMU, [n, 3]).T # 3 x n
    # arg
    arg_3d = arg
    # calc b
    b = A.dot(mu_3d + arg_3d) + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    b = np.reshape(b.T, [-1, 1]) # 2n x 1
    
    # --- solve
    equation_left += np.dot(pc.T, pc) + lamb * np.diagflat(1/sigma**2)
    x = np.reshape(x.T, [-1, 1])
    equation_right += np.dot(pc.T, x - b)
    
    if type == 'shape':
        para = np.linalg.solve(equation_left, equation_right)
    elif type == 'exp':
        # para = nnls(equation_left, equation_right.flatten())[0][:, np.newaxis]
        para = np.linalg.solve(equation_left, equation_right)
        
    return para

# def optim_regression(x, fullMU, argPC, argEV, arg, f, R, t2d, lamb=100, type='shape'):
#     '''
#     Args:
#         x: (2, n). image points (to be fitted)
        
#         fullMU: (3n, 1)
#         argPC: (3n, n_arg) (arg:shape/exp)
#         argEV: (n_arg, 1)
#         arg: (3, n)
#         f: scale
#         R: (3, 3). rotation matrix
#         t2d: (2,). 2d translation
        
#         intact_arg: (3, n_full)
#         intact_argPC: (3n_full, n_arg)
#         lambda: regulation coefficient
    
#     Returns:
#         para: (n_arg, 1) principal parameters
#     '''
#     x = x.copy()
#     assert(fullMU.shape[0] == argPC.shape[0])
#     assert(fullMU.shape[0] == x.shape[1]*3)   

#     dof = argPC.shape[1]
#     n = x.shape[1]
#     sigma = argEV
#     t2d = np.array(t2d)
#     P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
#     A = f*P@R
    
#     # --- calc pc
#     pc_3d = np.resize(argPC.T, [dof, n, 3]) 
#     pc_3d = np.reshape(pc_3d, [dof*n, 3]) 
#     pc_2d = pc_3d @ A.T
#     pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 29
    
#     # --- calc b
#     # fullMU
#     mu_3d = np.resize(fullMU, [n, 3]).T # 3 x n
#     # arg
#     arg_3d = arg
#     # 
#     b = A.dot(mu_3d + arg_3d) + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
#     b = np.reshape(b.T, [-1, 1]) # 2n x 1
    
#     # --- solve
#     equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1/sigma**2)
#     x = np.reshape(x.T, [-1, 1])
#     equation_right = np.dot(pc.T, x - b)
    
#     if type == 'shape':
#         para = np.linalg.solve(equation_left, equation_right)
#     elif type == 'exp':
#         # para = nnls(equation_left, equation_right.flatten())[0][:, np.newaxis]
#         para = np.linalg.solve(equation_left, equation_right)
        
#     return para

def smooth_optim(x, fullMU, argPC, pre_arg, arg, f, R, t2d, lamb=100, type='shape'):
    '''
    Args:
        x: (2, n). image points (to be fitted)
        fullMU: (3n, 1)
        argPC: (3n, n_arg) (arg:shape/exp)
        pre_arg: (n_arg, 1)
        arg: (3, n)
        f: scale
        R: (3, 3). rotation matrix
        t2d: (2,). 2d translation
        lambda: regulation coefficient
    
    Returns:
        para: (n_arg, 1) principal parameters
    '''
    x = x.copy()
    assert(fullMU.shape[0] == argPC.shape[0])
    assert(fullMU.shape[0] == x.shape[1]*3)
    
    dof = argPC.shape[1]
    
    n = x.shape[1]
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    A = f*P@R
    
    # --- calc pc
    pc_3d = np.resize(argPC.T, [dof, n, 3]) 
    pc_3d = np.reshape(pc_3d, [dof*n, 3]) 
    pc_2d = pc_3d @ A.T
    pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 29
    
    # --- calc b
    # fullMU
    mu_3d = np.resize(fullMU, [n, 3]).T # 3 x n
    # arg
    arg_3d = arg
    # 
    b = A.dot(mu_3d + arg_3d) + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    b = np.reshape(b.T, [-1, 1]) # 2n x 1
    
    # --- solve
    equation_left = np.dot(pc.T, pc) + lamb * np.eye(pc.shape[1])
    x = np.reshape(x.T, [-1, 1])
    equation_right = np.dot(pc.T, x - b) + lamb * pre_arg
    
    if type == 'shape':
        para = np.linalg.solve(equation_left, equation_right)
    elif type == 'exp':
        para = nnls(equation_left, equation_right.flatten())[0][:, np.newaxis]
        
    return para




def err(X3d, x2d, shape_para, exp_para, f, R, t):
    P = np.array([[1,0,0], [0,1,0]], dtype=np.float32)
    A = f*P@R
    Xproject = X3d @ A.T
    err = np.sum((Xproject.flatten()-x2d.flatten())**2) / np.sum(Xproject.flatten()**2)
    return err


