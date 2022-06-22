import scipy
import scipy.optimize as op
import numpy as np
from cvxopt import solvers
from cvxopt import matrix
from config import Config
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os

edge_vpair = np.load('train/ep.npy') # every edge has two vertices index, [n_edges=n_tri, 2] (every triangle has 2 edges, every edge is used by two triangle, so n_tri=n_edges
face_edge_index = np.load('train/fe.npy') # edge index of edge_vpair for every triangles [n_tri, 2]
pose_lframe = np.load('train/pose_lframe.npy') # pose localframe[20, n_tri, 3, 3]
restpose = pose_lframe[0] # restpose localframe[n_tri, 3, 3]
model_list = np.load('model/pose_lframe.npy') # model localframe [47, n_tri, 3, 3]
triangles = np.load('train/triangles.npy') # triangles [n_tri, 3]
n_tri = triangles.shape[0] # number of triangles
pose_vert = np.load('train/pose_vert.npy') # pose vertices [20, n_ver, 3]
weight = np.random.rand(Config.BLENDSHAPENUM-1, Config.POSENUM) # weight[46,20]


def A2star(A, B0):
    '''
    Astar = (A0 + Ai)(Ai)^-1(B0) - B0
    Parameters:
        A: [47, n_tri, 3, 3]
    Returns:
        Astar: [46, n_tri, 3, 3]
    '''
    Astar = np.zeros((46, n_tri, 3, 3))
    A_invert = np.linalg.inv(A[0])
    # A_left = np.zeros_like(A_invert)
    
    for i in range(46):
        for j in range(n_tri):
            Astar[i][j] = (A[0][j] + A[i+1][j])@A_invert[j]@B0[j] - B0[j]
            
    return Astar 

def solve_localframe(restpose, pose_list, model_list, weight):
    '''
    Mi^A* need to be got first, Mi^A* = (M0^A+Mi^A)(M0^A)^-1M0^B - M0^B
    solved_lframe = (\sum alpha*alpha.T + beta*W)^-1 * (beta*W*Mi^A* - \sum alpha*M0^B + \sum alpha*M0^S)
    Parameters:
        we only take the first 2 columns to calculate
        restpose: localframe of restpose[n_tri, 3, 3] -> M0^B
        pose_list: pose_localframe [20, n_tri, 3, 3] -> Mj^S
        model_list: model_localframe [47, n_tri, 3, 3] -> Mi^A
        weight: [46, 20] ->alpha
    Returns:
        solved_lframe: [46, n_tri, 3, 3] ->Mi^B
    '''
    
    model_lframe = A2star(model_list, restpose) # Mi^A* [46, n_tri, 3, 3]
    W = np.zeros((46, 46))
    for i in range(46):
        W[i][i] = (1 + np.linalg.norm(model_list[i+1])) / (Config.Ka + np.linalg.norm(model_list[i+1]))
        W[i][i] = W[i][i]**2
    
    model_lf = np.reshape(model_lframe, (model_lframe.shape[0], model_list.shape[1]*model_lframe.shape[2]*model_lframe.shape[3])) # 46*m
    equation_right = Config.Beta*W@model_lf # 46 * m
    equation_left = Config.Beta*W
    for i in range(20):
        rp = restpose.flatten()
        pl = pose_list[i,:].flatten()
        equation_right += weight[:,i,np.newaxis]@pl[np.newaxis,:] - weight[:,i,np.newaxis]@rp[np.newaxis,:]
        equation_left += weight[:,i,np.newaxis]@weight[:,i,np.newaxis].T
    solved_lframe = np.linalg.solve(equation_left, equation_right)
    solved_lframe = np.reshape(solved_lframe, (46, n_tri, 3, 3))
    
    return solved_lframe

A = np.zeros((n_tri*2, pose_vert.shape[1]))
for i in range(n_tri):
    v = triangles[i]
    A[i*2][v[2]] = 1; A[i*2][v[0]] = -1
    A[i*2+1][v[1]] = 1; A[i*2+1][v[0]] = -1
coeff = 1e-8
A_temp = A.T@A
A_left = A_temp + coeff*np.eye(A.shape[1])
def lframe2vertice(lframe):
    '''
    Parameters:
        lframe:[ntri, 2, 3]
    Returns:
        vertice:[nver, 3]
    '''
    ll_test = np.reshape(lframe, (lframe.shape[0]*2, 3)) #n_tri*2 x 3
    A_right = A.T@ll_test+coeff*pose_vert[0]
    vertice = scipy.sparse.linalg.spsolve(A_left, A_right)
    print(np.linalg.norm(vertice - pose_vert[0]))
    return vertice
    
def solve_weight(pose_list, blendshape_list, pre_weight=0):
    '''
    Parameters:
        pose_list: vertices of every pose [20, n_ver, 3]
        blendshape_list: blendshape vertices [47, nver, 3]
        pre_weight: alpha* user specified weight
    Return:
        weight: [46, 20]
    '''
    blendshape_model = blendshape_list[0]
    blendshape_list = blendshape_list[1:47]
    weightnum = (Config.BLENDSHAPENUM-1) * len(pose_list)
    P = np.zeros((weightnum,weightnum))
    Q = np.zeros((weightnum,1))
    G = np.zeros((weightnum*2,weightnum))
    H = np.zeros((weightnum*2,1))
    # A = np.zeros((20, weightnum))
    # B = np.ones((20, 1))
    G[0:weightnum, :] = -np.eye(weightnum)
    G[weightnum:, :] = np.eye(weightnum)
    H[weightnum:, :] = np.ones((weightnum,1))
    
    bs_shape = np.reshape(blendshape_list, (blendshape_list.shape[0], blendshape_list.shape[1]*blendshape_list.shape[2])) # 46 * m
    p_shape = np.reshape(pose_list, (pose_list.shape[0], pose_list.shape[1]*pose_list.shape[2])) # 20 * m
    bs_model = blendshape_model.flatten() # m
    P_temp = 2 * bs_shape @ bs_shape.T # 2 = 1/2*xT*P*x
    
                                                                                                                      
    for i in range(20):
        P[i*46:(i+1)*46, i*46:(i+1)*46] = P_temp
        Q[i*46:(i+1)*46, :] = 2*bs_shape @ (p_shape[i] - bs_model)[:, np.newaxis]
        # A[i, i*46:(i+1)*46] = np.ones(46)
    gamma = 0
    P += P + gamma*np.eye(weightnum)
    
    p=matrix(P); q=matrix(Q); g=matrix(G); h=matrix(H);
    solution = solvers.qp(p, q, g, h)
    
    solved_weight = np.reshape(np.array(solution['x']), (46, 20), order='F')
    return solved_weight

if __name__ == '__main__':
    
    start=time.time()
    blend_vert = np.zeros((47, pose_vert.shape[1], pose_vert.shape[2])) # blendshape vertices [47, n_vert, 3]
    for i in range(3):
        lf = solve_localframe(restpose, pose_lframe, model_list, weight)
        lf[:,:,2,:] = np.cross(lf[:,:,0,:], lf[:,:,1,:]) # Mi^B [46, n_tri, 3, 3]
        blend_vert[0] = lframe2vertice(restpose[:,:2,:])
    
        for i in range(46):
            start = time.time()
            blend_vert[i+1] = lframe2vertice(lf[i][:,:2,:])
            print('blendshape %d succeed' %i)
            end = time.time()
            print('generate blendshape time: %d' %(end-start))
        weight = solve_weight(pose_vert, blend_vert)
        
        bl_vert = np.reshape(blend_vert, (47, blend_vert.shape[1]*blend_vert.shape[2]))
        ps_pre = bl_vert[0] + weight.T@bl_vert[1:47]
        ps_gt = np.reshape(pose_vert, (20, pose_vert.shape[1]*pose_vert.shape[2]))
        loss = np.linalg.norm(ps_pre - ps_gt)
        print('loss:%f'%loss)
    
    # fig = plt.figure()
    # ax1 = plt.axes(projection='3d')
    # ax1.scatter3D(blend_vert[4][:,0], blend_vert[4][:,1], blend_vert[4][:,2],'MarkerSize',1)
    # ax1.view_init(elev=90)
    # plt.show()
    np.save('./model/blend_vert.npy', blend_vert)
    
    
    for i in range(47):
        FILE_PATH = './blendshapes/'
        filename = 'shape' + str(i) + '.obj'
        obj_mesh_name = os.path.join(FILE_PATH, filename)
        with open(obj_mesh_name, 'w') as fp:
            for v in blend_vert[i]:
                fp.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            for f in triangles:
                fp.write("f %d %d %d\n" % (f[0]+1, f[1]+1, f[2]+1))
    end=time.time()
    print('total time cost: %f' %(end-start))


    

