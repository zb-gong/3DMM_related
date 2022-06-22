import numpy as np
import os
from scipy.sparse.linalg import spsolve
import time
import util
#-------------------------
# this is a implement of deformation transfer, but the difference is that in our case we have a reference with the same topology as the target mesh
#-------------------------
vert_s, tri_s = util.load_obj('D:/Users/S9031009/Desktop/zibo/blendshape/data/Tester_101/TrainingPose/pose_0.obj', type='tri')
vert_t, tri_t = util.load_obj('D:/Users/S9031009/Desktop/zibo/blendshape/data/Tester_1/TrainingPose/pose_0.obj', type='tri')
ntri = tri_s.shape[0]
nver = vert_s.shape[0]

num_kpt = 68
landmark = np.random.randint(0, nver, size=num_kpt)

        
def v4_transfer(vertices, triangles):
    '''
    v4 = v1 + (v2-v1)x(v3-v1)/sqrt(|(v2-v1)x(v3-v1)|)
    Parameters:
        vertices: [nver,3]
        triangles: [ntri, 3]
    Returns:
        vert_target: [nver+ntri, 3]
        tri_target: [ntri, 4]
    '''
    vert_target = np.zeros((nver+ntri,3))
    vert_target[:nver,:] = vertices
    tri_target = np.zeros((ntri,4), dtype=int)
    tri_target[:,:3] = triangles
    for i in range(ntri):
        vert_idx = triangles[i]
        edge1 = vertices[vert_idx[1]] - vertices[vert_idx[0]]
        edge2 = vertices[vert_idx[2]] - vertices[vert_idx[0]]
        norm = np.cross(edge1, edge2)
        norm = norm / np.sqrt(np.linalg.norm(norm))
        v4 = vertices[vert_idx[0]] + norm
        vert_target[i+nver] = v4
        tri_target[i, 3] = i+nver
    return vert_target, tri_target

def V_generate(vs, ts):
    '''
    V = [v2-v1, v3-v1, v4-v1]^T
    Parameters:
        vs: [nver+ntri, 3]
        ts: [ntri, 4]
    Returns:
        V: [ntri, 3, 3] V[i, 0] is v2-v1
    '''
    V = np.zeros((ntri, 3, 3))
    for i in range(ntri):
        v_temp = ts[i]
        edge1 = vs[v_temp[1]] - vs[v_temp[0]]
        edge2 = vs[v_temp[2]] - vs[v_temp[0]]
        edge3 = vs[v_temp[3]] - vs[v_temp[0]]
        V[i] = np.array([edge1, edge2, edge3])
    return V
    
def trans_matrix(vs1, ts1, vs2, ts2):
    '''
    Q = Vs2 @ Vs1.inv [ntri, 3, 3] @ [ntri, 3, 3]
    Parameters:
        vs1, vs2: [nver+ntri, 3]
        ts1, ts2: [ntri, 4]
    Returns:
        Q: [n_tri, 3, 3]
    '''
    Vs1 = V_generate(vs1, ts1)
    Vs2 = V_generate(vs2, ts2)
    Vs1_inv = np.linalg.inv(Vs1)
    Q = np.zeros((ntri, 3, 3))
    for i in range(ntri):
        Q[i] = Vs2[i] @ Vs1_inv[i]
    return Q

def deform(vs1, ts1, vs2, ts2, vt, tt):
    '''
    Parameters:
        vs1, vs2, vt: [nver+ntri, 3]
        ts1, ts2, tt: [ntri, 3]
        actually the ts1 == ts2 == ts3
    Returns:
        Vt2: [ntri, 3, 3]
    '''
    Q = trans_matrix(vs1, ts1, vs2, ts2) # ntri x 3 x 3
    Vt1 = V_generate(vt, tt)
    Vt1_inv = np.linalg.inv(Vt1) # ntri x 3 x 3
    Vt2 = np.zeros_like(Vt1_inv)
    for i in range(ntri):
        Vt2[i] = Q[i] @ Vt1_inv[i].T @ (Vt1[i].T@Vt1[i])
    return Vt2 


A = np.zeros((ntri*2, nver))
for i in range(ntri):
    v = tri_t[i]
    A[i*2][v[1]] = 1; A[i*2][v[0]] = -1
    A[i*2+1][v[2]] = 1; A[i*2+1][v[0]] = -1
coeff = 1e-8
A_temp = A.T@A
A_left = A_temp + coeff*np.eye(A.shape[1])
def V2vertice(Vt):
    '''
    Parameters:
        Vt: [ntri, 3, 3]
        tt: target triangles[ntri, 3]
    Returns:
        vertice: [nver, 3]
    '''
    Vt_test = Vt[:,:2,:]
    Vt_test = np.reshape(Vt_test, (Vt_test.shape[0]*2, 3)) #n_tri*2 x 3
    A_right = A.T@Vt_test+coeff*vert_t
    vertice = spsolve(A_left, A_right)
    print(np.linalg.norm(vertice - vert_t))
    return vertice
def deformation_transfer(vs1, ts1, vs2, ts2, vt, tt):
    Vt2 = deform(vs1, ts1, vs2, ts2, vt, tt)
    vt2 = V2vertice(Vt2)
    return vt2

def laplacian_weight(vert, tri):
    '''
    calculate the weight of each 
    Parameters:
        vert: [nver, 3]
        tri: [ntri, 3]
    Returns:
        norm: [nver, 3] 
    '''
    norm = np.zeros_like(vert)
    

def fine_generate(coarse_vert, coarse_tri, gt_vert, gt_tri):
    '''
    1. Using Gradient Descent to iterate vertices:
        gradient_i = w1(coarse_vert_i - gt_vert_i) + w2(coarse_vert_i - gt_vert_i)
        gradient_j = w2(coarse_vert_i - gt_vert_i)
    2. Newton Gauss algorithm:
        ...
    Parameters:
        coarse_vert: [nver, 3]
        gt_vert: [nver, 3]
    Returns:
        fine_vert: [nver, 3]
    ''' 
    w1 = 0.5; w2 = 1; eta = 0.5
    fine_vert = coarse_vert.copy()
    
    # GD way to 
    for i in range(20):
        
        gradient1 = w2*(fine_vert - gt_vert)
        gradient2 = w1*(fine_vert[landmark] - gt_vert[landmark])
        fine_vert = fine_vert - eta*gradient1
        fine_vert[landmark] = fine_vert[landmark] - eta*gradient2
        print('epoch %d: the difference is %f' %(i, np.linalg.norm(fine_vert - gt_vert)))
    
    
    return fine_vert      

if __name__ == '__main__':
    vs4, ts4 = v4_transfer(vert_s, tri_s)
    vt4, tt4 = v4_transfer(vert_t, tri_t)
    path = 'D:/Users/S9031009/Desktop/facescape_trainset_001_100/21/models_reg/'
    target_path = 'D:/Users/S9031009/Desktop/fsmview_trainset_shape_021-040/fsmview_trainset/22/'
    # exp_num = 20
    # for i in range(1, exp_num):

    start = time.time()
    source_path = path + '3_mouth_stretch.obj'
    gt_path = target_path + '3_mouth_stretch.obj'
    vert_s2, tri_s2 = util.load_obj(source_path, type='tri')
    vs4_2, ts4_2 = v4_transfer(vert_s2, tri_s2)
    vert_t2 = deformation_transfer(vs4, ts4, vs4_2, ts4_2, vt4, tt4)
    # gt_vert2, tri_t2 = util.load_obj(gt_path)
    # fine_vert = fine_generate(vert_t2, tri_t, gt_vert2, tri_t2)
    
    # problem is very huge, I use vert of the same topography
    # if different topography is used, the mapping is the original mesh's closest point
    
    coarse_path = 'mesh_init/3_mouth_stretch.obj'
    fine_path = 'fine_mesh/pose' + str(i) + '.obj'
    util.write_obj(vert_t2, tri_t, coarse_path)
    # util.write_obj(fine_vert, tri_t, fine_path)
    