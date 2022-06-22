import numpy as np
import scipy.io as sio
import utility
import fitting
import cv2
import matplotlib as plt
import sys
import time    

class morphable_model(object):       
    
    def __init__(self, path):
        '''
        Parameters
        ----------
        shapeMU, expMU: (3*nver, 1)
        shapePC: (3*nver, nshape), nshape=199
        expPC: (3*nver, nexp), nexp=29
        texPC: (3*nver, ntex), ntex=199
        tri: (ntri, 3)
        kpt_idx: (1, 68)
        
        '''
        super(morphable_model, self).__init__()
        self.model = sio.loadmat(path)
        self.model = self.model['model']
        self.model = self.model[0,0]
        
        self.shapeMU = self.model['shapeMU']
        self.shapePC = self.model['shapePC']
        self.shapeEV = self.model['shapeEV']
        self.expMU = self.model['expMU']
        self.expPC = self.model['expPC']
        self.expEV = self.model['expEV']
        self.fullMU = self.shapeMU + self.expMU
        
        self.texMU = self.model['texMU']
        self.texPC = self.model['texPC']
        self.texEV = self.model['texEV']
        
        self.tri = self.model['tri'].T - 1
        self.mouth_tri = self.model['tri_mouth'].T - 1
        self.full_tri = np.vstack((self.tri, self.mouth_tri))
        self.kpt_idx = self.model['kpt_idx'].astype(np.int32)-1
        
        self.nver = int(self.shapeMU.shape[0]/3)
        self.ntri = self.tri.shape[0]
        self.nshape = self.shapePC.shape[1]
        self.nexp = self.expPC.shape[1]
        self.ntex = self.texPC.shape[1]
        
        self.laplacian_matrix = np.eye(self.nver, dtype=np.float32)
        for i in range(self.nver):
            self.laplacian_matrix[i][i] = -1
        for face in self.tri:
            self.laplacian_matrix[face[0]][face[1]] = 1
            self.laplacian_matrix[face[0]][face[2]] = 1
            self.laplacian_matrix[face[1]][face[0]] = 1
            self.laplacian_matrix[face[1]][face[2]] = 1
            self.laplacian_matrix[face[2]][face[0]] = 1
            self.laplacian_matrix[face[2]][face[1]] = 1
        for i in range(self.nver):
            summation = np.sum(self.laplacian_matrix[i]) + 1
            idx = np.where(self.laplacian_matrix[i]==1)
            self.laplacian_matrix[i][idx] /= summation 
    
    def generate_para(self, type, flag):
        if type == "shape":
            if flag:
                para = np.random.uniform(1e04, 4e05, (self.nshape, 1))
            else:
                para = np.zeros((self.nshape, 1))
            return para
        
        elif type == "tex":
            if flag:
                para = np.random.uniform(size = (self.ntex, 1))
            else:
                para = np.zeros((self.ntex, 1))
            return para
        
        elif type == "exp":
            if flag:
                para = np. random.uniform(size = (self.nexp, 1))
            else:
                para = np.zeros((self.nexp, 1))
            return para
        
        else: 
            print("input type error")
            sys.exit()
                        
    
    def generate_vertices(self, shape_para, exp_para=0):
        vertices = self.fullMU + self.shapePC @ shape_para + self.expPC @ exp_para
        vertices = np.reshape(vertices, (self.nver, 3))
        return vertices
    
    def generate_tex(self, tex_para):
        tex = self.texMU + self.texPC @ tex_para
        tex = np.reshape(tex, (self.nver, 3))
        return tex
    
    def transform(self, vertices, f, R, t):
        trans_vertices = f * vertices @ R.T + t[np.newaxis,:]
        return trans_vertices
    
    
    def fit(self, kpt, tol=1e-4, iteration=10):
        '''
        Parameters:
            tol: error tolerance
            kpt: [68, 2], 2d position of input images
            kpt_vertices: [68, 3], 3d position of fixed landmarks
        fitting ||Xproject - kpt||^2 + l2 norm,
        where Xproject = f*P*R*kpt_vertices + t[:2]
        '''
        
        #-- init
        shape_para = np.zeros((self.nshape, 1), dtype = np.float32)
        exp_para = np.zeros((self.nexp, 1), dtype = np.float32)
        
        #-------------------- estimate
        kpt3d = np.tile(self.kpt_idx, [3, 1])*3
        kpt3d[1, :] += 1
        kpt3d[2, :] += 2
        valid_idx = kpt3d.flatten('F')
    
        fullMU = self.fullMU[valid_idx, :]
        shapePC = self.shapePC[valid_idx, :]
        expPC = self.expPC[valid_idx, :]
        
    
        for i in range(iteration):
            start = time.time()
            kpt_vertices = fullMU + shapePC @ shape_para + expPC @ exp_para
            kpt_vertices = np.reshape(kpt_vertices, (int(len(kpt_vertices)/3), 3))

            P = fitting.get_affine_matrix(kpt_vertices, kpt)
            f, R, t = fitting.P2Rft(P)

            shape = shapePC @ shape_para
            shape = np.reshape(shape, [int(len(shape)/3), 3]).T
            intact_shape = self.shapePC @ shape_para
            intact_shape = np.reshape(intact_shape, [int(len(intact_shape)/3), 3]).T
            exp_para = fitting.optim_regression(kpt.T, fullMU, expPC, self.expEV, shape, f, R, t[:2], intact_shape, self.expPC, self.laplacian_matrix, self.fullMU, lamb=100, type='exp')
            
            expression = expPC @ exp_para
            expression = np.reshape(expression, [int(len(expression)/3), 3]).T
            intact_exp = self.expPC @ exp_para
            intact_exp = np.reshape(intact_exp, [int(len(intact_exp)/3), 3]).T
            shape_para = fitting.optim_regression(kpt.T, fullMU, shapePC, self.shapeEV, expression, f, R, t[:2], intact_exp, self.shapePC, self.laplacian_matrix, self.fullMU, lamb=1, type='shape')
            
            err = fitting.err(kpt_vertices, kpt, shape_para, exp_para, f, R, t)
            print("shape err: %f"%err)
            if err < tol:
                break
            end = time.time()
            print('{} time cost:{}'.format(i, end-start))
            
        return shape_para, exp_para, f, R, t

    def frame_smooth_fit(self, pre_shape_para, pre_exp_para, kpt, iteration=100):
        '''
        it's a fitting process adding frame smooth regularization, we don't need eigenvalue as prior condition
        '''
        shape_para = pre_shape_para
        exp_para = pre_exp_para
    
        #-------------------- estimate
        kpt3d = np.tile(self.kpt_idx, [3, 1])*3
        kpt3d[1, :] += 1
        kpt3d[2, :] += 2
        valid_idx = kpt3d.flatten('F')
    
        fullMU = self.fullMU[valid_idx, :]
        shapePC = self.shapePC[valid_idx, :]
        expPC = self.expPC[valid_idx, :]
    
        for i in range(iteration):
            kpt_vertices = fullMU + shapePC @ shape_para + expPC @ exp_para
            kpt_vertices = np.reshape(kpt_vertices, (int(len(kpt_vertices)/3), 3))

            P = fitting.get_affine_matrix(kpt_vertices, kpt)
            f, R, t = fitting.P2Rft(P)

            shape = shapePC @ shape_para
            shape = np.reshape(shape, [int(len(shape)/3), 3]).T
            exp_para = fitting.smooth_optim(kpt.T, fullMU, expPC, pre_exp_para, shape, f, R, t[:2], lamb=1e-6, type='exp')

            expression = expPC @ exp_para
            expression = np.reshape(expression, [int(len(expression)/3), 3]).T
            shape_para = fitting.smooth_optim(kpt.T, fullMU, shapePC, pre_shape_para, expression, f, R, t[:2], lamb=1e-6, type='shape')
            
            err = fitting.err(kpt_vertices, kpt, shape_para, exp_para, f, R, t)
            print("shape err: %f"%err)
            
        return shape_para, exp_para, f, R, t
     
    def exp_fit(self, shape_para, kpt, tol=1e-4, iteration=10):
        '''
        Parameters:
            tol: error tolerance
            kpt: [68, 2], 2d position of input images
            kpt_vertices: [68, 3], 3d position of fixed landmarks
        fitting ||Xproject - kpt||^2 + l2 norm,
        where Xproject = f*P*R*kpt_vertices + t[:2]
        '''
        
        #-- init
        exp_para = np.zeros((self.nexp, 1), dtype = np.float32)
    
        #-------------------- estimate
        kpt3d = np.tile(self.kpt_idx, [3, 1])*3
        kpt3d[1, :] += 1
        kpt3d[2, :] += 2
        valid_idx = kpt3d.flatten('F')
    
        fullMU = self.fullMU[valid_idx, :]
        shapePC = self.shapePC[valid_idx, :]
        expPC = self.expPC[valid_idx, :]
    
        for i in range(iteration):
            kpt_vertices = fullMU + shapePC @ shape_para + expPC @ exp_para
            kpt_vertices = np.reshape(kpt_vertices, (int(len(kpt_vertices)/3), 3))

            P = fitting.get_affine_matrix(kpt_vertices, kpt)
            f, R, t = fitting.P2Rft(P)

            shape = shapePC @ shape_para
            shape = np.reshape(shape, [int(len(shape)/3), 3]).T
            exp_para = fitting.optim_regression(kpt.T, fullMU, expPC, self.expEV, shape, f, R, t[:2], lamb=50, type='exp')
  
            err = fitting.err(kpt_vertices, kpt, shape_para, exp_para, f, R, t)
            print("shape err: %f"%err)
            if err < tol:
                break
            
        return exp_para, f, R, t               
        
    def tex_fit(self, kpt, lamb=0, tol=1e-4, iteration=10):
        '''
        Parameters
        ----------
        kpt: [68, 3], rgb of keypoints of input images
        fitting ||kpt_tex - kpt||^2 + l2norm,
        where kpt_tex = texMU + texPC * tex_para
        
        Returns
        -------
        tex_para: [ntex, 1]
        '''
        # init
        tex_para = np.zeros((self.ntex, 1), dtype=np.float32)
        
        kpt3d = np.tile(self.kpt_idx, [3, 1])*3
        kpt3d[1, :] += 1
        kpt3d[1, :] += 2
        valid_idx = kpt3d.flatten('F')
        
        texMU = self.texMU[valid_idx, :]
        texPC = self.texPC[valid_idx, :]
        
        kpt_copy = kpt.copy()
        kpt_copy = np.reshape(kpt_copy, (kpt_copy.shape[0]*kpt_copy.shape[1], 1))    
        # linear regression 
        expression_left = texPC.T @ texPC + lamb*np.diagflat(1/self.texEV**2)
        expression_right = texPC.T @ (texMU - kpt_copy)
        tex_para = np.linalg.solve(expression_left, expression_right)
        
        # # gradient descent
        # for i in range(iteration):
        #     grad = texPC.T @ (texMU - kpt_copy + texPC@tex_para)
        #     tex_para = tex_para - lamb*grad
        #     err = np.sum((kpt_copy - texMU - texPC@tex_para)**2) / np.sum(kpt_copy**2)
        #     if err < tol:
        #         break
        #     print(err)
        
            
        # # nesterov momentum
        # Vdw = 0; beta = 0.5
        # for i in range(iteration):
        #     grad = texPC.T @ (texMU - kpt_copy + texPC@tex_para)
        #     Vdw = beta*Vdw + (1-beta)*grad
        #     tex_para = tex_para - lamb*Vdw
        #     err = np.sum((kpt_copy - texMU - texPC@tex_para)**2) / np.sum(kpt_copy**2)
        #     if err < tol:
        #         break
        #     print(err)
        return tex_para
    
    def tex_fine_fit(self, tex_para, valid_index, kpt, lamb=0, tol=1e-4, iteration=10):
        '''
        Parameters
        ----------
        tex_para: roughly fitted tex parameters[ntex, 1]
        kpt: full images rgb
        
        Returns
        -------
        '''
        kpt_copy = kpt.copy()
        kpt_copy = np.reshape(kpt_copy, (-1, 1))
        
        # # GD
        # for i in range(iteration):
        #     grad = self.texPC.T @ (self.texMU - kpt_copy + self.texPC@tex_para)
        #     tex_para = tex_para - lamb*grad/grad.shape[0]
        #     err = np.sum((kpt_copy - self.texMU - self.texPC@tex_para)**2) / np.sum(kpt_copy**2)
        #     if err < tol:
        #         break
        #     print(err)
            
        # nesterov momentum
        Vdw = 0; beta = 0.9; eta=1e5
        texMU_masked = self.texMU[valid_index, :]
        texPC_masked = self.texPC[valid_index, :]
        for i in range(iteration):
            grad = texPC_masked.T @ (texMU_masked - kpt_copy + texPC_masked@tex_para) + eta*np.diagflat(1/self.texEV**2)@tex_para
            Vdw = beta*Vdw + (1-beta)*grad
            tex_para = tex_para - lamb*Vdw/Vdw.shape[0]
            err = np.sum((kpt_copy - texMU_masked - texPC_masked@tex_para)**2) / np.sum(kpt_copy**2)
            if err < tol:
                break
            print(err)
            
        return tex_para
    
    # def tex_last_fit(self, tex_para, kpt, lamb=0, tol=1e-4, iteration=10):
    #     '''
    #     Parameters
    #     ----------
    #     tex_para: roughly fitted tex parameters[ntex, 1]
    #     kpt: full images rgb
        
    #     Returns
    #     -------
    #     '''        
    #     kpt3d = np.tile(self.kpt_idx, [3, 1])*3
    #     kpt3d[1, :] += 1
    #     kpt3d[1, :] += 2
    #     valid_idx = kpt3d.flatten('F')
        
    #     texMU = self.texMU[valid_idx, :]
    #     texPC = self.texPC[valid_idx, :]
        
    #     kpt_copy = kpt.copy()
    #     kpt_copy = np.reshape(kpt_copy, (kpt_copy.shape[0]*kpt_copy.shape[1], 1))    
        
    #     # gradient descent
    #     for i in range(iteration):
    #         grad = texPC.T @ (texMU - kpt_copy + texPC@tex_para)
    #         tex_para = tex_para - lamb*grad
    #         err = np.sum((kpt_copy - texMU - texPC@tex_para)**2) / np.sum(kpt_copy**2)
    #         if err < tol:
    #             break
    #         print(err)
        
    #     # # nesterov momentum
    #     # Vdw = 0; beta = 0.5
    #     # for i in range(iteration):
    #     #     grad = texPC.T @ (texMU - kpt_copy + texPC@tex_para)
    #     #     Vdw = beta*Vdw + (1-beta)*grad
    #     #     tex_para = tex_para - lamb*Vdw
    #     #     err = np.sum((kpt_copy - texMU - texPC@tex_para)**2) / np.sum(kpt_copy**2)
    #     #     if err < tol:
    #     #         break
    #     #     print(err)
    #     return tex_para
    
    
            
            
        






