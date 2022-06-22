from pathlib import Path
import numpy as np
import cv2
import dlib
import scipy.io as sio
import menpo.io as mio
import menpo
import utility
from menpo.base import copy_landmarks_and_path
from menpo.shape import PointCloud, TriMesh, ColouredTriMesh
from menpo3d.rasterize import (
    rasterize_shape_image_from_barycentric_coordinate_images,
    rasterize_barycentric_coordinate_images
)
from menpo.transform import Scale
from cvxopt import matrix
from cvxopt.solvers import qp
from MModel import morphable_model
import torch
import mesh_numpy

def get_lm(img):
    detector = dlib.get_frontal_face_detector()
    face = detector(img, 1)
    predictor = dlib.shape_predictor("/home/notebook/data/group/zibo/MMfitting/shape_predictor_68_face_landmarks.dat")
    
    if len(face) != 1:
        print("face detector is wrong")
        raise NotImplementedError
    landmark = predictor(img, face[0])
    lm = np.array([[landmark.part(i).x, landmark.part(i).y] for i in range(68)])
    return lm

def process_image(img, min_side=256):
    size = img.shape
    h, w = size[0], size[1]

    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv2.resize(img, (new_w, new_h))

    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = int((min_side-new_h)/2), int((min_side-new_h)/2), int((min_side-new_w)/2 + 1), int((min_side-new_w)/2)
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = int((min_side-new_h)/2 + 1), int((min_side-new_h)/2), int((min_side-new_w)/2), int((min_side-new_w)/2)
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = int((min_side-new_h)/2), int((min_side-new_h)/2), int((min_side-new_w)/2), int((min_side-new_w)/2)
    else:
        top, bottom, left, right = int((min_side-new_h)/2 + 1), int((min_side-new_h)/2), int((min_side-new_w)/2 + 1), int((min_side-new_w)/2)
    pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

    return pad_img



def as_colouredtrimesh(self, colours=None, copy=True):
    
    ctm = ColouredTriMesh(self.points, trilist=self.trilist,
                          colours=colours, copy=copy)
    return copy_landmarks_and_path(self, ctm)



def per_vertex_occlusion(mesh_in_img, err_proportion=0.0001, render_diag=600):

    [x_r, y_r, z_r] = mesh_in_img.range()
    av_xy_r = (x_r + y_r) / 2.0

    rescale = render_diag / np.sqrt((mesh_in_img.range()[:2] ** 2).sum())
    rescale_z = av_xy_r / z_r

    mesh = Scale([rescale, rescale, rescale * rescale_z]).apply(mesh_in_img)
    mesh.points[...] = mesh.points - mesh.points.min(axis=0)
    mesh.points[:, :2] = mesh.points[:, :2] + 2
    shape = np.round(mesh.points.max(axis=0)[:2] + 2).astype(int)

    bc, ti = rasterize_barycentric_coordinate_images(mesh, shape)
    si = rasterize_shape_image_from_barycentric_coordinate_images(
        as_colouredtrimesh(mesh), bc, ti)

    # err_proportion=0.01 is 1% deviation of total range of 3D shape
    threshold = render_diag * err_proportion
    xyz_found = si.as_unmasked().sample(mesh.with_dims([0, 1]), order=1).T
    err = np.sum((xyz_found - mesh.points) ** 2, axis=1)

    visible = err < threshold
    return visible


model = morphable_model('data/model.mat')
# DIAGONAL_RANGE = 180
tri = model.tri

img_path = '/home/notebook/data/group/FGNET/images/010A12.JPG'
img_raw = cv2.imread(img_path)
img_origin = process_image(img_raw, 256)
landmark = get_lm(img_origin)
img_forshow = img_origin.copy()
for i in range(landmark.shape[0]):
    cv2.circle(img_forshow, (landmark[i][0], landmark[i][1]), 2, (0,0,255), -1)
cv2.imwrite('shit.jpg',img_forshow)
h = img_origin.shape[0]
w = img_origin.shape[1]
landmark[:,0] = landmark[:,0] - w/2
landmark[:,1] = h/2 - landmark[:,1] - 1
shape_para, exp_para, f, R, t  = model.fit(landmark, iteration=1)

vertices = model.generate_vertices(shape_para, exp_para)
trans_vertices = model.transform(vertices, f, R, t)
 

# path = './results/'
# file_path = path +img + '_shape.obj'
# with open(file_path, 'w') as f:
#     for i in range(len(trans_vertices)):
#         v = trans_vertices[i]
#         f.write('v %f %f %f\n' %(v[0],v[1],v[2]))
#     for tri in model.tri:
#         f.write('f %d %d %d\n' % (tri[0]+1, tri[1]+1, tri[2]+1))
img_vertices = trans_vertices.copy()
img_vertices[:,0] = img_vertices[:,0]+w/2   
img_vertices[:,1] = h - 1 - (img_vertices[:,1]+h/2)
img_vertices[:,0] = np.maximum(np.minimum(img_vertices[:,0], w-1), 0)
img_vertices[:,1] = np.maximum(np.minimum(img_vertices[:,1], h-1), 0)
img_trimesh = TriMesh(img_vertices, trilist=tri)

img = menpo.image.Image(img.transpose([2,0,1]))
img.landmarks['fit_2d'] = PointCloud(img_vertices[:, :2])

# img, tr = img.rescale_landmarks_to_diagonal_range(DIAGONAL_RANGE, return_transform=True)
# img_vertices[:,:2] = img_vertices[:, :2]/tr.scale[0]

color = img.sample(PointCloud(img_vertices[:,[1,0]])).T
color = color[:, ::-1]
mask = per_vertex_occlusion(img_trimesh, 0.001, 1000)
mask_idx = []
for i in range(len(mask)):
    if mask[i]==True:
        mask_idx.append(i)
# color[~mask] = 0


color = color.reshape((-1,1))
texMU = model.texMU
texPC = model.texPC
texEV = model.texEV

kpt3d = np.tile(mask_idx, [3, 1])*3
kpt3d[1, :] += 1
kpt3d[1, :] += 2
valid_idx = kpt3d.flatten('F')
texMU_masked = texMU[valid_idx, :]
texPC_masked = texPC[valid_idx, :]
color_masked = color[valid_idx, :]

#  quadratic method 
# lamb = 0
# p = 2*(texPC.T @ texPC + lamb*np.diagflat(1/texEV**2)).astype(float)
# q = -2*texPC.T @ (texMU-color).astype(float)
# g = np.zeros((2*texPC.shape[0], texPC.shape[1]), dtype=float)
# h = np.zeros((2*texPC.shape[0], 1), dtype=float)
# # a = texPC_masked.astype(float)
# # b = (color_masked - texMU_masked).astype(float)
# g[:texPC.shape[0]] = texPC
# g[texPC.shape[0]:] = -texPC
# h[:texPC.shape[0]] = 255*np.ones((texPC.shape[0], 1)) - texMU
# h[texPC.shape[0]:] = texMU
# P = matrix(p); Q = matrix(q); G = matrix(g); H = matrix(h)
# # A = matrix(a); B = matrix(b)
# tex_para = cvxopt.solvers.qp(P,Q,G,H)['x']

lamb = 0
p = 2*(texPC_masked.T @ texPC_masked + lamb*np.diagflat(1/texEV**2)).astype(float)
q = -2*texPC_masked.T @ (texMU_masked-color_masked).astype(float)
g = np.zeros((2*texPC_masked.shape[0], texPC_masked.shape[1]), dtype=float)
h = np.zeros((2*texPC_masked.shape[0], 1), dtype=float)
g[:texPC_masked.shape[0]] = texPC_masked
g[texPC_masked.shape[0]:] = -texPC_masked
h[:texPC_masked.shape[0]] = 255*np.ones((texPC_masked.shape[0], 1)) - texMU_masked
h[texPC_masked.shape[0]:] = texMU_masked
P = matrix(p); Q = matrix(q); G = matrix(g); H = matrix(h)
tex_para = qp(P,Q,G,H)['x']

tex_para = model.tex_fine_fit(tex_para, valid_idx, color_masked, lamb=10, iteration=80)

tex = texMU + texPC @ tex_para
badpoints = np.where(tex>255)
tex[badpoints[0], badpoints[1]] = 255
badpoints = np.where(tex<0)
tex[badpoints[0], badpoints[1]] = 0
judge_criteria = np.linalg.norm(tex[valid_idx,:] - color_masked)
print('the err is {}'.format(judge_criteria))
tex = tex.reshape((-1,3))

path = './results/'
file_path = path + 'method_002A26.obj'
co = color.reshape(-1,3)
with open(file_path, 'w') as f:
    for i in range(img_vertices.shape[0]):
        v = trans_vertices[i]
        t = co[i]
        f.write('v %f %f %f %f %f %f\n' %(v[0], v[1], v[2], t[0], t[1], t[2]))
    for triangles in tri:
        f.write('f %d %d %d\n' %(triangles[0]+1, triangles[1]+1, triangles[2]+1))
        
texr = np.zeros_like(tex)+192
image_base = utility.ver2img(trans_vertices, tri, texr, img_origin, 256, 256, type="render")
cv2.imwrite('shit.jpg', image_base)        
        








# colortrimesh = mio.import_pickle('D:/Users/S9031009/Desktop/itwmm-master/result/color_test_x.pkl')
# mask_idx = mio.import_pickle('D:/Users/S9031009/Desktop/itwmm-master/result/mask_idx.pkl')
# tri = colortrimesh.trilist
# pt = colortrimesh.points
# color = colortrimesh.colours.reshape((-1,1)) * 255
# MModel = morphable_model('D:/Users/S9031009/Desktop/zibo/MMfitting/data/model.mat')
# texMU = MModel.texMU
# texPC = MModel.texPC
# texEV = MModel.texEV

# kpt3d = np.tile(mask_idx, [3, 1])*3
# kpt3d[1, :] += 1
# kpt3d[1, :] += 2
# valid_idx = kpt3d.flatten('F')
# texMU_masked = texMU[valid_idx, :]
# texPC_masked = texPC[valid_idx, :]
# color_masked = color[valid_idx, :]

# lamb = 0
# p = 2*(texPC_masked.T @ texPC_masked + lamb*np.diagflat(1/texEV**2)).astype(float)
# q = -2*texPC_masked.T @ (texMU_masked-color_masked).astype(float)
# g = np.zeros((2*texPC_masked.shape[0], texPC_masked.shape[1]), dtype=float)
# h = np.zeros((2*texPC_masked.shape[0], 1), dtype=float)
# g[:texPC_masked.shape[0]] = texPC_masked
# g[texPC_masked.shape[0]:] = -texPC_masked
# h[:texPC_masked.shape[0]] = 255*np.ones((texPC_masked.shape[0], 1)) - texMU_masked
# h[texPC_masked.shape[0]:] = texMU_masked
# P = matrix(p); Q = matrix(q); G = matrix(g); H = matrix(h)
# tex_para = cvxopt.solvers.qp(P,Q,G,H)['x'] 


# tex = texMU + texPC @ tex_para
# tex -= np.min(tex)
# tex /= np.max(tex)
# judge_criteria = np.linalg.norm(tex[valid_idx,:]*255 - color_masked)
# print('the err is {}'.format(judge_criteria))
# tex = tex.reshape((-1,3))

# path = 'D:/Users/S9031009/Desktop/zibo/MMfitting/'
# file_path = path + 'method1_8.obj'
# with open(file_path, 'w') as f:
#     for i in range(pt.shape[0]):
#         v = pt[i]
#         t = tex[i]
#         f.write('v %f %f %f %f %f %f\n' %(v[0], v[1], v[2], t[0], t[1], t[2]))
#     for triangles in tri:
#         f.write('f %d %d %d\n' %(triangles[0]+1, triangles[1]+1, triangles[2]+1))

