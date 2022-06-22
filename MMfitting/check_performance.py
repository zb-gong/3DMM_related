# import torch
# import numpy as np
# import cv2
# import sys
# import utility
# import fitting
# from MModel import morphable_model
# sys.path.append('../self_supervised_crosstrain')
# from net import shape_network, tex_network


# # collect trained exp_para and shape_para
# shape_net = shape_network()
# tex_net = tex_network()
# shape_checkpoint = torch.load('../self_supervised_crosstrain/results/shapemodel_wreg30wsparse10wphoto1_cross_wreg30wsparse1wphoto1000_wstd1_lr0.001_lr0.1_200.pth', map_location=torch.device('cpu'))
# shape_net.load_state_dict(shape_checkpoint)
# tex_checkpoint = torch.load('../self_supervised_crosstrain/results/texmodel_wreg30wsparse10wphoto1_cross_wreg30wsparse1wphoto1000_wstd1_lr0.001_lr0.1_200.pth', map_location=torch.device('cpu'))
# tex_net.load_state_dict(tex_checkpoint)

# imgfile = '010A12'
# # imgfile = 'Adrian_McPherson_0002'
# h = w = 192
# origin_img = cv2.imread('/home/notebook/data/group/FGNET/images/'+imgfile+'.JPG')
# # origin_img = cv2.imread('/home/notebook/data/group/lfw/'+imgfile+'.jpg')
# pad_img = utility.process_image(origin_img, min_side=800)
# cropped_img, landmark = utility.crop_image_landmark(pad_img, h, w)
# landmark = landmark[27:]
# # img_kpt = cropped_img[kpt[:,0], kpt[:,1], :].astype(np.float32)
# img_forshow = cropped_img.copy()
# for i in range(landmark.shape[0]):
#     cv2.circle(img_forshow, (landmark[i][0], landmark[i][1]), 2, (0,0,255), -1)
# cv2.imwrite('shit.jpg',img_forshow)
# # cv2.waitKey(2000) 
# # cv2.destroyAllWindows()
# img = cropped_img[np.newaxis, :].astype(np.float)
# img = torch.from_numpy(img)

# # paras, corr_tex, corr_shape = net.forward(img)
# paras= shape_net.forward(img)
# tex_para = tex_net.forward(img)
# paras = paras.detach().numpy().T # shape_para, exp_para
# shape_para = paras[:199]
# exp_para = paras[199:228]
# tex_para = tex_para.detach().numpy().T


# model = morphable_model('data/model.mat')
# kpt_idx = model.kpt_idx.squeeze()[27:]
# vertices_idx = np.load('/home/notebook/data/group/zibo/self-supervised/vertices_idx.npy')
# triangles = np.load('/home/notebook/data/group/zibo/self-supervised/triangles.npy')
# vertices = model.generate_vertices(shape_para, exp_para)[vertices_idx] #nver x 3
# tex = model.generate_tex(tex_para)[vertices_idx]


# vert_kpt = vertices[kpt_idx, :] #68 x 3
# landmarks_img = np.zeros_like(landmark)
# landmarks_img[:,0] = landmark[:,0] - w/2
# landmarks_img[:,1] = h/2 - landmark[:,1] - 1
# P = fitting.get_affine_matrix(vert_kpt, landmarks_img) 
# f, R, t = fitting.P2Rft(P)
# vertices = f*vertices@R.T + t
# img_vert = np.zeros_like(vertices)
# img_vert[:,0] = vertices[:,0]+w/2
# img_vert[:,1] = h-1-(vertices[:,1]+h/2)
# img_vert[:,0] = np.maximum(np.minimum(img_vert[:,0], w-1), 0)
# img_vert[:,1] = np.maximum(np.minimum(img_vert[:,1], h-1), 0)
# color = utility.sample(cropped_img,img_vert)

# path = './para_check/'
# file_path = path + imgfile + '_origin.obj'
# with open(file_path, 'w') as file:
#     for i in range(len(vertices)):
#         v = vertices[i]
#         t = color[i]
#         file.write('v %f %f %f %f %f %f \n' %(v[0],v[1],v[2],t[2],t[1],t[0]))
#     for tri in triangles:
#         file.write('f %d %d %d\n' % (tri[0]+1, tri[1]+1, tri[2]+1))

# path = './para_check/'
# file_path = path + imgfile + '_base_e20.obj'
# with open(file_path, 'w') as file:
#     for i in range(len(vertices)):
#         v = vertices[i]
#         t = tex[i]
#         file.write('v %f %f %f %f %f %f \n' %(v[0],v[1],v[2],t[0],t[1],t[2]))
#     for tri in triangles:
#         file.write('f %d %d %d\n' % (tri[0]+1, tri[1]+1, tri[2]+1))
# a = 1


# second simplified design
# import torch
# import numpy as np
# import cv2
# import sys
# import utility
# import fitting
# from MModel import morphable_model
# sys.path.append('../self_supervised_newdesign')
# from net import network



# # collect trained exp_para and shape_para
# net = network()
# checkpoint = torch.load('../self_supervised_newdesign/results/model_wreg30wsparse10wphoto1wsmo1lr0.01_100.pth', map_location=torch.device('cpu'))
# net.load_state_dict(checkpoint)

# imgfile = '010A12'
# # imgfile = 'Adrian_McPherson_0002'
# h = w = 192
# origin_img = cv2.imread('/home/notebook/data/group/FGNET/images/'+imgfile+'.JPG')
# # origin_img = cv2.imread('/home/notebook/data/group/lfw/'+imgfile+'.jpg')
# pad_img = utility.process_image(origin_img, min_side=800)
# cropped_img, landmark = utility.crop_image_landmark(pad_img, h, w)
# landmark = landmark[27:]
# # img_kpt = cropped_img[kpt[:,0], kpt[:,1], :].astype(np.float32)
# img_forshow = cropped_img.copy()
# for i in range(landmark.shape[0]):
#     cv2.circle(img_forshow, (landmark[i][0], landmark[i][1]), 2, (0,0,255), -1)
# cv2.imwrite('shit.jpg',img_forshow)
# # cv2.waitKey(2000) 
# # cv2.destroyAllWindows()
# img = cropped_img[np.newaxis, :].astype(np.float)
# img = torch.from_numpy(img)

# # paras, corr_tex, corr_shape = net.forward(img)
# paras= net.forward(img)
# paras = paras.detach().numpy().T # shape_para, exp_para, tex_para, t, angles, f
# shape_para = paras[:199]
# exp_para = paras[199:228]
# # tex_para = paras[228:427]
# tex_para = np.zeros_like(shape_para)


# model = morphable_model('data/model.mat')
# kpt_idx = model.kpt_idx.squeeze()[27:]
# vertices_idx = np.load('/home/notebook/data/group/zibo/self-supervised/vertices_idx.npy')
# triangles = np.load('/home/notebook/data/group/zibo/self-supervised/triangles.npy')
# vertices = model.generate_vertices(shape_para, exp_para)[vertices_idx] #nver x 3
# tex = model.generate_tex(tex_para)[vertices_idx]


# vert_kpt = vertices[kpt_idx, :] #68 x 3
# landmarks_img = np.zeros_like(landmark)
# landmarks_img[:,0] = landmark[:,0] - w/2
# landmarks_img[:,1] = h/2 - landmark[:,1] - 1
# P = fitting.get_affine_matrix(vert_kpt, landmarks_img) 
# f, R, t = fitting.P2Rft(P)
# vertices = f*vertices@R.T + t
# img_vert = np.zeros_like(vertices)
# img_vert[:,0] = vertices[:,0]+w/2
# img_vert[:,1] = h-1-(vertices[:,1]+h/2)
# img_vert[:,0] = np.maximum(np.minimum(img_vert[:,0], w-1), 0)
# img_vert[:,1] = np.maximum(np.minimum(img_vert[:,1], h-1), 0)
# color = utility.sample(cropped_img,img_vert)

# path = './para_check/'
# file_path = path + imgfile + '_origin.obj'
# with open(file_path, 'w') as file:
#     for i in range(len(vertices)):
#         v = vertices[i]
#         t = color[i]
#         file.write('v %f %f %f %f %f %f \n' %(v[0],v[1],v[2],t[2],t[1],t[0]))
#     for tri in triangles:
#         file.write('f %d %d %d\n' % (tri[0]+1, tri[1]+1, tri[2]+1))

# path = './para_check/'
# file_path = path + imgfile + '_base_e20.obj'
# with open(file_path, 'w') as file:
#     for i in range(len(vertices)):
#         v = vertices[i]
#         t = tex[i]
#         file.write('v %f %f %f %f %f %f \n' %(v[0],v[1],v[2],t[0],t[1],t[2]))
#     for tri in triangles:
#         file.write('f %d %d %d\n' % (tri[0]+1, tri[1]+1, tri[2]+1))
# a = 1


# the first design
import torch
import numpy as np
import cv2
import sys
import utility
import fitting
from MModel import morphable_model
sys.path.append('../self_supervised_corrtex')
from net import network, tex_network, corr_tex_network


with open('/home/notebook/data/group/yuyunjie/facescape/facescape_bilinear_model_v1_3/data/predef_front_faces.pkl', 'rb') as f:
    faces_front = pickle.load(f)
# face indices(exclude head, ears and neck)
with open('/home/notebook/data/group/yuyunjie/facescape/facescape_bilinear_model_v1_3/data/front_indices.pkl', 'rb') as f:
    indices_front = pickle.load(f)
# triangle head
with open('/home/notebook/data/group/yuyunjie/facescape/facescape_bilinear_model_v1_3/data/predef_faces.pkl', 'rb') as f:
    faces_full = pickle.load(f)
# texture coordinates
with open('/home/notebook/data/group/yuyunjie/facescape/facescape_bilinear_model_v1_3/data/predef_texcoords.pkl', 'rb') as f:
    texcoords = pickle.load(f)

pdb.set_trace()
# bilinear model with 52 expression parameters and 50 identity parameters
# We perform Tucker decomposition only along the identity dimension to reserve the semantic meaning of parameters in expression dimension as speciﬁc blendshape weights
core_tensor = np.load('/home/notebook/data/group/yuyunjie/facescape/facescape_bilinear_model_v1_3/data/core_847_50_52.npy')
factors_id = np.load('/home/notebook/data/group/yuyunjie/facescape/facescape_bilinear_model_v1_3/data/factors_id_847_50_52.npy')

matrix_tex = np.load('/home/notebook/data/group/yuyunjie/facescape/facescape_bilinear_model_v1_3/data/matrix_text_847_100.npy')
mean_tex = np.load('/home/notebook/data/group/yuyunjie/facescape/facescape_bilinear_model_v1_3/data/mean_text_847_100.npy')
factors_tex = np.load('/home/notebook/data/group/yuyunjie/facescape/facescape_bilinear_model_v1_3/data/factors_tex_847_100.npy')



id = factors_id[0]
exp = np.zeros(52)
exp[0] = 1

core_tensor = core_tensor.transpose((2, 1, 0))
mesh_vertices_full = core_tensor.dot(id).dot(exp).reshape((-1, 3))
mesh_vertices_front = mesh_vertices_full[indices_front]

tex = mean_tex + matrix_tex.dot(factors_tex[0])
tex = tex.reshape((-1, 3)) / 255



# collect trained exp_para and shape_para
net = network()
tex_net = tex_network()
corrtex_net = corr_tex_network()
checkpoint = torch.load('../self-supervised/results/model_reg30photo10smo0.1newdata_100.pth', map_location=torch.device('cpu'))
tex_checkpoint = torch.load('../self_supervised_wholepara/results/model_wreg30wphoto1000wfinal1wstd0.002wsmo1wsta0.08_100.pth', map_location=torch.device('cpu'))
corrtex_checkpoint = torch.load('../self_supervised_corrtex/results/model_wreg30wphoto1000wsmo0wsta0.08lr1_100.pth', map_location=torch.device('cpu'))
net.load_state_dict(checkpoint)
tex_net.load_state_dict(tex_checkpoint)
corrtex_net.load_state_dict(corrtex_checkpoint)

imgfile = '024A25'
# imgfile = 'Aaron_Sorkin_0001'
h = w = 192
origin_img = cv2.imread('/home/notebook/data/group/FGNET/images/'+imgfile+'.JPG')
# origin_img = cv2.imread('/home/notebook/data/group/lfw/'+imgfile+'.jpg')
pad_img = utility.process_image(origin_img, min_side=800)
cropped_img, landmark = utility.crop_image_landmark(pad_img, h, w)
landmark = landmark[27:]
# img_kpt = cropped_img[kpt[:,0], kpt[:,1], :].astype(np.float32)
img_forshow = cropped_img.copy()
# for i in range(landmark.shape[0]):
#     cv2.circle(img_forshow, (landmark[i][0], landmark[i][1]), 2, (0,0,255), -1)
# cv2.imwrite('shit.jpg',img_forshow)
# cv2.waitKey(2000) 
# cv2.destroyAllWindows()
img = cropped_img[np.newaxis, :].astype(np.float)
img = torch.from_numpy(img)

# paras, corr_tex, corr_shape = net.forward(img)
paras, _, _ = net.forward(img)
tex_para, corr_tex = tex_net.forward(img)
corr_tex = corrtex_net.forward(img)
paras = paras.detach().numpy().T # shape_para, exp_para, tex_para, t, angles, f
corr_tex = corr_tex.detach().numpy().T # 3*nver x 1
# corr_shape = corr_shape.detach().numpy().T # 3*nver x 1
tex_para = tex_para.detach().numpy().T #199 x 1
shape_para = paras[:199]
exp_para = paras[199:228]
# tex_para = paras[228:427]


model = morphable_model('data/model.mat')
kpt_idx = model.kpt_idx.squeeze()[27:]
vertices_idx = np.load('/home/notebook/data/group/zibo/self-supervised/vertices_idx.npy')
triangles = np.load('/home/notebook/data/group/zibo/self-supervised/triangles.npy')
vertices = model.generate_vertices(shape_para, exp_para)[vertices_idx] #nver x 3
tex = model.generate_tex(tex_para)[vertices_idx]
# trans_vertices = model.transform(vertices, f, R, t)

tex_medium = corr_tex.reshape(vertices.shape[0], 3)
# shape_medium = corr_shape.reshape(vertices.shape[0], 3)
tex_final = tex + tex_medium
# shape_final = vertices + shape_medium
# t_final = tex_final-np.min(tex_final)
# tex_final = t_final/np.max(t_final)


vert_kpt = vertices[kpt_idx, :] #68 x 3
landmarks_img = np.zeros_like(landmark)
landmarks_img[:,0] = landmark[:,0] - w/2
landmarks_img[:,1] = h/2 - landmark[:,1] - 1
P = fitting.get_affine_matrix(vert_kpt, landmarks_img) 
f, R, t = fitting.P2Rft(P)
vertices = f*vertices@R.T + t
img_vert = np.zeros_like(vertices)
img_vert[:,0] = vertices[:,0]+w/2
img_vert[:,1] = h-1-(vertices[:,1]+h/2)
img_vert[:,0] = np.maximum(np.minimum(img_vert[:,0], w-1), 0)
img_vert[:,1] = np.maximum(np.minimum(img_vert[:,1], h-1), 0)
color = utility.sample(cropped_img,img_vert)

# texr = tex[:,::-1]
# image_base = ver2img(vertices, triangles, texr, cropped_img, h, w, type="render")
# cv2.imwrite('shit.jpg', image_base)
# texr = tex_final[:,::-1]
# image_base = ver2img(vertices, triangles, texr, cropped_img, h, w, type="render")
# cv2.imwrite('shit.jpg', image_base)
# path = './para_check/'
file_path = path + imgfile + '_origin.obj'
with open(file_path, 'w') as file:
    for i in range(len(vertices)):
        v = vertices[i]
        t = color[i]
        file.write('v %f %f %f %f %f %f \n' %(v[0],v[1],v[2],t[2],t[1],t[0]))
    for tri in triangles:
        file.write('f %d %d %d\n' % (tri[0]+1, tri[1]+1, tri[2]+1))

path = './para_check/'
file_path = path + imgfile + '_base_e20.obj'
with open(file_path, 'w') as file:
    for i in range(len(vertices)):
        v = vertices[i]
        t = tex[i]
        file.write('v %f %f %f %f %f %f \n' %(v[0],v[1],v[2],t[0],t[1],t[2]))
    for tri in triangles:
        file.write('f %d %d %d\n' % (tri[0]+1, tri[1]+1, tri[2]+1))

path = './para_check/'
file_path = path + imgfile + '_final_e20.obj'
with open(file_path, 'w') as file:
    for i in range(vertices.shape[0]):
        v = vertices[i]
        t = tex_final[i]
        file.write('v %f %f %f %f %f %f \n' %(v[0],v[1],v[2],t[0],t[1],t[2]))
    for tri in triangles:
        file.write('f %d %d %d\n' % (tri[0]+1, tri[1]+1, tri[2]+1))

a = 1

        


# # h = w = 256
# # img_vertices = trans_vertices.copy()
# # depth_img = utility.ver2img(img_vertices, model.tri, tex, h, w, type='depth')
# # depth_img -= np.min(depth_img)
# # depth_img /= np.max(depth_img)
# # depth_show = np.power(depth_img, 2)
# # renderred_img = utility.ver2img(img_vertices, model.tri, tex, h, w, type='render')
# # renderred_img = renderred_img/np.max(renderred_img)


# # cv2.imshow('projected_img', renderred_img[:,:,::-1])
# # cv2.imshow('depth_img', depth_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # cv2.imwrite('data/results/test.jpg', projected_img[:,:,::-1]*255)


