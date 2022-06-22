from MModel import morphable_model
import utility
import numpy as np
import cv2
import scipy.io
import os

# # extract 68 keypoints of image
path = "/home/notebook/data/group/FGNET/images/"
img = '010A12'
img_path = path + img + '.JPG'
h = 192; w = 192
cropped_img, kpt = utility.crop_image_landmark(img_path, h, w)
img_kpt = cropped_img[kpt[:,0], kpt[:,1], :].astype(np.float32)
# img_forshow = cropped_img.copy()
# for i in range(kpt.shape[0]):
#     cv2.circle(img_forshow, (kpt[i][0], kpt[i][1]), 2, (0,0,255), -1)
# # cv2.imwrite('data/results/landmark_'+file[:-4]+'.jpg', img_forshow)
# cv2.imshow('test',img_forshow)
# cv2.waitKey(000)
# cv2.destroyAllWindows()

kpt[:,0] = kpt[:,0] - w/2
kpt[:,1] = h/2 - kpt[:,1] - 1

# shape fitting
model = morphable_model('data/model.mat')

shape_para, exp_para, f, R, t  = model.fit(kpt, iteration=20)
# t = np.array([0,0,0])
# f = 0.001
# R = np.eye(3)
# pre_shape_para = shape_para
# pre_exp_para = exp_para
    
# forward projection
# ----------------------
vertices = model.generate_vertices(shape_para, exp_para)
trans_vertices = model.transform(vertices, f, R, t)
path = './results/'
file_path = path + img + '_shape.obj'
with open(file_path, 'w') as f:
    for i in range(len(trans_vertices)):
        v = trans_vertices[i]
        f.write('v %f %f %f\n' %(v[0],v[1],v[2]))
    for tri in model.tri:
        f.write('f %d %d %d\n' % (tri[0]+1, tri[1]+1, tri[2]+1))

img_vertices = trans_vertices.copy()

img_vertices[:,0] = img_vertices[:,0]+w/2   
img_vertices[:,1] = h - 1 - (img_vertices[:,1]+h/2)
tex = np.zeros_like(img_vertices)
for i in range(tex.shape[0]):
    coor = img_vertices[i][:2]
    img_coor = np.round(coor).astype(int)
    img_coor = np.maximum(np.minimum(img_coor, 191),0)
    tex[i] = cropped_img[img_coor[1],img_coor[0],::-1]
path = './'
file_path = path + str(4) + '_groundtruth.obj'
f = open(file_path, 'w')
for i in range(len(trans_vertices)):
    v = trans_vertices[i]
    t = tex[i]
    f.write('v %f %f %f %f %f %f\n' %(v[0],v[1],v[2], t[0],t[1],t[2]))
for tri in model.tri:
    f.write('f %d %d %d\n' % (tri[0]+1, tri[1]+1, tri[2]+1))

# tex fitting 
# img_vertices = trans_vertices.copy()
# full_img_kpt = utility.get_fulltex(img_vertices, cropped_img, h, w)
# tex_para = model.tex_fit(img_kpt, lamb=50, iteration=30)
# tex_para = model.tex_fine_fit(tex_para, full_img_kpt, lamb=10, iteration=80)

h = w = 256
tex_para = model.texEV
# img_vertices = utility.smooth(trans_vertices, model.tri)
img_vertices = trans_vertices.copy()
tex = model.generate_tex(tex_para)
tex = tex/np.max(tex)
tex_const = np.zeros_like(tex) + 0.5
tex_MU = model.texMU
tex_MU = tex_MU.reshape(-1, 3)
# f = 180/(np.max(img_vertices[:,1]) - np.min(img_vertices[:,1]))
# angles = np.array([0, 0, 0])
# t = np.array([0, 0, 0])
# img_vertices = model.transform(img_vertices, f, angles, t)




# # ------------------------------ 3. modify colors/texture(add light)
# # -- add point lights. light positions are defined in world space
# # set lights
# light_positions = np.array([[0, 0, 1000]])
# light_intensities = np.array([[0.0001, 0.0001, 0.0001]])
# lit_colors = mesh_numpy.light.add_light(img_vertices, model.tri, tex_const, light_positions, light_intensities)
# lit_colors2 = mesh_numpy.light.add_light(img_vertices, model.tri, tex_MU, light_positions, light_intensities)

# # depth_img = utility.ver2img(img_vertices, model.tri, tex, h, w, type="depth")
# # depth_img -= np.min(depth_img)
# # depth_img /= np.max(depth_img)
# # depth_show = np.power(depth_img, 2)
# renderred_img = utility.ver2img(trans_vertices, model.tri, tex, h, w, type="render")
# renderred_img = renderred_img/np.max(renderred_img)
# renderred_img2 = utility.ver2img(img_vertices, model.tri, lit_colors2, h, w, type="render")
# renderred_img2 = renderred_img2/np.max(renderred_img2)
# # scipy.io.savemat("example.mat", {'vertices':img_vertices, 'colors': tex, 'triangles':model.tri})


    
# cv2.imshow('projected_img', renderred_img)
# cv2.waitKey(000)
# cv2.destroyAllWindows()
# cv2.imwrite('./notex_frame_smooth/notex'+str(i)+'.jpg', renderred_img*255)
# # cv2.imshow('projected_img', renderred_img2[:,:, ::-1])
# # cv2.waitKey(000)
# # cv2.destroyAllWindows()
# cv2.imwrite('./tex_frame_smooth/tex'+str(i)+'.jpg', renderred_img2[:,:,::-1]*255)
# # cv2.imwrite('data/results/fit_test_'+ file[:-4] +'.jpg', renderred_img[:,:,::-1]*255)
# # cv2.imwrite('data/results/depth_test_'+ file[:-4] +'.jpg', depth_show*255)







###old school###
# from MModel import morphable_model
# import utility
# import numpy as np
# import cv2
# import scipy.io
# import os

# # extract 68 keypoints of image
# path = "./home/notebook/data/group/zibo/MMfitting/data/test_img/"
# files = os.listdir(path)
# # for file in files:
# file = 'exp_img3.jpg'
# img_path = path + file
# h = 182; w = 182
# cropped_img, kpt = utility.crop_image_landmark(img_path, h, w)
# img_kpt = cropped_img[kpt[:,0], kpt[:,1], :].astype(np.float32)
# img_forshow = cropped_img.copy()
# for i in range(kpt.shape[0]):
#     cv2.circle(img_forshow, (kpt[i][0], kpt[i][1]), 2, (0,0,255), -1)
# # cv2.imwrite('data/results/landmark_'+file[:-4]+'.jpg', img_forshow)
# cv2.imshow('test',img_forshow)
# cv2.waitKey(2000)
# cv2.destroyAllWindows()
# kpt[:,0] = kpt[:,0] - w/2
# kpt[:,1] = h/2 - kpt[:,1] - 1

# # shape fitting
# model = morphable_model('/home/notebook/data/group/zibo/MMfitting/data/model.mat')
# shape_para, exp_para, f, R, t  = model.fit(kpt, iteration=20)

    
    
# # forward projection
# # ----------------------
# vertices = model.generate_vertices(shape_para, exp_para)
# angles = utility.R2ang(R)
# trans_vertices = model.transform(vertices, f, angles, t)


# # tex fitting 
# img_vertices = trans_vertices.copy()
# full_img_kpt = utility.get_fulltex(img_vertices, cropped_img, h, w)
# tex_para = model.tex_fit(img_kpt, lamb=50, iteration=30)
# tex_para = model.tex_fine_fit(tex_para, full_img_kpt, lamb=10, iteration=100)
# # tex_para = model.tex_last_fit(tex_para, img_kpt, lamb=20, iteration=100)

# h = w = 256
# img_vertices = trans_vertices.copy()
# tex = model.generate_tex(tex_para)
# tex = tex/np.max(tex)
# # f = 180/(np.max(img_vertices[:,1]) - np.min(img_vertices[:,1]))
# # angles = np.array([0, 0, 0])
# # t = np.array([0, 0, 0])
# # img_vertices = model.transform(img_vertices, f, angles, t)
# depth_img = utility.ver2img(img_vertices, model.tri, tex, h, w, type="depth")
# depth_img -= np.min(depth_img)
# depth_img /= np.max(depth_img)
# depth_show = np.power(depth_img, 2)
# renderred_img = utility.ver2img(img_vertices, model.tri, tex, h, w, type="render")
# renderred_img = renderred_img/np.max(renderred_img)
# # scipy.io.savemat("example.mat", {'vertices':img_vertices, 'colors': tex, 'triangles':model.tri})


# cv2.imshow('projected_img', renderred_img[:,:,::-1])
# cv2.waitKey(000)
# cv2.destroyAllWindows()
# # cv2.imwrite('data/results/fit_test_'+ file[:-4] +'.jpg', renderred_img[:,:,::-1]*255)
# # cv2.imwrite('data/results/depth_test_'+ file[:-4] +'.jpg', depth_show*255)


