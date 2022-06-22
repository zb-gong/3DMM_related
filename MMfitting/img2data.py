from MModel import morphable_model
import utility
import numpy as np
import cv2
import os
import time
# import cvxopt
# from cvxopt import matrix
# from cvxopt.solvers import qp
# import menpo
# from menpo.base import copy_landmarks_and_path
# from menpo.shape import PointCloud, TriMesh, ColouredTriMesh
# from menpo3d.rasterize import (
#     rasterize_shape_image_from_barycentric_coordinate_images,
#     rasterize_barycentric_coordinate_images
# )
# from menpo.transform import Scale
from sklearn.model_selection import train_test_split

# generate training data
h = w = 192
img_path, landmark_path = [], []
path = '/home/notebook/data/group/wenjing/CASIA-WebFace/'
target_path = '/home/notebook/data/group/zibo/CASIA_Face/'
for root, dirs, files in os.walk(path):
    for i in range(len(files)):
        if(files[i][-3:]=='jpg'):
            start = time.time()
            file_path = root + '/' + files[i]
            img = cv2.imread(file_path)
            pad_img = utility.process_image(img, min_side=800)
            flag = utility.judge_faces(pad_img, h, w)
            if flag:
                cropped_img, kpt = utility.crop_image_landmark(pad_img, h, w)
                cv2.imwrite(target_path+str(len(img_path))+'.jpg', cropped_img)
                np.save(target_path+str(len(img_path))+'.npy', kpt)
                img_path.append(target_path+str(len(img_path))+'.jpg')
                landmark_path.append(target_path+str(len(img_path))+'.npy')
            end = time.time()
            print(str(len(img_path))+' time:{}'.format(end-start))
            if(len(img_path)==50000):
                break
    else:
        continue
    break
    
train_img, test_img, train_landmark, test_landmark = train_test_split(img_path, landmark_path, test_size=0.05, random_state=0)

with open('/home/notebook/data/group/zibo/self-supervised/data/train.txt', 'w') as f:
    for i in range(len(train_img)):
        f.write(train_img[i]+' '+train_landmark[i]+'\n')
with open('/home/notebook/data/group/zibo/self-supervised/data/test.txt', 'w') as f:
    for i in range(len(test_img)):
        f.write(test_img[i]+' '+test_landmark[i]+'\n')


# generate the npy training data of lfw
# path = '/home/notebook/data/group/lfw/'
# files = os.listdir(path)
# for f in files:
#     if f[-3:] != 'jpg':
#         files.remove(f)


# valid_files = files.copy()
# h = w = 192
# for i in range(len(files)):
#     start = time.time()
#     img_path = path+files[i]
#     img = cv2.imread(img_path)
#     pad_img = utility.process_image(img, min_side=350)
#     flag = utility.judge_faces(pad_img, h, w)
#     if flag==0:
#         valid_files.remove(files[i])
#     end = time.time()
#     print('the %dth duration is:%f' %(i, end-start))

# img_restore = np.zeros((len(valid_files), h, w, 3))
# landmark_restore = np.zeros((len(valid_files), 68, 2), dtype=np.int32)
# for i in range(len(valid_files)):
#     start = time.time()
#     img_path = path + valid_files[i]
#     img = cv2.imread(img_path)
#     pad_img = utility.process_image(img, min_side=350)
#     cropped_img, kpt = utility.crop_image_landmark(pad_img, h, w)
#     img_restore[i] = cropped_img
#     landmark_restore[i] = kpt
#     end = time.time()
#     print('time for %dth images:%f'%(i,(end-start)))

# train_num = int(len(valid_files)/10*9)
# idx = np.arange(len(valid_files))
# shuffled_idx = np.random.permutation(idx)
# train_img = img_restore[shuffled_idx[:train_num]]
# train_landmarks = landmark_restore[shuffled_idx[:train_num]]
# test_img = img_restore[shuffled_idx[train_num:]]
# test_landmarks = landmark_restore[shuffled_idx[train_num:]]
# np.save('/home/notebook/data/group/lfw/npy_data/valid_images.npy', img_restore)
# np.save('/home/notebook/data/group/lfw/npy_data/train_images.npy', train_img)
# np.save('/home/notebook/data/group/lfw/npy_data/train_landmarks.npy', train_landmarks)
# np.save('/home/notebook/data/group/lfw/npy_data/test_images.npy', test_img)
# np.save('/home/notebook/data/group/lfw/npy_data/test_landmarks.npy', test_landmarks)







# test code
# train_images = np.load('/home/notebook/data/group/lfw/npy_data/train_images.npy')
# train_landmarks = np.load('/home/notebook/data/group/lfw/npy_data/train_landmarks.npy')
# for i in range(len(train_images)):
#     img_forshow = train_images[i]
#     for j in range(train_landmarks.shape[1]):
#         cv2.circle(img_forshow, (train_landmarks[i,j,0], train_landmarks[i,j,1]), 2, (0,0,255), -1)
#     cv2.imwrite('shit.jpg',img_forshow)


# try to get some training parameters data
# train_images = np.load('/home/notebook/data/group/lfw/npy_data/train_images.npy')
# train_landmarks = np.load('/home/notebook/data/group/lfw/npy_data/train_landmarks.npy')
# model = morphable_model('/home/notebook/data/group/zibo/MMfitting/data/model.mat')
# texMU = model.texMU
# texPC = model.texPC
# texEV = model.texEV
# tri = model.tri
# para_restore = np.zeros((len(train_images), 199+29+199+3+3+1))#shape, exp, tex, t, angles, f
# for i in range(len(train_images)):
#     start = time.time()
#     cropped_img = train_images[i]
#     kpt = train_landmarks[i]
#     h = cropped_img.shape[0]
#     w = cropped_img.shape[1]
#     kpt[:,0] = kpt[:,0] - w/2
#     kpt[:,1] = h/2 - kpt[:,1] - 1
#     # shape fitting
    
#     shape_para, exp_para, f, R, t  = model.fit(kpt, iteration=5)
#     vertices = model.generate_vertices(shape_para, exp_para)
#     trans_vertices = model.transform(vertices, f, R, t)
#     img_vertices = trans_vertices.copy()
#     img_vertices[:,0] = img_vertices[:,0]+w/2   
#     img_vertices[:,1] = h - 1 - (img_vertices[:,1]+h/2)
#     img_vertices[:,0] = np.maximum(np.minimum(img_vertices[:,0], w-1), 0)
#     img_vertices[:,1] = np.maximum(np.minimum(img_vertices[:,1], h-1), 0)
#     img_trimesh = TriMesh(img_vertices, trilist=tri)
#     img = menpo.image.Image(cropped_img.transpose([2,0,1]))
#     img.landmarks['fit_2d'] = PointCloud(img_vertices[:, :2])
#     color = img.sample(PointCloud(img_vertices[:,[1,0]])).T

#     lamb = 0
#     p = 2*(texPC.T @ texPC + lamb*np.diagflat(1/texEV**2)).astype(float)
#     q = -2*texPC.T @ (texMU-color).astype(float)
#     g = np.zeros((2*texPC.shape[0], texPC.shape[1]), dtype=float)
#     h = np.zeros((2*texPC.shape[0], 1), dtype=float)
#     g[:texPC.shape[0]] = texPC
#     g[texPC.shape[0]:] = -texPC
#     h[:texPC.shape[0]] = 255*np.ones((texPC.shape[0], 1)) - texMU
#     h[texPC.shape[0]:] = texMU
#     P = matrix(p); Q = matrix(q); G = matrix(g); H = matrix(h)
#     tex_para = qp(P,Q,G,H)['x']

#     para_restore[i] = np.squeeze(np.vstack((shape_para, exp_para, tex_para, t, angles, f)))
#     end = time.time()
#     print('time for %dth images:%f'%(i,(end-start)))





# old style(extracting parameters(199+29))
# # extract 68 keypoints of image
# path = "/home/notebook/data/group/lfw/"
# files = os.listdir(path)
# for f in files:
#     if f[-3:] != 'jpg':
#         files.remove(f)

# valid_files = files.copy()
# h = w = 192
# for i in range(len(files)):
#     start = time.time()
#     img_path = path+files[i]
#     flag = utility.judge_faces(img_path, h, w)
#     if flag==0:
#         valid_files.remove(files[i])
#     end = time.time()
#     print('the %dth duration is:%f' %(i, end-start))

# img_restore = np.zeros((len(valid_files), h, w, 3))
# para_restore = np.zeros((len(valid_files), 199+29))
# for i in range(len(valid_files)):
#     start = time.time()
#     img_path = path + valid_files[i]
#     cropped_img, kpt = utility.crop_image_landmark(img_path, h, w)
#     img_restore[i] = cropped_img
#     img_kpt = cropped_img[kpt[:,0], kpt[:,1], :].astype(np.float32)
    
#     kpt[:,0] = kpt[:,0] - w/2
#     kpt[:,1] = h/2 - kpt[:,1] - 1
    
#     # shape fitting
#     model = morphable_model('/home/notebook/data/group/zibo/MMfitting/data/model.mat')
#     shape_para, exp_para, f, R, t  = model.fit(kpt, iteration=20)
#     para_restore[i] = np.squeeze(np.vstack((shape_para, exp_para)))
    
#     end = time.time()
#     print('time for %dth images:%f'%(i,(end-start)))


# train_num = int(len(valid_files)/5*4)
# idx = np.arange(len(valid_files))
# shuffled_idx = np.random.permutation(idx)
# train_img = img_restore[shuffled_idx[:train_num]]
# train_lable = para_restore[shuffled_idx[:train_num]]
# test_img = img_restore[shuffled_idx[train_num:]]
# test_lable = para_restore[shuffled_idx[train_num:]]
# np.save('/home/notebook/data/group/lfw/npy_data/train_images.npy', train_img)
# np.save('/home/notebook/data/group/lfw/npy_data/train_labels.npy', train_lable)
# np.save('/home/notebook/data/group/lfw/npy_data/test_images.npy', test_img)
# np.save('/home/notebook/data/group/lfw/npy_data/test_labels.npy', test_lable)




