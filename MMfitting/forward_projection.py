# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 10:42:33 2020

@author: S9031009
"""
from MModel import morphable_model
import numpy as np
import utility
import cv2
import matplotlib as plt


model = morphable_model('data/model.mat')
shape_para = model.generate_para("shape", 0)
exp_para = model.generate_para("exp", 0)

shape_para = np.load('./para_check/shape_para.npy')
exp_para = np.load('./para_check/exp_para.npy')
vertices = model.generate_vertices(shape_para, exp_para)

tex_para = model.generate_para("tex", 0)
tex = model.generate_tex(tex_para)

f = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1]))
angles = np.array([0, 0, 0])
t = np.array([0, 0, 0])
trans_vertices = model.transform(vertices, f, angles, t)

h = w = 256
img_vertices = trans_vertices.copy()
projected_img = utility.ver2img(img_vertices, model.tri, tex, h, w)

cv2.imshow('projected_img', projected_img[:,:,::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('data/results/test.jpg', projected_img[:,:,::-1]*255)