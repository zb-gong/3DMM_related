import pickle
import pdb
import numpy as np

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


