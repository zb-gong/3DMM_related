import numpy as np
from config import Config

basepath = 'D:/Users/S9031009/Desktop/facescape_bilinear_model_v1_3/blendshape_result/'
objFilePath = '../data/Tester_1/Blendshape/shape_0.obj'
posepath = basepath + '/TrainingPose'
bspath = basepath + '/Blendshape'

points =[] # vertices [n_vert, 3]
face_vertex = [] # triangles [n_tri, 3]
edge_vpair = [] # every edge has two vertices index, [n_edges=n_tri, 2] (every triangle has 2 edges, every edge is used by two triangle, so n_tri=n_edges 
face_edge_index = [] # edge index of edge_vpair for every triangles [n_tri, 2]
pose_vlist = [] #各pose包含的顶点坐标[20, n_ver, 3]，pose_vlist[0][1]对应第一个pose的第二个顶点
pose_lflist = [] #各pose的localframe信息[20, n_tri, 3, 3]，pose_lflist[0][1]对应第一个pose的第二个三角面片的localframe(3*3)
vert_num = tri_num = 0


def localframe(tri,pi): #求解lf的函数
    data = np.array(pose_vlist[pi])
    a = data[tri[2]] - data[tri[0]]
    b = data[tri[1]] - data[tri[0]]
    n = np.cross(a,b)
    lf = np.array([a,b,n])
    return lf

# 1st pose, we need to get face_vertex, edge_vpair, face_edge_index(they share the same topography)
first_pose = posepath + '\pose_0.obj'
with open(first_pose) as file:
    while 1:
        line = file.readline()
        sfv1 = []
        sfv2 = []
        sfvt1 = []
        sfvt2 = []
        if not line:
            break
        strs = line.split(" ")
        if strs[0] == "v":
            points.append([(float(strs[1]), float(strs[2]), float(strs[3]))])
        if strs[0] == "f":
            for i in range(1, 5):  # 将四边形切分为三角面片
                dum = strs[i].split("/")
                if (i != 4):
                    sfv1.append(int(dum[0]))
                    sfvt1.append(int(dum[1]))
                if (i != 2):
                    sfv2.append(int(dum[0]))
                    sfvt2.append(int(dum[1]))
            face_vertex.append(sfv1)
            face_vertex.append(sfv2)

    points = np.squeeze(np.array(points))
    face_vertex = np.array(face_vertex) 
    face_vertex -= 1 
    vert_num = points.shape[0]
    tri_num = face_vertex.shape[0]
    for i in range(tri_num):
        face_edge_index.append([0, 0])
    i = 0
    for fi in range(tri_num):
        edge1 = [face_vertex[fi][2], face_vertex[fi][0]]
        edge2 = [face_vertex[fi][1], face_vertex[fi][0]]  # lf的两个对应边
        if (edge1 not in edge_vpair):
            edge_vpair.append(edge1)
            face_edge_index[fi][0] = i
            i = i + 1
        else:
            face_edge_index[fi][0] = edge_vpair.index(edge1)
        if (edge2 not in edge_vpair):
            edge_vpair.append(edge2)
            face_edge_index[fi][1] = i
            i = i + 1
        else:
            face_edge_index[fi][1] = edge_vpair.index(edge2)
        # print(fi)
    edge_vpair = np.array(edge_vpair)
    face_edge_index = np.array(face_edge_index)
    np.save('train/triangles.npy', face_vertex)
    np.save('train/ep.npy', edge_vpair)
    np.save('train/fe.npy', face_edge_index)

##################################################################
#-------------------------------------------------------------------
##################################################################
# get the local frame for each pose   

for pi in range(0,Config.POSENUM):
    posename = '\pose_' + str(pi) + '.obj'
    with open(posepath + posename) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append([(float(strs[1]), float(strs[2]), float(strs[3]))])
    
    points = np.squeeze(np.array(points)) # vertices [n_ver, 3]
    pose_vlist.append(points)
    localframes = np.zeros((tri_num, 3, 3))
    for i in range(tri_num):
        res = localframe(face_vertex[i], pi)
        localframes[i] = res
    pose_lflist.append(localframes)
    print(pi)

pose_vlist = np.array(pose_vlist)
pose_lflist = np.array(pose_lflist)
np.save('train/pose_vert.npy', pose_vlist)
np.save('train/pose_lframe.npy', pose_lflist)
