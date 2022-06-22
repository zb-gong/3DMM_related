import numpy as np
from config import Config




def localframe(tri, points): #求解lf的函数
    a = points[tri[2]] - points[tri[0]]
    b = points[tri[1]] - points[tri[0]]
    n = np.cross(a,b)
    lf = np.array([a,b,n])
    return lf

bs_path = '../data/Tester_101/Blendshape'

face_vertex = np.load('./train/triangles.npy')
tri_num = Config.tri_num
pose_lflist = []

for pi in range(0,Config.BLENDSHAPENUM):
    bs_name = '/shape_' + str(pi) + '.obj'
    with open(bs_path + bs_name) as file:
        points = []
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
    
    points = np.squeeze(np.array(points)) # vertices [n_ver, 3]

    localframes = np.zeros((tri_num, 3, 3))
    for i in range(tri_num):
        res = localframe(face_vertex[i], points)
        localframes[i] = res
    pose_lflist.append(localframes)
    print(pi)

pose_lflist = np.array(pose_lflist)
np.save('model/pose_lframe.npy', pose_lflist)

