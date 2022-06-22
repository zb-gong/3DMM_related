import numpy as np

def load_obj(path, type='square'):
    vert = []
    tri = []
    file = open(path)
    while(1):
        line = file.readline()
        if not line:
            break
        coor = line.split(' ')
        if coor[0] == 'v':
            vert.append([float(coor[1]), float(coor[2]), float(coor[3])])
        elif coor[0] == 'f' and type == 'square':
            sfv1 = []
            sfv2 = []
            for i in range(1, 5):  # 将四边形切分为三角面片
                dum = coor[i].split("/")
                if (i != 4):
                    sfv1.append(int(dum[0]))
                if (i != 2):
                    sfv2.append(int(dum[0]))
            tri.append(sfv1)
            tri.append(sfv2)
        elif coor[0] == 'f' and type == 'tri':
            sfv = []
            for i in range(1, 4):
                dum = coor[i].split('/')
                sfv.append(int(dum[0]))
            tri.append(sfv)
    vert = np.array(vert)
    tri = np.array(tri) - 1
    return vert, tri

def write_obj(vert, tri, path):
    with open(path, 'w') as fp:
        for v in vert:
            fp.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for f in tri:
            fp.write("f %d %d %d\n" % (f[0]+1, f[1]+1, f[2]+1))
