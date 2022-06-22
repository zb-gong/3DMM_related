import os
import shutil
import numpy as np
import dlib
import cv2
import math
import sys


def copy_mat(path, target_path):
    # path = "D:\\Users\\S9031009\\Desktop\\face3d-master\\examples\\Data\\BFM\\3ddfa"
    # target_path = "D:\\Users\\S9031009\\Desktop\\face3d-master\\examples\\Data\\BFM\\3ddfa"
    
    for root, dirs, files in os.walk(path):
        for i in range(len(files)):
            if(files[i][-3:] == "mat"):
                file_path = root + '/' + files[i]
                new_file = target_path + '/' + files[i]
                try:
                    shutil.copy(file_path, new_file)
                except shutil.Error:
                    continue
                    
def process_image(img, min_side=600):
    size = img.shape
    h, w = size[0], size[1]
    top = int((min_side - h)/2)
    bottom = int((min_side - h + 1)/2)
    left = int((min_side - w)/2)
    right = int((min_side - w + 1)/2)
    pad_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    return pad_img

def img_resize(img, rectangle, size):
    '''
    Parameters
    ----------
    img : input image
    rectangle : detected rectangles (attributes: right, left, bottom, top)
    size : multiply parameter

    Returns
    -------
    resized_img

    '''
    if rectangle.right()-rectangle.left() > rectangle.bottom()-rectangle.top():
        difference = (rectangle.right()-rectangle.left())-(rectangle.bottom()-rectangle.top())
        rectangle_top = rectangle.top() - int((difference+1)/2)
        rectangle_bot = rectangle.bottom() + int(difference/2)
        rectangle_right = rectangle.right()
        rectangle_left = rectangle.left()
    else:
        difference = -(rectangle.right()-rectangle.left())+(rectangle.bottom()-rectangle.top())
        rectangle_left = rectangle.left() - int((difference+1)/2)
        rectangle_right = rectangle.right() + int(difference/2)
        rectangle_top = rectangle.top()
        rectangle_bot = rectangle.bottom()
        
    increment = int((rectangle_right - rectangle_left)*size/2)
    resize_left = rectangle_left-increment
    resize_right = rectangle_right+increment
    resize_top = rectangle_top
    resize_bottom = rectangle_bot+2*increment
    if resize_left<0 or resize_right>img.shape[1]-1 or resize_top<0 or resize_bottom>img.shape[0]-1:
        print('Warning: the size need to be decreased')
        return img
    resized_img = img[resize_top:resize_bottom, resize_left:resize_right, :]            
    return resized_img

# def rect_resize(rectangle, landmark):
#     landmark = np.array([[landmark.part(i).x, landmark.part(i).y] for i in range(68)])
#     landmark_left = np.min(landmark[:, 0]) - 15
#     landmark_right = np.max(landmark[:, 0]) + 15
#     landmark_top = np.min(landmark[:, 1]) - 15
#     landmark_bottom = np.max(landmark[:, 1]) + 15
#     rect_left = np.minimum(landmark_left, rectangle.left())
#     rect_right = np.maximum(landmark_right, rectangle.right())
#     rect_top = np.minimum(landmark_top, rectangle.top())
#     rect_bottom = np.maximum(landmark_bottom, rectangle.bottom())
    
#     rect = dlib.rectangle(int(rect_left), int(rect_top), int(rect_right), int(rect_bottom))
#     return rect

def judge_faces(img, h, w):
    '''
    judge if faces can be recognized in input image 
    '''
    # detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/notebook/data/group/zibo/MMfitting/shape_predictor_68_face_landmarks.dat")
    
    
    faces = detector(img,1)
    if len(faces) == 0:
        return 0
    else: 
        # landmark = predictor(img, faces[0])        
        resized_img = img_resize(img, faces[0], 0.3) 
        cropped_img = cv2.resize(resized_img, (w, h), interpolation=cv2.INTER_AREA)
        crop_face = detector(cropped_img, 1)
        if len(crop_face)==0:
            return 0
        else:
            landmark = predictor(cropped_img, crop_face[0])
            landmark = np.array([[landmark.part(i).x, landmark.part(i).y] for i in range(68)])
            if np.max(landmark)>h-1 or np.min(landmark)< 0:
                return 0
            else:
                return 1
    
def crop_image_landmark(img, h, w):
    '''
    Parameters
    ----------
    path : (str) image path need cropping 
    
    Returns
    -------
    crop_img : (h, w, 3) 
    landmark : (68, 2)
    
    '''
    # detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/notebook/data/group/zibo/MMfitting/shape_predictor_68_face_landmarks.dat")
    
    
    faces = detector(img,1)
    print("number of faces: %d" %len(faces))
    
    # landmark = predictor(img, faces[0])    
    resized_img = img_resize(img, faces[0], 0.3) 
    cropped_img = cv2.resize(resized_img, (w, h), interpolation=cv2.INTER_AREA)
    crop_face = detector(cropped_img, 1)
    if len(crop_face)==0:
        print("no face detected")
        sys.exit()
    else:
        landmark = predictor(cropped_img, crop_face[0])
        landmark = np.array([[landmark.part(i).x, landmark.part(i).y] for i in range(68)])
        landmark[:,0] = np.array([np.maximum(np.minimum(landmark[j,0], w-1),0) for j in range(68)])
        landmark[:,1] = np.array([np.maximum(np.minimum(landmark[j,1], h-1),0) for j in range(68)])
    return cropped_img, landmark

def R2ang(R):
    '''
    Parameters
    ----------
    R: (3, 3)
    
    Returns
    -------
    x, y, z(pitch, yaw, roll)
    '''
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    return rx, ry, rz    

def ang2R(angles):
    '''
    Parameters
    ----------
    angles : np.array [3,] (roll, pitch, yaw)->(x, y, z)

    Returns
    -------
    R: [3,3]

    '''
    angles = np.deg2rad(angles)
    
    Rx=np.array([[1,                  0,                  0],
                 [0,  np.cos(angles[0]),  -np.sin(angles[0])],
                 [0,  np.sin(angles[0]),   np.cos(angles[0])]])

    Ry=np.array([[ np.cos(angles[1]), 0, np.sin(angles[1])],
                 [                 0, 1,                  0],
                 [-np.sin(angles[1]), 0, np.cos(angles[1])]])

    Rz=np.array([[ np.cos(angles[2]), -np.sin(angles[2]), 0],
                 [ np.sin(angles[2]),  np.cos(angles[2]), 0],
                 [                 0,                  0, 1]])
    R = Rz@Ry@Rx
    return R      


# forward projection 3d -> 2d
# ------------------------------------------------------

def isPointInTri(point, tri_points):
    ''' Judge whether the point is in the triangle
    Method:
        http://blackpawn.com/texts/pointinpoly/
    Parameters:
        point: (2,). [u, v] or [x, y] 
        tri_points: (3 vertices, 2 coords). three vertices(2d points) of a triangle. 
    Returns:
        bool: true for in triangle
    '''
    tp = tri_points

    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)

def get_point_weight(point, tri_points):
    ''' Get the weights of the position
    Methods: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
     -m1.compute the area of the triangles formed by embedding the point P inside the triangle
     -m2.Christer Ericson's book "Real-Time Collision Detection". faster.(used)
    Parameters:
        point: (2,). [u, v] or [x, y] 
        tri_points: (3 vertices, 2 coords). three vertices(2d points) of a triangle. 
    Returns:
        w0: weight of v0
        w1: weight of v1
        w2: weight of v3
     '''
    tp = tri_points
    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    w0 = 1 - u - v
    w1 = v
    w2 = u

    return w0, w1, w2

def ver2img(vertices, triangles, colors, img, h, w, type="render"):
    ''' render mesh with colors
    Parameters:
        vertices: [nver, 3]
        triangles: [ntri, 3] 
        colors: [nver, 3]
        h: height
        w: width    
    Returns:
        image: [h, w, 3]. 
        depth_image; [h, w]
    '''
    colors = np.reshape(colors, vertices.shape)
    assert vertices.shape[0] == colors.shape[0]
    
    # initial
    vert = vertices.copy()
    vert[:,0] = vert[:,0]+w/2
    vert[:,1] = h - 1 - (vert[:,1]+h/2)
    # image = np.zeros((h, w, 3))
    image = img.copy()
    depth_image = np.zeros((h, w))
    depth_buffer = np.zeros([h, w]) - 999999.

    for i in range(triangles.shape[0]):
        tri = triangles[i, :] # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vert[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(vert[tri, 0]))), w-1)

        vmin = max(int(np.ceil(np.min(vert[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(vert[tri, 1]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not isPointInTri([u,v], vert[tri, :2]): 
                    continue
                w0, w1, w2 = get_point_weight([u, v], vert[tri, :2])
                point_depth = w0*vert[tri[0], 2] + w1*vert[tri[1], 2] + w2*vert[tri[2], 2]

                if point_depth > depth_buffer[v, u]:
                    depth_buffer[v, u] = point_depth
                    image[v, u, :] = w0*colors[tri[0], :] + w1*colors[tri[1], :] + w2*colors[tri[2], :]
                    depth_image[v,u] = w0*vert[tri[0], 2] + w1*vert[tri[1], 2] + w2*vert[tri[2], 2]
    if type == "render":
        return image
    elif type == "depth":
        return depth_image
    else:
        print("render type error")
        sys.exit()
    
def get_fulltex(vertices, img, h, w):
    '''
    Parameters
    ----------
    vertices: shape fitted vertices[nver, 3]
    img: input cropped image [w, h, 3]
    w0 |-----|w1
       |     | 
    w2 |-----|w3 
    
    Returns
    -------
    tex: tex for render[nver, 3]
    '''
    tex = np.zeros_like(vertices)
    vertices[:,0] = vertices[:,0] + w/2
    vertices[:,1] = h - 1 - (vertices[:,1] + h/2)
    # img_position = np.around(vertices).astype(np.int32)
    img_position = vertices.copy()
    img_position[:,0] = np.maximum(np.minimum(vertices[:,0], h-1), 0)
    img_position[:,1] = np.maximum(np.minimum(vertices[:,1], w-1), 0)
    for i in range(img_position.shape[0]):
        left = np.floor(img_position[i, 1]).astype(np.int32)
        right = np.ceil(img_position[i, 1]).astype(np.int32)
        top = np.floor(img_position[i, 0]).astype(np.int32)
        bottom = np.ceil(img_position[i, 0]).astype(np.int32)
        w0 = (img_position[i,1]-left)**2 + (img_position[i,0]-top)**2
        w1 = (right-img_position[i,1])**2 + (img_position[i,0]-top)**2
        w2 = (img_position[i,1]-left)**2 + (bottom-img_position[i,0])**2
        w3 = (right-img_position[i,1])**2 + (bottom-img_position[i,0])**2
        summ = w0 + w1 + w2 + w3
        w0 = w0/summ; w1 = w1/summ; w2 = w2/summ; w3 = w3/summ;
        tex[i] = img[left, top]*w0 + img[right, top]*w1 + img[left, bottom]*w2 + img[right, bottom]*w3
    
    # tex = img[img_position[:,1], img_position[:,0], ::-1]
    tex = tex[:,::-1]
    return tex

def sample(images, vertices):
    '''
    Parameters:
        images: (h, w, 3)
        vertices: (3*nver)
    Returns:
        sampled: (nver, 3)
    '''
    vert = vertices.reshape(-1, 3)
    int_vert = np.round(vert).astype(np.int32)
    proj_x = int_vert[:,1]
    proj_y = int_vert[:,0]
    color = images[proj_x, proj_y, :]
    return color