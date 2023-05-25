
# THIS SCRIPT IS TO GET IOU EVALUATION METRICS AS SHOWN IN RESULTS SECTION IN REPORT
# WE GET BOTH 2D AND 3D IOU SCORES FOR CARS, PEDESTRIAN AND CYCLIST IN EASY, MODERATE AND HARD DOMAIN

import os
import cv2
import errno
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
from lib.DataUtils import *
from lib.Utils import *
from tqdm import tqdm
from lib import Model, ClassAverages



###########################################################################################################################################################


# 3D IoU caculate code for 3D object detection 
# Kent 2018/12

import numpy as np
from scipy.spatial import ConvexHull
from numpy import *

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d

    
if __name__=='__main__':
    print('------------------')

    exp_no = 8  
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset_gt =  Dataset('/home/drajani/Downloads/3D-Bounding-Boxes-From-Monocular-Images/Kitti/validation') # Dataset('/home/drajani/Downloads/3D-Bounding-Boxes-From-Monocular-Images/Kitti/validation/CARS')
    all_images = dataset_gt.all_objects()


    # in all validation images
    GT_list = []
    Val_list = []

    for key in tqdm(sorted(all_images.keys())):


        data = all_images[key]
        objects = data['Objects']  # (= no. of images)
        
        # in one GT image (GROUND TRUTH)
       
        for object in objects:
            label = object.label
            GT_list.append(label) # LEN = 1488
    

        # in one val image (PREDICTED/DETECTED)
        file = open("/home/drajani/Downloads/3D-Bounding-Boxes-From-Monocular-Images/Kitti/results/validation/labels/exp_" + str(exp_no) + "/data/" + str(key) + ".txt",'r')

        for i in file.readlines():
            num = i.split()
            Val_list.append(num) # LEN = 1488

    


    #  gt[3] alpha; gt[8,9,10] dimensions; gt[11,12,13] locations
    #  det[3] alpha; det[8,9,10] dimesnions; det[11,12,13] locations
    #  get_3d_box(box_size, heading_angle, center) == (dimension coords, alpha, location coords)
    sum_car_e,sum_car_m,sum_car_h = 0,0,0
    sum_car_e_2d,sum_car_m_2d,sum_car_h_2d = 0,0,0
    count_car_e,count_car_m,count_car_h = 0,0,0

    classes = ['Car','Pedestrian','Cyclist']

    for types in classes:

        for i in range(len(Val_list)):
            if (GT_list[i]['Class'] == types) and (Val_list[i][0] == types):
                corners_3d_ground  = get_3d_box((GT_list[i]['Dimensions'][0] ,GT_list[i]['Dimensions'][1] ,GT_list[i]['Dimensions'][2]), GT_list[i]['Alpha'], (GT_list[i]['Location'][0] , GT_list[i]['Location'][1] ,GT_list[i]['Location'][2])) 
                corners_3d_predict = get_3d_box((float(Val_list[i][8]) ,float(Val_list[i][9]) ,float(Val_list[i][10])), float(Val_list[i][3]), (float(Val_list[i][11]) ,float(Val_list[i][12]) ,float(Val_list[i][13]))) 
                (IOU_3d,IOU_2d)=box3d_iou(corners_3d_predict,corners_3d_ground)

                if IOU_3d >= 0.7 and GT_list[i]['occlusion'] == 0: # easy
                    count_car_e += 1
                    sum_car_e = sum_car_e + IOU_3d
                    sum_car_e_2d = sum_car_e_2d + IOU_2d

                if IOU_3d >= 0.5 and GT_list[i]['occlusion'] == 1: # moderate
                    count_car_m += 1
                    sum_car_m = sum_car_m + IOU_3d
                    sum_car_m_2d = sum_car_m_2d + IOU_2d

                if IOU_3d >= 0.5 and GT_list[i]['occlusion'] == 2: # hard
                    count_car_h += 1
                    sum_car_h = sum_car_h + IOU_3d
                    sum_car_h_2d = sum_car_h_2d + IOU_2d


        avg_car_e =    sum_car_e / count_car_e 
        avg_car_m =    sum_car_m / count_car_m 
        avg_car_h =    sum_car_h / count_car_h

        print('{0}'.format(types))
        print("Avg IOU_3d  {0}  with difficulty  EASY: {1}".format(types, avg_car_e))
        print("Avg IOU_3d  {0}  with difficulty  MODERATE: {1}".format(types, avg_car_m))
        print("Avg IOU_3d  {0}  with difficulty  HARD: {1}".format(types, avg_car_h))
        print('')

        avg_car_e_2d =    sum_car_e_2d / count_car_e 
        avg_car_m_2d =    sum_car_m_2d / count_car_m 
        avg_car_h_2d =    sum_car_h_2d / count_car_h


        print("Avg IOU_2d  {0}  with difficulty  EASY: {1}".format(types, avg_car_e_2d))
        print("Avg IOU_2d  {0}  with difficulty  MODERATE: {1}".format(types, avg_car_m_2d))
        print("Avg IOU_2d  {0}  with difficulty  HARD: {1}".format(types, avg_car_h_2d))
        print('--------------------------------------------------------------------------------')
















    
    


"""

import os
import cv2
import errno
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
from lib.DataUtils import *
from lib.Utils import *
from tqdm import tqdm
from lib import Model, ClassAverages



exp_no = 6
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dataset_gt = Dataset('/home/drajani/Downloads/3D-Bounding-Boxes-From-Monocular-Images/Kitti/validation')
all_images = dataset_gt.all_objects()
averages = ClassAverages.ClassAverages()
all_images = dataset_gt.all_objects()


# # in all validation images
Global_Orient= []
for key in tqdm(sorted(all_images.keys())):
    data = all_images[key]
    objects = data['Objects']  # (= no. of images)
    
    # in one GT image
    GT_list = []
    for object in objects:
        label = object.label
        print(label)
        GT_list.append(label)
        
    print('----------------------------------')

    # in one val image
    Val_list = []
    file = open("/home/drajani/Downloads/3D-Bounding-Boxes-From-Monocular-Images/Kitti/results/validation/labels/exp_" + str(exp_no) + "/epoch_10/" + str(key) + ".txt",'r')
    for i in file.readlines():
        num = i.split()
        print(num)
        Val_list.append(num)
        




    add = 0.0
    for i in range(len(Val_list)):
        if (GT_list[i]['Class'] == Val_list[i][0]) and (float(Val_list[i][-1]) >= 0.80):
                error = abs(GT_list[i]['Ry'] - float(Val_list[i][-2]))
                add += (1 + np.cos(error))/2.0
                # break
    average = float(add/len(Val_list))    # All the detections of one image


    Global_Orient.append(average)       # collecting erros for all images sequuentially



print(Global_Orient) # len = no. of images we pass for validation


"""





            




    
    




