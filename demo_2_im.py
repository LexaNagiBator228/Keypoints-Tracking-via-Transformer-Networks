import argparse
import glob
import numpy as np
import os
import time
import torch.utils.data
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor
import json
import torch.nn.functional as F
import time
import cv2 as cv
import torch

from pathlib import Path

import argparse

from PIL import Image
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader
from models_S.superpoint import SuperPoint
from model import Linear_net_small, fine_mod
from torchvision import datasets, models, transforms
from models_S.matching import Matching
from os import listdir
from os.path import isfile, join, isdir
from numpy.linalg import inv
from keypoint_match import image_match


parser = argparse.ArgumentParser()
parser.add_argument('--image1_path', type=str,
                        help='The image path of the testing image', default='./media/holl1.jpg')
parser.add_argument('--image2_path', type=str,
                        help='The path to the second image', default='./media/holl2.jpg' ) 
parser.add_argument("-s","--save", 
                        help='Whether to show results', action="store_true") 

      
parser.add_argument('--model_dir', type=str,
                        help='The directory of trained model',
                        default='./weights/model_new_temp40.pth')         

args = parser.parse_args()
print(args)



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
num_points =400
im_match  = image_match(num_points = num_points).to(device)
im1_path = args.image1_path
im2_path = args.image2_path
print('Im path',im1_path,im2_path)
im1 = cv2.resize(cv2.imread(im1_path, 0),(512,512))
im2 = cv2.resize(cv2.imread(im2_path, 0),(512,512))

p1, p2 = im_match(im1, im2, False)

E, mask = cv2.findEssentialMat(
        p1, p2, threshold = 5.0,  method=cv2.RANSAC)


p1_after = p1[mask[:,0]==1]
p2_after = p2[mask[:,0]==1]
#print(p1_after.shape)

vis = np.concatenate((im1, im2), axis=1)
img_rgb = np.stack([vis, vis, vis], axis=2)
for i in range(p1_after.shape[0]):
    p__1 = p1_after[i].astype(int)
    p__2 = p2_after[i].astype(int)

    img_rgb = cv2.line(img_rgb, (p__1[0], p__1[1]), (p__2[0]+ 512, p__2[1]), (0, 255, 0), 1)
    cv.circle(img_rgb, (p__1[0], p__1[1]), 5, (0, 255, 0), -1)
    cv.circle(img_rgb, (p__2[0] + 512, p__2[1]), 5, (0, 255, 0), -1)



config = {
                'superpoint': {
                    'nms_radius': 4,
                    'keypoint_threshold': 0.005,
                    'max_keypoints': num_points,
                    'remove_borders':8
                },
                'superglue': {
                    'weights': 'indoor',
                    'sinkhorn_iterations': 80,
                    'match_threshold': 0.2,
                }
            }




cv2.imshow('img_rgb',img_rgb)
cv2.waitKey(0)
if (args.save):
    cv2.imwrite('./results/res_match.jpg',img_rgb )