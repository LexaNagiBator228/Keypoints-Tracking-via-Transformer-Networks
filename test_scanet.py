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
from model import  Linear_net_small, fine_mod
from torchvision import datasets, models, transforms

from os import listdir
from os.path import isfile, join, isdir
from numpy.linalg import inv
from keypoint_match import image_match


class ScanNet1488(Dataset):
    def __init__(self, path): 
        print('load ScanNet')
        #print(listdir(path))
        self.data_scenes = [ join(path, f) for f in listdir(path) if isdir(join(path, f))]

        print(self.data_scenes)
        self.data = []
        for scene in self.data_scenes:
            scene_or = scene
            scene = scene + '/color'
            im_names = [f for f in listdir(scene) if isfile(join(scene, f))]

            im_ind = [int(name.split('.')[0]) for name in im_names]
            im_ind.sort()
            #print(im_ind)
            im_bool = np.array(im_ind)
            im_bool[:-1] = np.abs(im_bool[:-1] - im_bool[1:])
            #print(im_bool)
            im_bool = im_bool<=90
            #print(im_bool) 
            for i in range(im_bool.shape[0]):
                if(im_bool[i]):
                    self.data.append({"scene_dir":scene_or, "im1":im_ind[i], "im2":im_ind[i+1]})
            #print(self.data)
            #input('ss')
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        im_inf = self.data[index]
        scene_dir = im_inf['scene_dir']
        im_0_ind = im_inf['im1']
        im_1_ind = im_inf['im2']
        im_0 = cv.imread(scene_dir + '/color/' + str(im_0_ind) + '.jpg',0 )
        im_1 = cv.imread(scene_dir + '/color/' + str(im_1_ind) + '.jpg',0 )
        im_0 = cv2.resize(im_0,(512,512))
        im_1 = cv2.resize(im_1,(512,512))
        K = np.loadtxt(scene_dir+ '/intrinsic/'+'intrinsic_color.txt')
        K = K[:3,:3]
        origin_fx  = K[0,2]
        origin_fy  = K[1,2]
        K[0,2]= 256.
        K[1,2]= 256.
        pose0  = np.loadtxt(scene_dir+ '/pose/'+ str(im_0_ind)+'.txt')
        pose1  = np.loadtxt(scene_dir+ '/pose/'+ str(im_1_ind)+'.txt')
        T0to1 = np.matmul(pose1, inv(pose0))
        T1to0 = np.matmul(pose0, inv(pose1))

        data = {'im0': im_0,
                'im1': im_1,
                'K': K,
                'T0to1': T0to1,
                'T1to0':T1to0}
        return data
        
dataset = ScanNet1488('/home/nagibator/SLAM/scannet_test_1500')
print(len(dataset))
result_auc = []
sp_result_auc = []
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
num_points =256
im_match  = image_match(num_points = num_points).to(device)
for i_ind in range(len(dataset)):
#for i_ind in range(20):
    data = dataset.__getitem__(i_ind )

    #im1 = cv2.resize(cv2.imread('./300.jpg', 0),(512,512))
    #im2 = cv2.resize(cv2.imread('./360.jpg', 0),(512,512))
    #im1 = cv2.resize(cv2.imread('./comp1.jpg', 0),(512,512))
    #im2 = cv2.resize(cv2.imread('./comp2.jpg', 0),(512,512))
    im1 = data['im0']
    im2 = data['im1']
    

    
    
    p1, p2 = im_match(im1, im2, False)
    #print(p1)
    #print(p2)

    E, mask = cv2.findEssentialMat(
            p1, p2, threshold = 5.0,  method=cv2.RANSAC)

    #E, mask = cv2.findEssentialMat(
    #        p1, p2, np.eye(3),  method=cv2.RANSAC)
    #print(E)

    #print(mask.shape)

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


    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, p1, p2, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n


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


    print('My points',p1_after.shape)



    def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
        # angle error between 2 vectors
        t_gt = T_0to1[:3, 3]
        n = np.linalg.norm(t) * np.linalg.norm(t_gt)
        t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
        t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
        if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
            t_err = 0

        # angle error between 2 rotation matrices
        R_gt = T_0to1[:3, :3]
        cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
        cos = np.clip(cos, -1., 1.)  # handle numercial errors
        R_err = np.rad2deg(np.abs(np.arccos(cos)))
        R_err = np.minimum(R_err, 180 - R_err)

        return t_err, R_err

    R, t, inliers = ret
    t_err, R_err = relative_pose_error(data['T0to1'], R, t, ignore_gt_t_thr=0.0)
    print(t_err)
    
    print('My error',R_err)
    result_auc.append(R_err)
    ### SP part ###
    from models_S.matching import Matching
    matching = Matching(config).eval().to(device)


    im1_torch = torch.from_numpy(im1).to(device).float()/255.
    im2_torch = torch.from_numpy(im2).to(device).float()/255.
    im1_torch = im1_torch[None, None]
    im2_torch = im2_torch[None, None]

    pred, t_super = matching({'image0': im1_torch, 'image1': im2_torch})
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    s_p1 = kpts0[matches>=0]
    s_p2 = kpts1[matches[matches>=0]]

    E, mask = cv2.findEssentialMat(
            s_p1, s_p2,  threshold = 5.0, method=cv2.RANSAC)

    s_p1_after = s_p1[mask[:,0]==1]
    s_p2_after = s_p2[mask[:,0]==1]
    #print(s_p2_after.shape)
    #print(s_p1_after.shape)
    #print(mask.shape)

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, s_p1, s_p2, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n
    R, t, inliers = ret
    t_err, R_err = relative_pose_error(data['T0to1'], R, t, ignore_gt_t_thr=0.0)
    
    
    print('SP error',R_err)
    sp_result_auc.append(R_err)

    
    #cv2.imshow('img_rgb',img_rgb)
    #cv2.imshow('img_rgb_s',img_rgb_s)
    #cv2.waitKey(0)


result_auc = np.array(result_auc)
sp_result_auc = np.array(sp_result_auc)
print('auc 5', np.sum(result_auc<5)/result_auc.shape[0])
print('auc 10', np.sum(result_auc<10)/result_auc.shape[0])
print('auc 20', np.sum(result_auc<20)/result_auc.shape[0]) 

print('auc 5', np.sum(sp_result_auc<5)/sp_result_auc.shape[0])
print('auc 10', np.sum(sp_result_auc<10)/sp_result_auc.shape[0])
print('auc 20', np.sum(sp_result_auc<20)/sp_result_auc.shape[0]) 