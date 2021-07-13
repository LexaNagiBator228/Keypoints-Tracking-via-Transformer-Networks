
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
#from models import SuperPointNet, SuperPointFrontend
#from superglue import SuperGlue
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
from os import listdir
from os.path import isfile, join


def draw_interest_points(img, points):
    """ Convert img in RGB and draw in green the interest points """
    img_rgb = np.stack([img, img, img], axis=2)
    for i in range(points.shape[0]):
        cv.circle(img_rgb, (points[i][0], points[i][1]), 5, (0, 255, 0), -1)
    return img_rgb



im_list = ['/1.ppm']


class image_match(torch.nn.Module):
    def __init__(self, num_points =-1, config_match ={}, config_path={}):
        super(image_match, self).__init__()

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.f_net  = Linear_net_small(4,256,8,256,256,4).to(device)



        path_m = './weights/model_new_temp40.pth'
        params = torch.load(path_m, map_location=device)
        self.f_net.load_state_dict(torch.load(path_m, map_location=device))
        self.f_net = self.f_net.to(device)

        self.fine_model = fine_mod(4,256,4).to(device)
        path_m = './weights/model_temp_2nd40.pth'
        params = torch.load(path_m, map_location=device)
        self.fine_model.load_state_dict(torch.load(path_m, map_location=device))
        self.fine_model = self.fine_model.to(device)

        self.fine_model.eval()   
        self.f_net.eval()
        default_config = {
                'descriptor_dim': 256,
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': num_points,
                'remove_borders': 8,
            }

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


        self.net = SuperPoint(default_config).to(device)
        self.net.eval()
        

    def forward(self, im1, im2, add_vis = False):
        origin_im1 = im1
        origin_im2 = im2
        

        device = self.device
        all_image_count =0
        print('poshel nah )0)0)))')
        count_try = 0
        acc_s = []
        acc_my = []
        acc_fine = []
        t_s = []
        t_my = []
        key_num_my = []
        key_num_sg = []


        with torch.no_grad():

            im1 = cv2.resize(im1,(512,512))
            im2 = cv2.resize(im2,(512,512))

            #### match ###

            im1_torch = torch.from_numpy(im1).to(device).float()/255.
            im2_torch = torch.from_numpy(im2).to(device).float()/255.
            im1_torch = im1_torch[None, None]
            im2_torch = im2_torch[None, None]
            
            ###########   


            out, desc = self.net(None, im1_torch )
            ps1 = out['keypoints']
            ps1 = np.array(ps1[0]).astype(float)
            kpts0 = ps1
            ps1 = torch.from_numpy(ps1).float().to(device)
            ps1 = ps1.view(1,ps1.shape[0], ps1.shape[1])
            d1 = out['descriptors'][0]

            d1 = d1.view(1,d1.shape[0], d1.shape[1])
            d1 = d1.transpose(1,2).to(device)

            out, desk2 = self.net(None, im2_torch )
            #print(desk2.shape)
            batch_size =1 
            desk2 = desk2.transpose(2,3).contiguous().view(batch_size,256,-1)
            d2  = desk2.transpose(1,2)
            
            radius = 256
            R= 2*radius//8
            points_net_8 = np.zeros((1,R*R,2))
            count = 0
            for i in range(R):
                for j in range(R):
                    points_net_8[:,count] = np.array([i,j])
                    count+=1
            save_p = points_net_8.copy()
            kpts1 = points_net_8[0]* 8 +4
            save_p = torch.from_numpy(save_p).float().to(device)
            test_p = save_p * 8 + 5#save_p.view(1,save_p.shape[0],2)*8
            points_net = torch.from_numpy(points_net_8).to(device).float()
            points_net = points_net + 0.5
            points_net = points_net/float(32)

            t = time.time()

            ps1 = ps1/float(radius)


            ps1_input = ps1.repeat([1,1,2])
            total_point_num = ps1_input.shape[1]
            ps2_input = points_net.repeat([1,1,2])
            
            #starter.record()
            out, point_map = self.f_net(None, d2, d1,  ps2_input, ps1_input, ocl= True)
            #ender.record()
            #torch.cuda.synchronize()

            t1 = time.time() - t 

            res = torch.argmax(point_map, dim=2)


            res = res.cpu().numpy()

            res = res[0]
            res[res<0.9] = -1
            res[res==kpts1.shape[0]] = -1

         
            res_t = torch.from_numpy(res).to(device)
            real_points_ind = res_t[res_t >=0]
            
            f_befor = d1[0,res_t >=0]
            f_after = out[0,res_t >=0]
            ### done for feature extraction 
            p_net = points_net[0]*32
            p_pos = p_net[real_points_ind]
            remove_boundary1 = p_pos>1.5
            remove_boundary2 = p_pos < R -1.5
            remove_boundary1 = remove_boundary1[:,0] * remove_boundary1[:,1]
            remove_boundary2 = remove_boundary2[:,0] * remove_boundary2[:,1]
            remove_boundary = remove_boundary1 * remove_boundary2

            funcs = [lambda x: x - R -1,lambda x: x - R, lambda x: x - R +1,
                lambda x: x -1, lambda x: x , lambda x: x +1,
                lambda x: x + R -1, lambda x: x + R, lambda x: x + R +1  ]
            d = [f(real_points_ind) for i, f in enumerate(funcs)]
            d = torch.stack(d, dim=1)
            #print(d.shape)
            d = d[remove_boundary,:]
            #print(d2.shape)
            new_9_f = d2[0,d]

            f_befor = f_befor[remove_boundary,:]
            f_after = f_after[remove_boundary,:]

            im_points, f_origin, f_after = new_9_f.to(device), f_befor.to(device), f_after.to(device)
            result_fine = self.fine_model(im_points, f_origin, f_after)

            my_p1 = kpts0[res>=0]
            my_p2 = kpts1[res[res>=0]]
            

            my_p2_fine = np.zeros_like(my_p2) 
            my_p2_fine[:] = my_p2[:]
            my_p2_fine[remove_boundary,:] = my_p2_fine[remove_boundary,:] - result_fine.cpu().numpy()*4

            if(add_vis):
                vis = np.concatenate((im1, im2), axis=1)
                img_rgb = np.stack([vis, vis, vis], axis=2)
                for i in range(my_p1.shape[0]):
                    p__1 = my_p1[i].astype(int)
                    p__2 = my_p2_fine[i].astype(int)
                    color = (255*(i%3==0), 255*(i%3==1), 255*(i%3==3))
                    img_rgb = cv2.line(img_rgb, (p__1[0], p__1[1]), (p__2[0]+ 512, p__2[1]), color, 1)
                    cv.circle(img_rgb, (p__1[0], p__1[1]), 5, color, -1)
                    cv.circle(img_rgb, (p__2[0] + 512, p__2[1]), 5, color, -1)
                cv2.imshow('vis',vis)
                cv2.imshow('img_rgb',img_rgb)
                cv2.waitKey(0)
            return my_p1, my_p2_fine



class Points_tracking(torch.nn.Module):
    def __init__(self, num_points =-1, config_match ={}, config_path={}):
        super(Points_tracking, self).__init__()

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.f_net  = Linear_net_small(4,256,8,256,256,4).to(device)

        path_m = './weights/model_new_temp40.pth'
        params = torch.load(path_m, map_location=device)
        self.f_net.load_state_dict(torch.load(path_m, map_location=device))
        self.f_net = self.f_net.to(device)

        self.fine_model = fine_mod(4,256,4).to(device)
        path_m = './weights/model_temp_2nd40.pth'
        params = torch.load(path_m, map_location=device)
        self.fine_model.load_state_dict(torch.load(path_m, map_location=device))
        self.fine_model = self.fine_model.to(device)

        self.fine_model.eval()   
        self.f_net.eval()
        default_config = {
                'descriptor_dim': 256,
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': num_points,
                'remove_borders': 8,
            }




        self.net = SuperPoint(default_config).to(device)
        self.net.eval()
        

    def forward(self, im1, im2,p_to_track, add_vis = False):
        origin_im1 = im1
        origin_im2 = im2
        

        device = self.device
        all_image_count =0
        print('poshel nah )0)0)))')
        count_try = 0
        acc_s = []
        acc_my = []
        acc_fine = []
        t_s = []
        t_my = []
        key_num_my = []
        key_num_sg = []

        kpts0 = p_to_track
        p1 = torch.from_numpy(p_to_track).to(device).view(1,p_to_track.shape[-2],p_to_track.shape[-1])
        


        with torch.no_grad():

            im1 = cv2.resize(im1,(512,512))
            im2 = cv2.resize(im2,(512,512))

            #### match ###

            im1_torch = torch.from_numpy(im1).to(device).float()/255.
            im2_torch = torch.from_numpy(im2).to(device).float()/255.
            im1_torch = im1_torch[None, None]
            im2_torch = im2_torch[None, None]
            
            ###########   

            ps1 = p1
            print(p1.shape)
            out, desc = self.net(None, im1_torch ,my_p = p1 )
            print()
            #d1 = out['descriptors'].transpose(2,3).contiguous().view(1,256,-1)
            d1 = out['descriptors'].transpose(1,2).to(device)
            print(d1.shape)
            out, desk2 = self.net(None, im2_torch )
            #print(desk2.shape)
            batch_size =1 
            desk2 = desk2.transpose(2,3).contiguous().view(batch_size,256,-1)
            d2  = desk2.transpose(1,2)
            
            radius = 256
            R= 2*radius//8
            points_net_8 = np.zeros((1,R*R,2))
            count = 0
            for i in range(R):
                for j in range(R):
                    points_net_8[:,count] = np.array([i,j])
                    count+=1
            save_p = points_net_8.copy()
            kpts1 = points_net_8[0]* 8 +4
            save_p = torch.from_numpy(save_p).float().to(device)
            test_p = save_p * 8 + 5#save_p.view(1,save_p.shape[0],2)*8
            points_net = torch.from_numpy(points_net_8).to(device).float()
            points_net = points_net + 0.5
            points_net = points_net/float(32)

            t = time.time()

            ps1 = ps1/float(radius)


            ps1_input = ps1.repeat([1,1,2])
            total_point_num = ps1_input.shape[1]
            ps2_input = points_net.repeat([1,1,2])
            print(d1.shape)
            print(ps1_input.shape)
            #starter.record()
            out, point_map = self.f_net(None, d2, d1,  ps2_input, ps1_input, ocl= True)
            #ender.record()
            #torch.cuda.synchronize()

            t1 = time.time() - t 

            res = torch.argmax(point_map, dim=2)


            res = res.cpu().numpy()

            res = res[0]
            res[res<0.2] = -1
            res[res==kpts1.shape[0]] = -1

         
            res_t = torch.from_numpy(res).to(device)
            real_points_ind = res_t[res_t >=0]
            
            f_befor = d1[0,res_t >=0]
            f_after = out[0,res_t >=0]
            ### done for feature extraction 
            p_net = points_net[0]*32
            p_pos = p_net[real_points_ind]
            remove_boundary1 = p_pos>1.5
            remove_boundary2 = p_pos < R -1.5
            remove_boundary1 = remove_boundary1[:,0] * remove_boundary1[:,1]
            remove_boundary2 = remove_boundary2[:,0] * remove_boundary2[:,1]
            remove_boundary = remove_boundary1 * remove_boundary2

            funcs = [lambda x: x - R -1,lambda x: x - R, lambda x: x - R +1,
                lambda x: x -1, lambda x: x , lambda x: x +1,
                lambda x: x + R -1, lambda x: x + R, lambda x: x + R +1  ]
            d = [f(real_points_ind) for i, f in enumerate(funcs)]
            d = torch.stack(d, dim=1)
            #print(d.shape)
            d = d[remove_boundary,:]
            #print(d2.shape)
            new_9_f = d2[0,d]

            f_befor = f_befor[remove_boundary,:]
            f_after = f_after[remove_boundary,:]

            im_points, f_origin, f_after = new_9_f.to(device), f_befor.to(device), f_after.to(device)
            result_fine = self.fine_model(im_points, f_origin, f_after)

            my_p1 = kpts0[res>=0]
            my_p2 = kpts1[res[res>=0]]
            

            my_p2_fine = np.zeros_like(my_p2) 
            my_p2_fine[:] = my_p2[:]
            my_p2_fine[remove_boundary,:] = my_p2_fine[remove_boundary,:] - result_fine.cpu().numpy()*4

            if(add_vis):
                vis = np.concatenate((im1, im2), axis=1)
                img_rgb = np.stack([vis, vis, vis], axis=2)
                for i in range(my_p1.shape[0]):
                    p__1 = my_p1[i].astype(int)
                    p__2 = my_p2_fine[i].astype(int)
                    color = (255*(i%3==0), 255*(i%3==1), 255*(i%3==3))
                    img_rgb = cv2.line(img_rgb, (p__1[0], p__1[1]), (p__2[0]+ 512, p__2[1]), color, 1)
                    cv.circle(img_rgb, (p__1[0], p__1[1]), 5, color, -1)
                    cv.circle(img_rgb, (p__2[0] + 512, p__2[1]), 5, color, -1)
                cv2.imshow('vis',vis)
                cv2.imshow('img_rgb',img_rgb)
                cv2.waitKey(0)
            return my_p1, my_p2_fine

