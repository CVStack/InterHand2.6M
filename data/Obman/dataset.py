import os
import os.path as osp
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.transforms import *
from utils.visualization import *
import cv2
import random
import json
import pickle
import math
import copy
from pycocotools.coco import COCO
import tqdm

import sys

from config import cfg
from utils.preprocessing import load_img, process_bbox, process_bbox2, augmentation, transform_input_to_output_space, trans_point2d, load_skeleton
from utils.transforms import pixel2cam
from utils.vis import vis_keypoints, vis_3d_keypoints

import os
import torch.nn.functional as F


class Dataset(torch.utils.data.Dataset):

    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.root_path = osp.join('/home/user/hspark/InterHand2.6M/data/Obman/data')
        
        self.joint_num = 21
        self.root_joint_idx = 0

        self.datalist = self.load_data()
        self.skeleton = load_skeleton(osp.join(self.root_path, 'skeleton.txt'), self.joint_num)
        # self.datalist = self.datalist[:256]

    def load_data(self):
        if self.data_split == 'train':
            with open(osp.join(self.root_path, 'train/result_training.pickle'), 'rb') as f:
                data = pickle.load(f)
                self.data_path = osp.join(self.root_path, 'train', 'rgb')
        
        else:
            with open(osp.join(self.root_path, 'test/result_testing.pickle'), 'rb') as f:
                data = pickle.load(f)
                self.data_path = osp.join(self.root_path, 'test', 'rgb')

        filelist = ["00029634.jpg", "00045305.jpg", "00047433.jpg", "00119666.jpg", "00186570.jpg"]
        
        datalist = []

        for idx in tqdm.tqdm(range(len(data['annotations']))):
            ann = data['annotations'][idx]
            img = data['images'][idx]

            image_id = img['id']
            img_path = osp.join(self.data_path, img['file_name'])

            if not osp.exists(img_path) or img['file_name'] in filelist:
                continue

            img_shape = (img['height'], img['width'])
            cam_param, joint_cam, joint_img = img['param'], ann['xyz'], ann['uvd']
            focal, princpt = np.array([cam_param[0,0], cam_param[1,1]],dtype=np.float32), np.array([cam_param[0,2], cam_param[1,2]],dtype=np.float32)
            # joint_cam = pixel2cam(joint_img, focal, princpt)

            # bbox = ann['bbox']
            bbox = process_bbox2(joint_img)
            
            if True in ((joint_img[:, :2] < 0) | (joint_img[:, :2] > img_shape[0])):
                continue           
            if (bbox[0] < 0) or (bbox[0] + bbox[2] > img['width']) or (bbox[1] < 0) or (bbox[1] + bbox[3] > img['height']):
                continue
            # if True in ((joint_img[:, 0] < 0) | (joint_img[:, 0] > img_shape[1])):
            #     continue
            # if True in ((joint_img[:, 1] < 0) | (joint_img[:, 1] > img_shape[0])):
            #     continue

            if bbox is None: continue
            abs_depth = joint_cam[0,2]
            side = ann['side']

            cam_param = {'focal': focal, 'princpt': princpt}
            joint = {'cam_coord': joint_cam, 'img_coord': joint_img}
            elem = {'img_path': img_path, 'bbox': bbox, 'cam_param': cam_param, 'joint': joint, 'abs_depth': abs_depth, 'side' : side, 'img_shape': img_shape}

            datalist.append(elem)
        return datalist
    
    def __len__(self):
        return len(self.datalist)

    
    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, bbox, joint = data['img_path'], data['bbox'], data['joint']
        height, width = data['img_shape']
        joint_coord = joint['img_coord'].copy()
        side = data['side']

        # img, mask laod     
        img = load_img(img_path)

        if side == 'left':
            img = cv2.flip(img, 1)
            joint[:, 0] = width - joint_coord[:, 0]
            bbox[0] = width - bbox[0] - bbox[2]

        img, joint_coord, inv_trans = augmentation(img, bbox, joint_coord, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
        rel_root_depth = np.zeros((1),dtype=np.float32)
        root_valid = np.zeros((1),dtype=np.float32)
        # transform to output heatmap space
        joint_coord, rel_root_depth, root_valid = transform_input_to_output_space(joint_coord,rel_root_depth, root_valid, self.root_joint_idx)
        
        inputs = {'img': img}
        targets = {'joint_coord': joint_coord, 'rel_root_depth': rel_root_depth}
        meta_info = {'root_valid': root_valid, 'inv_trans': inv_trans}

        return inputs, targets, meta_info


    def evaluate(self, preds):

        print() 
        print('Evaluation start...')

        gts = self.datalist
        preds_joint_coord, inv_trans = preds['joint_coord'], preds['inv_trans']
        assert len(gts) == len(preds_joint_coord)
        sample_num = len(gts)
        
        mpjpe = [[] for _ in range(self.joint_num)] # treat right and left hand identical
        x_mpjpe = [[] for _ in range(self.joint_num)]
        y_mpjpe = [[] for _ in range(self.joint_num)]
        z_mpjpe = [[] for _ in range(self.joint_num)]

        for n in range(sample_num):
            data = gts[n]
            bbox, cam_param, joint = data['bbox'], data['cam_param'], data['joint']
            focal = cam_param['focal']
            princpt = cam_param['princpt']
            gt_joint_coord = joint['cam_coord']

            # restore coordinates to original space
            pred_joint_coord_img = preds_joint_coord[n].copy()
            pred_joint_coord_img[:,0] = pred_joint_coord_img[:,0]/cfg.output_hm_shape[2]*cfg.input_img_shape[1]
            pred_joint_coord_img[:,1] = pred_joint_coord_img[:,1]/cfg.output_hm_shape[1]*cfg.input_img_shape[0]
            for j in range(self.joint_num):
                pred_joint_coord_img[j,:2] = trans_point2d(pred_joint_coord_img[j,:2],inv_trans[n])
            pred_joint_coord_img[:,2] = (pred_joint_coord_img[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)
            pred_joint_coord_img[:,2] = pred_joint_coord_img[:,2] + data['abs_depth']

            # back project to camera coordinate system
            pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)

            # root joint alignment
            pred_joint_coord_cam = pred_joint_coord_cam - pred_joint_coord_cam[self.root_joint_idx]
            gt_joint_coord = gt_joint_coord - gt_joint_coord[self.root_joint_idx]
    
            # mpjpe save
            for j in range(self.joint_num):
                mpjpe[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
                x_mpjpe[j].append(np.abs(pred_joint_coord_cam[j, 0] - gt_joint_coord[j, 0]))
                y_mpjpe[j].append(np.abs(pred_joint_coord_cam[j, 1] - gt_joint_coord[j, 1]))
                z_mpjpe[j].append(np.abs(pred_joint_coord_cam[j, 2] - gt_joint_coord[j, 2]))
                
            vis = False
            if vis:
                img_path = data['img_path']
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                _img = cvimg[:,:,::-1].transpose(2,0,1)
                vis_kps = pred_joint_coord_img.copy()
                filename = 'out_' + str(n) + '.jpg'
                vis_keypoints(_img, vis_kps, self.skeleton[:self.joint_num], filename)

            vis = False
            if vis:
                filename = 'out_' + str(n) + '_3d.png'
                vis_3d_keypoints(pred_joint_coord_cam, self.skeleton[:self.joint_num], filename)

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num):
            mpjpe[j] = np.mean(np.stack(mpjpe[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % (mpjpe[j] * 1000))

        print(eval_summary)
        print('MPJPE: %.2f' % (np.mean(mpjpe) * 1000))
        print('x_MPJPE: %.2f' % (np.mean(x_mpjpe) * 1000))
        print('y_MPJPE: %.2f' % (np.mean(y_mpjpe) * 1000))
        print('z_MPJPE: %.2f' % (np.mean(z_mpjpe) * 1000))
    