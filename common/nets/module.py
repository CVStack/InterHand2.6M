# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.nn import functional as F
from config import cfg
from nets.layer import make_linear_layers, make_conv_layers, make_deconv_layers, make_upsample_layers
from nets.resnet import ResNetBackbone
import math

class BackboneNet(nn.Module):
    def __init__(self):
        super(BackboneNet, self).__init__()
        self.resnet = ResNetBackbone(cfg.resnet_type)
    
    def init_weights(self):
        self.resnet.init_weights()

    def forward(self, img):
        img_feat = self.resnet(img)
        return img_feat

class PoseNet(nn.Module):
    def __init__(self, joint_num):
        super(PoseNet, self).__init__()
        self.joint_num = joint_num 
        
        self.joint_deconv = make_deconv_layers([2048,256,256,256])
        self.joint_conv = make_conv_layers([256,self.joint_num*cfg.output_hm_shape[0]],kernel=1,stride=1,padding=0,bnrelu_final=False)    
    
    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 1)
        accu = heatmap1d * torch.arange(cfg.output_root_hm_shape).float().cuda()[None,:]
        coord = accu.sum(dim=1)
        return coord

    def forward(self, img_feat):
        joint_img_feat = self.joint_deconv(img_feat)
        joint_heatmap3d = self.joint_conv(joint_img_feat).view(-1,self.joint_num,cfg.output_hm_shape[0],cfg.output_hm_shape[1],cfg.output_hm_shape[2])

        return joint_heatmap3d
