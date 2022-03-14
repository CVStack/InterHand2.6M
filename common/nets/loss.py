# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from config import cfg
import math

class JointHeatmapLoss(nn.Module):
    def __ini__(self):
        super(JointHeatmapLoss, self).__init__()

    def forward(self, joint_out, joint_gt):
        loss = (joint_out - joint_gt) ** 2 
        return loss

