import numpy as np
import torch
from config import cfg
import cv2
from utils.utils import denormalize

# def plot_hand_3d(coords_xyz, axis, color_fixed=None, linewidth='1'):
#     """ Plots a hand stick figure into a matplotlib figure. """
#     # colors = np.array([[0., 0., 0.5],
#     #                    [0., 0., 0.73172906],
#     #                    [0., 0., 0.96345811],
#     #                    [0., 0.12745098, 1.],
#     #                    [0., 0.33137255, 1.],
#     #                    [0., 0.55098039, 1.],
#     #                    [0., 0.75490196, 1.],
#     #                    [0.06008855, 0.9745098, 0.90765338],
#     #                    [0.22454143, 1., 0.74320051],
#     #                    [0.40164453, 1., 0.56609741],
#     #                    [0.56609741, 1., 0.40164453],
#     #                    [0.74320051, 1., 0.22454143],
#     #                    [0.90765338, 1., 0.06008855],
#     #                    [1., 0.82861293, 0.],
#     #                    [1., 0.63979666, 0.],
#     #                    [1., 0.43645606, 0.],
#     #                    [1., 0.2476398, 0.],
#     #                    [0.96345811, 0.0442992, 0.],
#     #                    [0.73172906, 0., 0.],
#     #                    [0.5, 0., 0.]])

#     colors = np.array([[0., 0., 1.],
#                        [0., 0., 1.],
#                        [0., 0., 1.],
#                        [0., 0., 1.],
#                        [1., 0., 1.],
#                        [1., 0., 1.],
#                        [1., 0., 1.],
#                        [1., 0., 1.],
#                        [1., 0., 0.],
#                        [1., 0., 0.],
#                        [1., 0., 0.],
#                        [1., 0., 0.],
#                        [0., 1., 0.],
#                        [0., 1., 0.],
#                        [0., 1., 0.],
#                        [0., 1., 0.],
#                        [1., 0.5, 0.],
#                        [1., 0.5, 0.],
#                        [1., 0.5, 0.],
#                        [1., 0.5, 0.]])

#     # define connections and colors of the bones
#     bones = [((1, 0), colors[5, :]),
#              ((2, 1), colors[5, :]),
#              ((3, 2), colors[5, :]),
#              ((4, 3), colors[5, :]),

#              ((0, 5), colors[5, :]),
#              ((5, 6), colors[5, :]),
#              ((6, 7), colors[5, :]),
#              ((7, 8), colors[5, :]),

#              ((0, 9), colors[5, :]),
#              ((9, 10), colors[5, :]),
#              ((10, 11), colors[5, :]),
#              ((11, 12), colors[5, :]),

#              ((0, 13), colors[5, :]),
#              ((13, 14), colors[5, :]),
#              ((14, 15), colors[5, :]),
#              ((15, 16), colors[5, :]),

#              ((0, 17), colors[5, :]),
#              ((17, 18), colors[5, :]),
#              ((18, 19), colors[5, :]),
#              ((19, 20), colors[5, :])]

#     for connection, color in bones:
#         coord1 = coords_xyz[connection[0], :]
#         coord2 = coords_xyz[connection[1], :]
#         coords = np.stack([coord1, coord2])
#         if color_fixed is None:
#             axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=linewidth)
#         else:
#             axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color_fixed, linewidth=linewidth)
#     # for joint_idx in range(coords_xyz.shape[0]):
#     #     axis.text(coords_xyz[joint_idx, 0], coords_xyz[joint_idx, 1], coords_xyz[joint_idx, 2], str(joint_idx))
#     # axis.view_init(azim=-90., elev=90.)
#     # axis.view_init(azim=-130, elev=-120)



def uvd_to_xyz(uvd, intrinsic):
    
    fx = intrinsic[0][0]
    tx = intrinsic[0][2]
    fy = intrinsic[1][1]
    ty = intrinsic[1][2]
    cam = torch.tensor([fx, fy, tx, ty])
    toxyz = torch.zeros([21,3]).to(torch.cuda.current_device())
    cam = torch.tensor(cam) #.to(device)
    
    toxyz[:,2] = toxyz[:,2] + uvd[:, 2]
    toxyz[:,0] = toxyz[:,0] + (((uvd[:, 0]-cam[2]) * uvd[:, 2]) / cam[0])
    toxyz[:,1] = toxyz[:,1] + (((uvd[:, 1]-cam[3]) * uvd[:, 2]) / cam[1])

    return toxyz


def all_uvd_to_xyz(uvd, intrinsic, bb2img_trans, root_depth, _type):
    uvd[:,:,:-1] = uvd[:,:,:-1] / 64 * 256
    tmp = torch.cat((uvd[:,:,:2], torch.ones_like(uvd[:,:,:1]).reshape(-1,21,1)),2)
    uvd[:,:,:2] = torch.matmul(bb2img_trans, tmp.transpose(1,2)).transpose(1,2)[:,:,:2]
    # uvd[:,:,2] = (uvd[:,:,2] / 64 * 2. - 1) * (cfg.bbox_3d_size / 2) + root_depth 
    # uvd[:,:,2] = uvd[:,:,2] / 64 * (400 / 2)  + root_depth

    for i in range(uvd.shape[0]):
        if _type[i] == 'STB':
            uvd[i,:,2:] = (uvd[i,:,2:] / 64 * 2. - 1) * (cfg.bbox_3d_size / 2)
            uvd[i,:,2:] = uvd[i,:,2:] + (root_depth[i]/1000)
        elif _type[i] == 'Rhp':
            # uvd[i,:,2:] = uvd[i,:,2:] * 1000
            uvd[i,:,2:] = (uvd[i,:,2:] / 64 * 2. - 1) * (cfg.bbox_3d_size / 2) + root_depth[i]

    fx = intrinsic[:,0,0].reshape(uvd.shape[0],1)
    tx = intrinsic[:,0,2].reshape(uvd.shape[0],1)
    fy = intrinsic[:,1,1].reshape(uvd.shape[0],1)
    ty = intrinsic[:,1,2].reshape(uvd.shape[0],1)

    # cam = torch.tensor([fx, fy, tx, ty])
    toxyz = torch.zeros([uvd.shape[0],21,3]).to(uvd.device)
    # cam = torch.tensor(cam) #.to(device)
    
    toxyz[:,:,2] = toxyz[:,:,2] + uvd[:,:, 2]
    toxyz[:,:,0] = toxyz[:,:,0] + (((uvd[:,:, 0]-tx) * uvd[:,:, 2]) / fx)
    toxyz[:,:,1] = toxyz[:,:,1] + (((uvd[:,:, 1]-ty) * uvd[:,:, 2]) / fy)

    # for i in range(uvd.shape[0]):
    #     if _type[i] == 'STB':
    #         toxyz[:,:,:2] = toxyz[:,:,:2] / 1000
    #     else:
    #         print("test")

    # toxyz[:,:,0] = toxyz[:,:,0] + (((uvd[:,:, 0]-tx * uvd[:,:, 2])) / fx)
    # toxyz[:,:,1] = toxyz[:,:,1] + (((uvd[:,:, 1]-ty * uvd[:,:, 2])) / fy)

    return toxyz


def xyz_to_uvd(xyz, intrinsic):
    
    # cam = [intrinsic[0, 0], intrinsic[0, 2], intrinsic[1, 1], intrinsic[1, 2]]
    # cam = torch.from_numpy(np.array(cam))
    # new = torch.unsqueeze(cam, 0)
    # uvds = torch.matmul(intrinsic, xyz)
    uvds = torch.matmul(intrinsic, xyz.permute(1,0)).permute(1,0)
    uvds[:,:2] = uvds[:,:2] / uvds[:,2:]

    return uvds


def all_xyz_to_uvd(xyz, intrinsic, bb2img_trans, root_depth, _type):    

    dived_root_depth = torch.zeros(root_depth.shape)

    for i in range(xyz.shape[0]):
        if _type[i] == 'STB':
            dived_root_depth[i] = root_depth[i]/1000
        else:
            dived_root_depth[i] = root_depth[i]

    xyz = xyz / 1000
    uvds = torch.matmul(intrinsic, xyz.permute(0,2,1).clone()).permute(0,2,1).cuda()
    uvds2 = uvds.clone()
    uvds2[:,:,:2] = uvds[:,:,:2] / uvds[:,:,2:]
    
    uvds2[:,:,2] = (uvds2[:,:,2] - dived_root_depth.reshape(-1,1).cuda()) / (0.3/2) + 1 
    uvds2[:,:,2] = uvds2[:,:,2] * 64 / 2

    bb2_img_trans2 = torch.cat((bb2img_trans, torch.zeros((uvds2.shape[0],1,3)).cuda()), axis=1)
    bb2_img_trans2[:,-1,-1] = 1
    bb2_img_trans2 = torch.inverse(bb2_img_trans2)
    bb2_img_trans2 = bb2_img_trans2[:,:-1,:]
    uvds_tmp = torch.cat((uvds2[:,:,:2], torch.ones_like(uvds2[:,:,:1]).reshape(-1,xyz.shape[1],1)),2).cuda()
    uvds2[:,:,:2] = torch.matmul(bb2_img_trans2, uvds_tmp.permute(0,2,1)).permute(0,2,1)[:,:,:2]

    uvds2[:,:,:-1] = uvds2[:,:,:-1] * 64 /256
    
    return uvds2


'''
def all_xyz_to_uvd_before(xyz, intrinsic, bb2img_trans, root_depth, _type):    
    
    # xyz = np.concatenate((xyz[:,:2], np.ones_like(xyz[:,:1])),1)
    # xyz[:,:2] = np.dot(bb2img_trans, xyz.transpose(1,0)).transpose(1,0)[:,:2]

    # root_joint_depth = xyz[self.vertex_num + self.root_joint_idx][2]
    # xyz[:,2] = xyz[:,2] - root_joint_depth

    # xyz[:,0] = xyz[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
    # xyz[:,1] = xyz[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
    # xyz[:,2] = (xyz[:,2] / (cfg.bbox_3d_size / 2) + 1)/2. * cfg.output_hm_shape[0]

    
    # cam = [intrinsic[0, 0], intrinsic[0, 2], intrinsic[1, 1], intrinsic[1, 2]]
    # cam = torch.from_numpy(np.array(cam))
    # new = torch.unsqueeze(cam, 0)
    # uvds = torch.matmul(intrinsic, xyz)

    xyz = xyz / 1000
    uvds = torch.matmul(intrinsic, xyz.permute(0,2,1).clone()).permute(0,2,1).cuda()
    uvds2 = uvds.clone()
    uvds2[:,:,:2] = uvds[:,:,:2] / uvds[:,:,2:]
    
    # uvds2[:,:,2] = (uvds2[:,:,2] - root_depth.reshape(-1,1).cuda()) / (0.3/2) + 1 
    # uvds2[:,:,2] = (uvds2[:,:,2] - root_depth.reshape(-1,1).cuda() / 1000) / (0.3/2) + 1

    for i in range(xyz.shape[0]):
        if _type[i] == 'STB':
            uvds2[i,:,2] = (uvds2[i,:,2] * 1000 - root_depth[i]) / 400 + 0.5
        elif _type[i] == 'Rhp':
            uvds2[i,:,2] = (uvds2[i,:,2] * 1000 - root_depth[i]) * 2.5 + 0.5

    uvds2[:,:,2] = uvds2[:,:,2] * 64 

    # uvd[:,:,2] = (uvd[:,:,2] / 64 * 2. - 1) * (400 / 2)  + root_depth

    bb2_img_trans2 = torch.cat((bb2img_trans, torch.zeros((uvds2.shape[0],1,3)).cuda()), axis=1)
    bb2_img_trans2[:,-1,-1] = 1
    bb2_img_trans2 = torch.inverse(bb2_img_trans2)
    bb2_img_trans2 = bb2_img_trans2[:,:-1,:]
    uvds_tmp = torch.cat((uvds2[:,:,:2], torch.ones_like(uvds2[:,:,:1]).reshape(-1,xyz.shape[1],1)),2).cuda()
    uvds2[:,:,:2] = torch.matmul(bb2_img_trans2, uvds_tmp.permute(0,2,1)).permute(0,2,1)[:,:,:2]

    uvds2[:,:,:-1] = uvds2[:,:,:-1] * 64 / 256
    
    return uvds2
'''

def Esti_plot_hand(coords_hw, axis, color_fixed=None, linewidth='1'):
    """ Thumb (blue), Index finger (pink), Middle finger (red), Ring finger (green), Pinky (Orange)).
        <RGB> Blue: (0,0,255), Pink: (255,0,255), Red: (255,0,0), Green: (0,255,0), Orange: (255,122,0)"""
    
    tensor_type = torch.Tensor([0])

    if type(coords_hw) == type(tensor_type):
        coords_hw = np.array(coords_hw.cpu())
    else:
        coords_hw = coords_hw
        
        
    colors = np.array([[0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.]])

    bones = [((1, 0), colors[5, :]),
             ((2, 1), colors[5, :]),
             ((3, 2), colors[5, :]),
             ((4, 3), colors[5, :]),

             ((0, 5), colors[5, :]),
             ((5, 6), colors[5, :]),
             ((6, 7), colors[5, :]),
             ((7, 8), colors[5, :]),

             ((0, 9), colors[5, :]),
             ((9, 10), colors[5, :]),
             ((10, 11), colors[5, :]),
             ((11, 12), colors[5, :]),

             ((0, 13), colors[5, :]),
             ((13, 14), colors[5, :]),
             ((14, 15), colors[5, :]),
             ((15, 16), colors[5, :]),

             ((0, 17), colors[5, :]),
             ((17, 18), colors[5, :]),
             ((18, 19), colors[5, :]),
             ((19, 20), colors[5, :])]

    for connection, color in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], color_fixed, linewidth=linewidth)


def GT_plot_hand(coords_hw, axis, color_fixed=None, linewidth='4'):
    """ Thumb (blue), Index finger (pink), Middle finger (red), Ring finger (green), Pinky (Orange)).
        <RGB> Blue: (0,0,255), Pink: (255,0,255), Red: (255,0,0), Green: (0,255,0), Orange: (255,122,0)"""
    # coords_hw = coords_hw.cpu().detach()
    colors = np.array([[0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.]])

    bones = [((1, 0), colors[0, :]),
             ((2, 1), colors[0, :]),
             ((3, 2), colors[0, :]),
             ((4, 3), colors[0, :]),

             ((0, 5), colors[0, :]),
             ((5, 6), colors[0, :]),
             ((6, 7), colors[0, :]),
             ((7, 8), colors[0, :]),

             ((0, 9), colors[0, :]),
             ((9, 10), colors[0, :]),
             ((10, 11), colors[0, :]),
             ((11, 12), colors[0, :]),

             ((0, 13), colors[0, :]),
             ((13, 14), colors[0, :]),
             ((14, 15), colors[0, :]),
             ((15, 16), colors[0, :]),

             ((0, 17), colors[0, :]),
             ((17, 18), colors[0, :]),
             ((18, 19), colors[0, :]),
             ((19, 20), colors[0, :])]

    for connection, color in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], color_fixed, linewidth=linewidth)

def Esti_plot_hand_3d(coords_xyz, axis, color_fixed=None, linewidth='4'):
    """ Plots a hand stick figure into a matplotlib figure. """
    # colors = np.array([[0., 0., 0.5],
    #                    [0., 0., 0.73172906],
    #                    [0., 0., 0.96345811],
    #                    [0., 0.12745098, 1.],
    #                    [0., 0.33137255, 1.],
    #                    [0., 0.55098039, 1.],
    #                    [0., 0.75490196, 1.],
    #                    [0.06008855, 0.9745098, 0.90765338],
    #                    [0.22454143, 1., 0.74320051],
    #                    [0.40164453, 1., 0.56609741],
    #                    [0.56609741, 1., 0.40164453],
    #                    [0.74320051, 1., 0.22454143],
    #                    [0.90765338, 1., 0.06008855],
    #                    [1., 0.82861293, 0.],
    #                    [1., 0.63979666, 0.],
    #                    [1., 0.43645606, 0.],
    #                    [1., 0.2476398, 0.],
    #                    [0.96345811, 0.0442992, 0.],
    #                    [0.73172906, 0., 0.],
    #                    [0.5, 0., 0.]])
    coords_xyz = coords_xyz.cpu().detach()
    colors = np.array([[0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.]])

    # define connections and colors of the bones
    bones = [((1, 0), colors[0, :]),
             ((2, 1), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((4, 3), colors[3, :]),

             ((0, 5), colors[4, :]),
             ((5, 6), colors[5, :]),
             ((6, 7), colors[6, :]),
             ((7, 8), colors[7, :]),

             ((0, 9), colors[8, :]),
             ((9, 10), colors[9, :]),
             ((10, 11), colors[10, :]),
             ((11, 12), colors[11, :]),

             ((0, 13), colors[12, :]),
             ((13, 14), colors[13, :]),
             ((14, 15), colors[14, :]),
             ((15, 16), colors[15, :]),

             ((0, 17), colors[16, :]),
             ((17, 18), colors[17, :]),
             ((18, 19), colors[18, :]),
             ((19, 20), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_xyz[connection[0], :]
        coord2 = coords_xyz[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color_fixed, linewidth=linewidth)
    # for joint_idx in range(coords_xyz.shape[0]):
    #     axis.text(coords_xyz[joint_idx, 0], coords_xyz[joint_idx, 1], coords_xyz[joint_idx, 2], str(joint_idx))
    # axis.view_init(azim=-90., elev=90.)
    # axis.view_init(azim=-130, elev=-120)


def GT_plot_hand_3d(coords_xyz, axis, color_fixed=None, linewidth='4'):
    """ Plots a hand stick figure into a matplotlib figure. """
    # colors = np.array([[0., 0., 0.5],
    #                    [0., 0., 0.73172906],
    #                    [0., 0., 0.96345811],
    #                    [0., 0.12745098, 1.],
    #                    [0., 0.33137255, 1.],
    #                    [0., 0.55098039, 1.],
    #                    [0., 0.75490196, 1.],
    #                    [0.06008855, 0.9745098, 0.90765338],
    #                    [0.22454143, 1., 0.74320051],
    #                    [0.40164453, 1., 0.56609741],
    #                    [0.56609741, 1., 0.40164453],
    #                    [0.74320051, 1., 0.22454143],
    #                    [0.90765338, 1., 0.06008855],
    #                    [1., 0.82861293, 0.],
    #                    [1., 0.63979666, 0.],
    #                    [1., 0.43645606, 0.],
    #                    [1., 0.2476398, 0.],
    #                    [0.96345811, 0.0442992, 0.],
    #                    [0.73172906, 0., 0.],
    #                    [0.5, 0., 0.]])

    coords_xyz = coords_xyz.cpu().detach()

    colors = np.array([[0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.]])

    # define connections and colors of the bones
    bones = [((1, 0), colors[0, :]),
             ((2, 1), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((4, 3), colors[3, :]),

             ((0, 5), colors[4, :]),
             ((5, 6), colors[5, :]),
             ((6, 7), colors[6, :]),
             ((7, 8), colors[7, :]),

             ((0, 9), colors[8, :]),
             ((9, 10), colors[9, :]),
             ((10, 11), colors[10, :]),
             ((11, 12), colors[11, :]),

             ((0, 13), colors[12, :]),
             ((13, 14), colors[13, :]),
             ((14, 15), colors[14, :]),
             ((15, 16), colors[15, :]),

             ((0, 17), colors[16, :]),
             ((17, 18), colors[17, :]),
             ((18, 19), colors[18, :]),
             ((19, 20), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_xyz[connection[0], :]
        coord2 = coords_xyz[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color_fixed, linewidth=linewidth)
    # for joint_idx in range(coords_xyz.shape[0]):
    #     axis.text(coords_xyz[joint_idx, 0], coords_xyz[joint_idx, 1], coords_xyz[joint_idx, 2], str(joint_idx))
    # axis.view_init(azim=-90., elev=90.)
    # axis.view_init(azim=-130, elev=-120)

def plot_hand_rhp(coords_hw, axis, color_fixed=None, linewidth='2'):
    """ Thumb (blue), Index finger (pink), Middle finger (red), Ring finger (green), Pinky (Orange)).
        <RGB> Blue: (0,0,255), Pink: (255,0,255), Red: (255,0,0), Green: (0,255,0), Orange: (255,122,0)"""

    colors = np.array([[0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.]])

    bones = [((4, 0), colors[0, :]),
             ((2, 1), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((4, 3), colors[3, :]),

             ((0, 8), colors[4, :]),
             ((5, 6), colors[5, :]),
             ((6, 7), colors[6, :]),
             ((7, 8), colors[7, :]),

             ((0, 12), colors[8, :]),
             ((9, 10), colors[9, :]),
             ((10, 11), colors[10, :]),
             ((11, 12), colors[11, :]),

             ((0, 16), colors[12, :]),
             ((13, 14), colors[13, :]),
             ((14, 15), colors[14, :]),
             ((15, 16), colors[15, :]),

             ((0, 20), colors[16, :]),
             ((17, 18), colors[17, :]),
             ((18, 19), colors[18, :]),
             ((19, 20), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], color_fixed, linewidth=linewidth)
            for i in range(21):
                axis.scatter(coords_hw[i, 0], coords_hw[i, 1], c=color_fixed, marker='.')

def plot_hand(coords_hw, axis, color_fixed=None, linewidth='2'):
    """ Thumb (blue), Index finger (pink), Middle finger (red), Ring finger (green), Pinky (Orange)).
        <RGB> Blue: (0,0,255), Pink: (255,0,255), Red: (255,0,0), Green: (0,255,0), Orange: (255,122,0)"""

    colors = np.array([[0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.]])

    bones = [((1, 0), colors[0, :]),
             ((2, 1), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((4, 3), colors[3, :]),

             ((0, 5), colors[4, :]),
             ((5, 6), colors[5, :]),
             ((6, 7), colors[6, :]),
             ((7, 8), colors[7, :]),

             ((0, 9), colors[8, :]),
             ((9, 10), colors[9, :]),
             ((10, 11), colors[10, :]),
             ((11, 12), colors[11, :]),

             ((0, 13), colors[12, :]),
             ((13, 14), colors[13, :]),
             ((14, 15), colors[14, :]),
             ((15, 16), colors[15, :]),

             ((0, 17), colors[16, :]),
             ((17, 18), colors[17, :]),
             ((18, 19), colors[18, :]),
             ((19, 20), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], color_fixed, linewidth=linewidth)
            for i in range(21):
                axis.scatter(coords_hw[i, 0], coords_hw[i, 1], c=color_fixed, marker='.')

def plot_hand_stb(coords_hw, axis, color_fixed=None, linewidth='2'):
    """ Thumb (blue), Index finger (pink), Middle finger (red), Ring finger (green), Pinky (Orange)).
        <RGB> Blue: (0,0,255), Pink: (255,0,255), Red: (255,0,0), Green: (0,255,0), Orange: (255,122,0)"""

    colors = np.array([[0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.],
                       [1., 0.5, 0.]])

    bones = [
             ((0, 17), colors[0, :]),
             ((17, 18), colors[1, :]),
             ((18, 19), colors[2, :]),
             ((19, 20), colors[3, :]),
             
             ((0, 5), colors[4, :]),
             ((5, 6), colors[5, :]),
             ((6, 7), colors[6, :]),
             ((7, 8), colors[7, :]),

             ((0, 9), colors[8, :]),
             ((9, 10), colors[9, :]),
             ((10, 11), colors[10, :]),
             ((11, 12), colors[11, :]),

             ((0, 13), colors[12, :]),
             ((13, 14), colors[13, :]),
             ((14, 15), colors[14, :]),
             ((15, 16), colors[15, :]),

             ((1, 0), colors[16, :]),
             ((2, 1), colors[17, :]),
             ((3, 2), colors[18, :]),
             ((4, 3), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], color_fixed, linewidth=linewidth)
            for i in range(21):
                axis.scatter(coords_hw[i, 0], coords_hw[i, 1], c=color_fixed, marker='.')



