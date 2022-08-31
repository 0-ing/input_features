# -*- coding:utf-8 -*-
"""
作者:wensong
日期:2022年08月26日
"""

# For Point Cloud
from point_cloud.preprocess import read_and_prep_points

import numpy as np
from utils.data_util import get_pos, get_frame_info

from utils.kitti_util import read_calib_file

from utils.data_util import generate_seq_dets, generate_seq_gts, generate_seq_dets_rrc, LABEL, LABEL_VERSE, \
                        get_rotate_mat, align_pos, align_points, get_frame_det_info, get_transform_mat

import torch
import torch.nn


import numpy as np
import io
from PIL import Image
import pickle
import csv
import random

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from functools import partial

# For Point Cloud
from point_cloud.preprocess import read_and_prep_points


from .common import *

from utils.data_util import generate_seq_dets, generate_seq_gts, generate_seq_dets_rrc, LABEL, LABEL_VERSE, \
                        get_rotate_mat, align_pos, align_points, get_frame_det_info, get_transform_mat








class Inputprocess:

    def __init__(self,transform==None):

        self.root_dir = /data0/HR_dataset/KITTI_tracking/kitti/training
        self.oxts_seq = {}
        self.calib = {}

        if transform == None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform


        for i in range(21):
            seq_id = f'{i:04d}'  # 补齐数，前面为零
            with open(f"{self.root_dir}/oxts/{seq_id}.txt") as f_oxts:
                self.oxts_seq[seq_id] = f_oxts.readlines()  # 汽车物理位置数据   seq_id:0000
                self.calib[seq_id] = read_calib_file(f"{root_dir}/calib/{seq_id}.txt")  # 标定校准数据

        self.get_pointcloud = partial(read_and_prep_points, root_path=self.root_dir,
                                      use_frustum=use_frustum, without_reflectivity=without_reflectivity,
                                      num_point_features=num_point_features, det_type=self.det_type)

    def generate_img_lidar(self, input1, input2):

        inputs = [input1, input2]
        frames = []
        for input in inputs:
            frame = []
            # 拿出一个图片的pos,rad数据，注意，也是从kitti数据集拿出
            img_seq_id = input["img_seq_id"]
            img_frame_id = input["img_frame_id"]
            pos, rad = get_pos(self.oxts_seq[img_seq_id], int(img_frame_id))    # img_seq_id：0020， img_frame_id: 000000
            # 将一个图片的calib,pos,rad信息，打包成一个字典
            frame_info = get_frame_info(img_seq_id, img_frame_id,
                                        calib[img_seq_id], pos, rad)

            detection = {}
            detection["dimensions"] = np.array(input['detection_3D_fusion']['dets_3d_fusion'])[:, 0:3]
            detection["location"] = np.array(input['detection_3D_fusion']['dets_3d_fusion'])[:, 3:6]
            detection["rotation_y"] = np.array(input['detection_3D_fusion']['dets_3d_fusion'])[:, -1]
            detection["bbox"] = np.array(input['detection_3D_fusion']['dets_3d_fusion_info'])[:, 2:6]
            detection["image_idx"] = input["img_seq_id"] + "-" + input["img_frame_id"]





            # frames = self.metas[idx]  # 一个检测对
            # 数据预处理
            frame = {"detection":detection,
                      "frame_info":frame_info,
                      "point_path":input['point_path']}

            # frame = {"detection":{"bbox":None, "location":None, "rotation_y":None},
            #           "frame_info":{"image_shape":None, "pos":None, "rad":None, "calib/Tr_velo_to_cam":None,
            #                         "calib/Tr_imu_to_velo":None, "calib/R0_rect":None},
            #           "point_path":None}

            frames.append(frame)



        det_imgs = []  #
        det_split = []
        dets = []
        det_info = get_frame_det_info()  # 其实就是一个字典
        R = []
        T = []
        pos = []
        rad = []
        delta_rad = []
        first_flag = 0
        for frame in frames:  # 遍历一个检测对里的每一个帧，这里就是遍历两帧
            path = f"{self.root_dir}/image_02/{frame['image_path']}"
            img = Image.open(path)
            det_num = len(frame['detection_3D_fusion']['dets_3d_fusion'])  # 每一帧中，检测出的目标数量
            frame['frame_info']['img_shape'] = np.array([img.size[1], img.size[0]])  # w, h -> h, w

            # 这里point_cloud是一个字典，包含一帧的所有点云和点云split
            point_cloud = self.get_pointcloud(info=frame['frame_info'], point_path=frame['point_path'],
                                              dets=frame['detection'], shift_bbox=frame['detection']['bbox'])
            pos.append(frame['frame_info']['pos'])
            rad.append(frame['frame_info']['rad'])

            # Align the bbox to the same coordinate
            if len(rad) >= 2:
                delta_rad.append(rad[-1] - rad[-2])
                R.append(get_rotate_mat(delta_rad[-1], rotate_order=[1, 2, 3]))
                T.append(get_transform_mat(pos[-1] - pos[-2], rad[-2][-1]))
            location, rotation_y = align_pos(R, T, frame['frame_info']['calib/Tr_velo_to_cam'],
                                             frame['frame_info']['calib/Tr_imu_to_velo'],
                                             frame['frame_info']['calib/R0_rect'], delta_rad,
                                             frame['detection']['location'],  # 这两个我们给的数据集是有的
                                             frame['detection']['rotation_y'])
            point_cloud['points'][:, :3] = align_points(R, T, frame['frame_info']['calib/Tr_imu_to_velo'],
                                                        # 从这可以看出，只用了点云的坐标，没用反射强度
                                                        point_cloud['points'][:, :3])

            for i in range(det_num):
                x1 = np.floor(frame['detection']['bbox'][i][0])
                y1 = np.floor(frame['detection']['bbox'][i][1])
                x2 = np.ceil(frame['detection']['bbox'][i][2])
                y2 = np.ceil(frame['detection']['bbox'][i][3])
                det_imgs.append(
                    self.transform(img.crop((x1, y1, x2, y2)).resize((224, 224), Image.BILINEAR)).unsqueeze(0))

            if 'image_idx' in frame['detection'].keys():
                frame['detection'].pop('image_idx')
            dets.append(frame['detection'])
            det_split.append(det_num)
            det_info['loc'].append(torch.Tensor(location))
            det_info['rot'].append(torch.Tensor(rotation_y))
            det_info['dim'].append(torch.Tensor(frame['detection']['dimensions']))
            det_info['points'].append(torch.Tensor(point_cloud['points']))
            det_info['points_split'].append(torch.Tensor(point_cloud['points_split'])[first_flag:])
            det_info['info_id'].append(frame['frame_info']['info_id'])
            if first_flag == 0:
                first_flag += 1

        det_imgs = torch.cat(det_imgs, dim=0)
        det_info['loc'] = torch.cat(det_info['loc'], dim=0)
        det_info['rot'] = torch.cat(det_info['rot'], dim=0)
        det_info['dim'] = torch.cat(det_info['dim'], dim=0)
        det_info['points'] = torch.cat(det_info['points'], dim=0)

        # Shift the point split idx
        start = 0
        for i in range(len(det_info['points_split'])):
            det_info['points_split'][i] += start
            start = det_info['points_split'][i][-1]
        det_info['points_split'] = torch.cat(det_info['points_split'], dim=0)

        # det_split :[第一帧的目标数量, 第二帧的目标数量]
        # dets:  一个检测对中的两帧的各自的检测数据，包括2维和3维的
        # det_info: 一个检测对中的点云数据，还有loc,rot,dim,，还有points_split,points.....注意这的loc,rot,dim，dets也有这些，但是这里的是利用dets里的对齐，处理过后得到的，所以感觉后面用的话，也是用这里的。
        # det_imgs: 一个检测对的图像数据
        return det_imgs, det_info, dets, det_split



# if __name__ == '__main__':
#     getdata = Inputprocess()
#
#     with open(r'2d_3d_dets\0006.pkl', "rb") as fp:
#         loaded_data1 = pickle.load(fp)
#         print(len(loaded_data1))
#
#         det_imgs, det_info, dets, det_split = getdata.generate_img_lidar(loaded_data1[str(1), str(2)])








# def get_frame(img_seq_id, img_frame_id, dets, frame_info):
#     id_path = f'{img_seq_id}-{img_frame_id}'
#     return {
#         'seq_id': img_seq_id,
#         'frame_id': img_frame_id,
#         'image_id': id_path,
#         'point_path': f'{id_path}.bin',
#         'image_path': f'{img_seq_id}/{img_frame_id}.png',
#         'frame_info': frame_info,
#         'detection': dets,
#     }

