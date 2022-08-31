# -*- coding:utf-8 -*-
"""
作者:wensong
日期:2022年08月26日
"""


from modules import TrackingNet
import argparse
import numpy as np

import pickle
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import yaml
from easydict import EasyDict

from utils.train_util import load_state

from test6 import Inputprocess


parser = argparse.ArgumentParser(description='MOT Input_Features Evaluation')
parser.add_argument('--config', default='/data0/HR_dataset/2023AAAI/2_liu/input_features/config.yaml')
parser.add_argument('--load-path', default='/data0/HR_dataset/2023AAAI/2_liu/input_features/pp_pv_40e_mul_C-gpu.pth', type=str)



def main():
    global args, config
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = EasyDict(config['common'])




    def build_model(config):
        model = TrackingNet(
            seq_len=config.sample_max_len,
            score_arch=config.model.score_arch,
            appear_arch=config.model.appear_arch,
            appear_len=config.model.appear_len,
            appear_skippool=config.model.appear_skippool,
            appear_fpn=config.model.appear_fpn,
            point_arch=config.model.point_arch,
            point_len=config.model.point_len,
            without_reflectivity=config.without_reflectivity,
            softmax_mode=config.model.softmax_mode,
            affinity_op=config.model.affinity_op,
            end_arch=config.model.end_arch,
            end_mode=config.model.end_mode,
            test_mode=config.model.test_mode,
            score_fusion_arch=config.model.score_fusion_arch,
            neg_threshold=config.model.neg_threshold,
            dropblock=config.dropblock,
            use_dropout=config.use_dropout,
        )
        return model






    # create model
    model = build_model(config)
    model.cuda()

    # optionally resume from a checkpoint
    load_state(args.load_path, model)

    cudnn.benchmark = True

    return model



if __name__ == '__main__':
    getdata = Inputprocess()
    model = main()

    with open('/data0/HR_dataset/2023AAAI/2_liu/input_features/2d_3d_dets/0006.pkl', "rb") as fp:      #加载pkl文件
        loaded_data1 = pickle.load(fp)

        det_imgs, det_info, dets, det_split = getdata.generate_img_lidar(loaded_data1[str(50)], loaded_data1[str(51)])       # 这边选的是第50帧和第51帧，
        #det_split 对应关联两帧的检测数量


        det_imgs = det_imgs.to("cuda")
        # det_info = det_info.to("cuda")
        del det_info['info_id']
        det_info={key:(det_info[key]).to("cuda") for key in det_info}
        det_split = torch.tensor(det_split).to("cuda")

        det_info['points'] = det_info['points'].unsqueeze(0)
        det_info['points_split'] = det_info['points_split'].unsqueeze(0)





        link_mat = model(det_imgs, det_info, det_split)     # （3, 1， N,  M) 3d-2d，3d, 2d

        print(link_mat)   # 图像，三维，融合 对应的关联矩阵
        print(link_mat.shape)
