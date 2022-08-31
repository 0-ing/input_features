# -*- coding:utf-8 -*-
"""
作者:wensong
日期:2022年08月24日
"""


from modules import TrackingNet
import argparse


import torch
import torch.backends.cudnn as cudnn
import torch.optim
import yaml
from easydict import EasyDict

from utils.train_util import load_state


parser = argparse.ArgumentParser(description='MOT Input_Features Evaluation')
parser.add_argument('--config', default=r'D:\pythonProject\input_features\config.yaml')
parser.add_argument('--load-path', default=r'D:\pythonProject\input_features\pp_pv_40e_mul_C-gpu.pth', type=str)



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



    input = torch.randn((12, 3, 224, 224))
    input = input.to('cuda')

    det_info = {}
    det_info['points'] = torch.randn((1, 3545, 3))
    det_info['points'] = det_info['points'].to('cuda')

    det_info['points_split'] = torch.tensor([[0, 230, 500, 899, 1000, 1200, 1500, 1900, 2200, 2500, 2750, 3100, 3545]])
    det_info['points_split'] = det_info['points_split'].to('cuda')

    # input（image）:1x12x3x224x224,
    # det_info[points]:所有点：1x3545x3, 12个目标的所有点云
    # det_info[point_split]:1x13:[0,455,849,...,3545], 两帧中的所有目标的点数，逐次累加。 这里的13.代表是12个目标

    feats, _ = model.feature(input, det_info)

    print(feats)
    print(feats.shape)    # 3x512x12


if __name__ == '__main__':

    main()


# seq_id = f'{i:04d}'
#
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