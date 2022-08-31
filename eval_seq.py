# -*- coding:utf-8 -*-
"""
作者:wensong
日期:2022年08月24日
"""

import argparse
import logging
import os
import pprint
import time

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import yaml
from easydict import EasyDict
from kitti_devkit.evaluate_tracking import evaluate
from torch.utils.data import DataLoader
from tracking_model import TrackingModule
from utils.build_util import build_augmentation, build_dataset, build_model
from utils.data_util import write_kitti_result
from utils.train_util import AverageMeter, create_logger, load_state


parser = argparse.ArgumentParser(description='PyTorch mmMOT Evaluation')
parser.add_argument('--config', default='/home/dlab/AI_Group/lWS/mmMOT/experiments/pp_pv_40e_mul_C/config.yaml')
parser.add_argument('--load-path', default='/home/dlab/AI_Group/lWS/mmMOT/experiments/pp_pv_40e_mul_C-gpu.pth', type=str)
parser.add_argument('--result-path', default='/home/dlab/AI_Group/lWS/mmMOT/results', type=str)
parser.add_argument('--recover', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--result_sha', default='eval')
parser.add_argument('--memory', action='store_true')


def main():

    global args, config, best_mota
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = EasyDict(config['common'])
    config.save_path = os.path.dirname(args.config)

    # create model
    model = build_model(config)
    model.cuda()

    # optionally resume from a checkpoint
    load_state(args.load_path, model)

    cudnn.benchmark = True




    tracking_module = TrackingModule(model, None, None, config.det_type)






    aligned_ids, aligned_dets, frame_start = tracking_module.predict(
                    input[0], det_info, dets, det_split)



