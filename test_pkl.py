# -*- coding:utf-8 -*-
"""
作者:wensong
日期:2022年08月25日
"""


import pickle
import numpy as np

with open(r'2d_3d_dets\0006.pkl', "rb") as fp:
    loaded_data1 = pickle.load(fp)
    print(len(loaded_data1))
    print(loaded_data1['66'])
    detection = {}
    detection["dimensions"] = np.array(loaded_data1['66']['detection_3D_fusion']['dets_3d_fusion'])[:, 0:3]
    detection["location"] = np.array(loaded_data1['66']['detection_3D_fusion']['dets_3d_fusion'])[:, 3:6]
    detection["rotation_y"] = np.array(loaded_data1['66']['detection_3D_fusion']['dets_3d_fusion'])[:, -1]
    detection["bbox"] = np.array(loaded_data1['66']['detection_3D_fusion']['dets_3d_fusion_info'])[:, 2:6]
    detection["image_idx"] = loaded_data1[str(66)]["img_seq_id"] + "-" + loaded_data1[str(66)]["img_frame_id"]
    print(detection)
    a = 1




