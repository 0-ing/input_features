# -*- coding:utf-8 -*-
"""
作者:wensong
日期:2022年08月26日
"""

import os

import pickle

seqs_pkl_list = os.listdir("2d_3d_dets_fusion_npz")


for seq_pkl_list in seqs_pkl_list:
    with open("2d_3d_dets_fusion_npz" + "/"+seq_pkl_list, "rb") as fp:
        loaded_data1 = pickle.load(fp)
        for i in range(len(loaded_data1)):

            loaded_data1[str(i)]["image_path"] = seq_pkl_list.split(".")[0] + "/" + f"{i:06d}.png"
            loaded_data1[str(i)]["point_path"] = seq_pkl_list.split(".")[0] + "/" + f"{i:06d}.bin"
            loaded_data1[str(i)]["img_seq_id"] = seq_pkl_list.split(".")[0]
            loaded_data1[str(i)]["img_frame_id"] = f"{i:06d}"

        with open(os.path.join("2d_3d_dets", seq_pkl_list), 'wb') as f:
            pickle.dump(loaded_data1, f)








