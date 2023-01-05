import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   #os.path.abspath(__file__) 作用： 获取当前脚本的完整路径
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import torch

# Decoupled solo
# config_file = '../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../weights/DECOUPLED_SOLO_R50_3x.pth'

#  Decoupled solo lite
# config_file = '../configs/solo/decoupled_solo_light_r50_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../weights/DECOUPLED_SOLO_LIGHT_R50_3x.pth'

# SOLOv2 lite
config_file = '/workspace/home/SOLO-master/configs/solov2/solov2_light_448_r18_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '/workspace/home/SOLO-master/work_dirs/testONNX/epoch_200.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = '/workspace/home/SOLO-master/expressImg/1657799798081.jpg'
result = inference_detector(model, img)

show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file="/workspace/home/SOLO-master/deploy/testONNX/demo_out_torch2.jpg")
