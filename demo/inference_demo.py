from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import time

config_file = '../configs/solov2/solov2_light_512_dcn_r50_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../checkpoints/SOLOv2_LIGHT_512_DCN_R50_3x.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = 'street.jpg'
start_time = time.time()
result = inference_detector(model, img)
end_time = time.time()
print("耗时: {:.2f}秒".format(end_time - start_time))
show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file="street_out.jpg")
