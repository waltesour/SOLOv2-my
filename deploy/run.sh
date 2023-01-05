#!/usr/bin/env bash
# run on pytorch
# python inference_demo.py

# run on onnxrt & trt
# SOLOv2 light R34
python onnx_exporter_solov2.py ../configs/solov2/solov2_light_448_r18_fpn_8gpu_3x.py  solov2_light_448_r18_fpn_8gpu_3x.onnx --checkpoint ../work_dirs/testONNX/epoch_200.pth --shape 448 768
# python inference_on_onnxrt_trt_solov2.py



# Decoupled SOLO
#python onnx_exporter_decoupled_solo.py ../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py  weights/DSOLO_R50.onnx --checkpoint ../weights/DECOUPLED_SOLO_R50_3x.pth --shape 800 1216
#python inference_on_onnxrt_trt_decoupled_solo.py