# 单GPU
python tools/train.py configs/solov2/solov2_light_448_r18_fpn_8gpu_3x.py

# 断点续训练
# python tools/train.py configs/solov2/solov2_light_448_r18_fpn_8gpu_3x.py --resume_from work_dirs/solov2_light_448_r18_fpn_8gpu_3x_part2p1_epoch200/epoch_97.pth

# 多GPU
# ./tools/dist_train.sh configs/solov2/solov2_light_448_r18_fpn_8gpu_3x.py  2