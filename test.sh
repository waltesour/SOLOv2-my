# 将train、val、test三者test指定的图片分割出结果
# 注意需要修改test_ins_vis.py 的vis_seg 函数中 class_names 为自己的类别数目
python tools/test_ins_vis.py configs/solov2/solov2_light_448_r18_fpn_8gpu_3x.py  work_dirs/20221227/latest.pth --show --save_dir  work_dirs/20221227_vis
