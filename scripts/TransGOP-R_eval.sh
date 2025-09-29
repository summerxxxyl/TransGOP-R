#!/bin/bash
python main.py \
    --config_file config/TransGOP-R/TransGOP-R_4scale.py \
    --output_dir logs/TransGOP-R/R50-MS4-eval \
    --coco_path /root/autodl-tmp/TransGOP-R/datasets/gooreal \
    --eval \
    --resume /root/autodl-tmp/TransGOP-R/logs/TransGOP-R/R50-MS4/checkpoint_best_regular.pth \
    --options dn_scalar=100 embed_init_tgt=TRUE \
    dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
    dn_box_noise_scale=1.0