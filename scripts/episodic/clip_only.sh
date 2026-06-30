#!/bin/bash
# Episodic (--reset 1): clip-only (per-sample gradient clipping, noise pinned to 0).

python main.py --algorithm tent_clip --arch vit --batch_size 64 --reset 1 \
  --max_norm 1 --noise 0 --lr 0.1 --output /output_dir/tent_clip_reset1_lr0.1_c1

python main.py --algorithm eata_clip --arch vit --batch_size 64 --reset 1 \
  --max_norm 30 --noise 0 --lr 0.005 --output /output_dir/eata_clip_reset1_lr0.005_c30

python main.py --algorithm sar_clip --arch vit --batch_size 64 --reset 1 \
  --max_norm 5 --noise 0 --lr 0.05 --output /output_dir/sar_clip_reset1_lr0.05_c5

python main.py --algorithm deyo_clip --arch vit --batch_size 64 --reset 1 \
  --max_norm 35 --noise 0 --lr 0.005 --output /output_dir/deyo_clip_reset1_lr0.005_c35

python main.py --algorithm deyo_come_clip --arch vit --batch_size 64 --reset 1 \
  --max_norm 20 --noise 0 --lr 0.01 --output /output_dir/deyo_come_clip_reset1_lr0.01_c20
