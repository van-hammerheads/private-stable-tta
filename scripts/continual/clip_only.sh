#!/bin/bash
# Continual / online (--reset 0): clip-only (per-sample gradient clipping, noise pinned to 0).

python main.py --algorithm tent_clip --arch vit --batch_size 64 --reset 0 \
  --max_norm 0.1 --noise 0 --lr 0.5 --output /output_dir/tent_clip_reset0_lr0.5_c0.1

python main.py --algorithm eata_clip --arch vit --batch_size 64 --reset 0 \
  --max_norm 0.1 --noise 0 --lr 0.5 --output /output_dir/eata_clip_reset0_lr0.5_c0.1

python main.py --algorithm sar_clip --arch vit --batch_size 64 --reset 0 \
  --max_norm 1 --noise 0 --lr 0.1 --output /output_dir/sar_clip_reset0_lr0.1_c1

python main.py --algorithm deyo_clip --arch vit --batch_size 64 --reset 0 \
  --max_norm 10 --noise 0 --lr 0.01 --output /output_dir/deyo_clip_reset0_lr0.01_c10

python main.py --algorithm deyo_come_clip --arch vit --batch_size 64 --reset 0 \
  --max_norm 20 --noise 0 --lr 0.01 --output /output_dir/deyo_come_clip_reset0_lr0.01_c20
