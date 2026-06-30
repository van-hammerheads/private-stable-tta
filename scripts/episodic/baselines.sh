#!/bin/bash
# Episodic (--reset 1): baselines (no clip, no noise; only reset/lr).

python main.py --algorithm tent --arch vit --batch_size 64 --reset 1 \
  --lr 0.001 --output /output_dir/tent_reset1_lr0.001

python main.py --algorithm eata --arch vit --batch_size 64 --reset 1 \
  --lr 0.001 --output /output_dir/eata_reset1_lr0.001

python main.py --algorithm sar --arch vit --batch_size 64 --reset 1 \
  --lr 0.005 --output /output_dir/sar_reset1_lr0.005

python main.py --algorithm deyo --arch vit --batch_size 64 --reset 1 \
  --lr 0.001 --output /output_dir/deyo_reset1_lr0.001

python main.py --algorithm deyo_come --arch vit --batch_size 64 --reset 1 \
  --lr 0.001 --output /output_dir/deyocome_reset1_lr0.001
