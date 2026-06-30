#!/bin/bash
# Continual / online (--reset 0): baselines (no clip, no noise; only reset/lr).

python main.py --algorithm tent --arch vit --batch_size 64 --reset 0 \
  --lr 0.0001 --output /output_dir/tent_reset0_lr0.0001

python main.py --algorithm eata --arch vit --batch_size 64 --reset 0 \
  --lr 0.0005 --output /output_dir/eata_reset0_lr0.0005

python main.py --algorithm sar --arch vit --batch_size 64 --reset 0 \
  --lr 0.005 --output /output_dir/sar_reset0_lr0.005

python main.py --algorithm deyo --arch vit --batch_size 64 --reset 0 \
  --lr 0.0005 --output /output_dir/deyo_reset0_lr0.0005

python main.py --algorithm deyo_come --arch vit --batch_size 64 --reset 0 \
  --lr 0.0005 --output /output_dir/deyocome_reset0_lr0.0005
