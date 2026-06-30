#!/bin/bash
# Continual / online (--reset 0): DP methods (clip + noise).

#################################
############ DP-TENT ############
#################################
# ε = 20 (σ=0.6186)
python main.py --algorithm tent_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 0.1 --noise 0.6186 --lr 0.1 --output /output_dir/dp_tent_bs64_c0.1n0.6186_lr0.1

# ε = 15 (σ=0.777)
python main.py --algorithm tent_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 0.1 --noise 0.777 --lr 0.05 --output /output_dir/dp_tent_bs64_c0.1n0.777_lr0.05

# ε = 10 (σ=1.084)
python main.py --algorithm tent_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 0.1 --noise 1.084 --lr 0.05 --output /output_dir/dp_tent_bs64_c0.1n1.084_lr0.05

# ε = 5 (σ=1.966)
python main.py --algorithm tent_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 1 --noise 1.966 --lr 0.001 --output /output_dir/dp_tent_bs64_c1n1.966_lr0.001

# ε = 1 (σ=8.594)
python main.py --algorithm tent_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 0.1 --noise 8.594 --lr 0.005 --output /output_dir/dp_tent_bs64_c0.1n8.594_lr0.005

#################################
############ DP-EATA ############
#################################
# ε = 20 (σ=0.6186)
python main.py --algorithm eata_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 1 --noise 0.6186 --lr 0.01 --output /output_dir/dp_eata_bs64_c1n0.6186_lr0.01

# ε = 15 (σ=0.777)
python main.py --algorithm eata_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 1 --noise 0.777 --lr 0.005 --output /output_dir/dp_eata_bs64_c1n0.777_lr0.005

# ε = 10 (σ=1.084)
python main.py --algorithm eata_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 1 --noise 1.084 --lr 0.005 --output /output_dir/dp_eata_bs64_c1n1.084_lr0.005

# ε = 5 (σ=1.966)
python main.py --algorithm eata_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 10 --noise 1.966 --lr 0.0001 --output /output_dir/dp_eata_bs64_c10n1.966_lr0.0001

# ε = 1 (σ=8.594)
python main.py --algorithm eata_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 0.1 --noise 8.594 --lr 0.005 --output /output_dir/dp_eata_bs64_c0.1n8.594_lr0.005

#################################
############ DP-SAR  ############
#################################
# ε = 20 (σ=0.6186)
python main.py --algorithm sar_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 10 --noise 0.6186 --lr 0.0005 --output /output_dir/dpsat_sar_bs64_c10n0.6186_lr0.0005

# ε = 15 (σ=0.777)
python main.py --algorithm sar_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 10 --noise 0.777 --lr 0.0005 --output /output_dir/dpsat_sar_bs64_c10n0.777_lr0.0005

# ε = 10 (σ=1.084)
python main.py --algorithm sar_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 15 --noise 1.084 --lr 0.0001 --output /output_dir/dpsat_sar_bs64_c15n1.084_lr0.0001

# ε = 5 (σ=1.966)
python main.py --algorithm sar_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 15 --noise 1.966 --lr 0.0001 --output /output_dir/dpsat_sar_bs64_c15n1.966_lr0.0001

# ε = 1 (σ=8.594)
python main.py --algorithm sar_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 5 --noise 8.594 --lr 0.0001 --output /output_dir/dpsat_sar_bs64_c5n8.594_lr0.0001

#################################
############ DP-Deyo ############
#################################
# ε = 20 (σ=0.6186)
python main.py --algorithm deyo_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 10 --noise 0.6186 --lr 5e-05 --output /output_dir/dp_deyo_bs64_c10n0.6186_lr5e-05

# ε = 15 (σ=0.777)
python main.py --algorithm deyo_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 10 --noise 0.777 --lr 5e-05 --output /output_dir/dp_deyo_bs64_c10n0.777_lr5e-05

# ε = 10 (σ=1.084)
python main.py --algorithm deyo_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 1 --noise 1.084 --lr 0.0005 --output /output_dir/dp_deyo_bs64_c1n1.084_lr0.0005

# ε = 5 (σ=1.966)
python main.py --algorithm deyo_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 0.1 --noise 1.966 --lr 0.005 --output /output_dir/dp_deyo_bs64_c0.1n1.966_lr0.005

# ε = 1 (σ=8.594)
python main.py --algorithm deyo_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 0.1 --noise 8.594 --lr 0.005 --output /output_dir/dp_deyo_bs64_c0.1n8.594_lr0.005

#######################################
############ DP-Deyo-Come  ############
#######################################
# ε = 20 (σ=0.6186)
python main.py --algorithm deyo_come_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 10 --noise 0.6186 --lr 0.001 --output /output_dir/dp_deyo_come_bs64_c10n0.6186_lr0.001

# ε = 15 (σ=0.777)
python main.py --algorithm deyo_come_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 5 --noise 0.777 --lr 0.001 --output /output_dir/dp_deyo_come_bs64_c5n0.777_lr0.001

# ε = 10 (σ=1.084)
python main.py --algorithm deyo_come_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 1 --noise 1.084 --lr 0.005 --output /output_dir/dp_deyo_come_bs64_c1n1.084_lr0.005

# ε = 5 (σ=1.966)
python main.py --algorithm deyo_come_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 5 --noise 1.966 --lr 0.001 --output /output_dir/dp_deyo_come_bs64_c5n1.966_lr0.001

# ε = 1 (σ=8.594)
python main.py --algorithm deyo_come_dp --arch vit --batch_size 64 --reset 0 \
  --max_norm 0.1 --noise 8.594 --lr 0.005 --output /output_dir/dp_deyo_come_bs64_c0.1n8.594_lr0.005
