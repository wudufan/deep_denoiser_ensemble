#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=train2d_wgan
#SBATCH --nodelist=gpu-node008
#SBATCH --cpus-per-task=16
#SBATCH --time=0


cd ..
python3 train2d_wgan.py --config "./config/wgan/l2_depth_3.cfg" --IO.source "dose_rate_6" --IO.tag "l2_depth_3_wgan/dose_rate_6" --Training.device "1" &> ./output/wgan/l2_depth_3_dose_rate_6.log &
wait
