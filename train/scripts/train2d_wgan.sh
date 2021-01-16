#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=train2d_wgan
#SBATCH --nodelist=gpu-node008
#SBATCH --cpus-per-task=16
#SBATCH --time=0


cd ..
python3 train2d_wgan.py --config "./config/wgan/l2_depth_3.cfg" --IO.source "dose_rate_2" --IO.tag "l2_depth_3_wgan/dose_rate_2" --Training.device "1" &> ./output/wgan/l2_depth_3_dose_rate_2.log &
python3 train2d_wgan.py --config "./config/wgan/l2_depth_3.cfg" --IO.source "dose_rate_4" --IO.tag "l2_depth_3_wgan/dose_rate_4" --Training.device "2" &> ./output/wgan/l2_depth_3_dose_rate_4.log &
python3 train2d_wgan.py --config "./config/wgan/l2_depth_3.cfg" --IO.source "dose_rate_8" --IO.tag "l2_depth_3_wgan/dose_rate_8" --Training.device "3" &> ./output/wgan/l2_depth_3_dose_rate_8.log &
wait
echo "3/6"
python3 train2d_wgan.py --config "./config/wgan/l2_depth_3.cfg" --IO.source "dose_rate_12" --IO.tag "l2_depth_3_wgan/dose_rate_12" --Training.device "1" &> ./output/wgan/l2_depth_3_dose_rate_12.log &
python3 train2d_wgan.py --config "./config/wgan/l2_depth_3.cfg" --IO.source "dose_rate_16" --IO.tag "l2_depth_3_wgan/dose_rate_16" --Training.device "2" &> ./output/wgan/l2_depth_3_dose_rate_16.log &
python3 train2d_wgan.py --config "./config/wgan/l2_depth_3_all.cfg" --IO.tag "l2_depth_3_wgan/all" --Training.device "3" &> ./output/wgan/l2_depth_3_all.log &
wait
