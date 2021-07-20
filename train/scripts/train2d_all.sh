#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=all2d
#SBATCH --nodelist=gpu-node008
#SBATCH --cpus-per-task=16
#SBATCH --time=0


cd ..
python3 train2d.py --config "./config/baseline/l2_depth_4_all.cfg" --Training.device "0" &> ./output/baseline/l2_depth_4_all.log
python3 train2d.py --config "./config/baseline/l2_depth_3_all.cfg" --Training.device "0" &> ./output/baseline/l2_depth_3_all.log