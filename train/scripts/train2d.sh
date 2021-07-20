#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=train2d
#SBATCH --nodelist=gpu-node008
#SBATCH --cpus-per-task=16
#SBATCH --time=0


cd ..
python3 train2d.py --config "./config/baseline/l2_depth_3.cfg" --IO.source "dose_rate_2" --IO.tag "l2_depth_3/dose_rate_2" --Training.device "0" &> ./output/baseline/l2_depth_3_dose_rate_2.log &
wait
echo "1/6"
python3 train2d.py --config "./config/baseline/l2_depth_3.cfg" --IO.source "dose_rate_4" --IO.tag "l2_depth_3/dose_rate_4" --Training.device "0" &> ./output/baseline/l2_depth_3_dose_rate_4.log &
wait
echo "2/6"
python3 train2d.py --config "./config/baseline/l2_depth_3.cfg" --IO.source "dose_rate_8" --IO.tag "l2_depth_3/dose_rate_8" --Training.device "0" &> ./output/baseline/l2_depth_3_dose_rate_8.log &
wait
echo "3/6"
python3 train2d.py --config "./config/baseline/l2_depth_3.cfg" --IO.source "dose_rate_6" --IO.tag "l2_depth_3/dose_rate_6" --Training.device "0" &> ./output/baseline/l2_depth_3_dose_rate_6.log &
wait
echo "4/6"
python3 train2d.py --config "./config/baseline/l2_depth_3.cfg" --IO.source "dose_rate_12" --IO.tag "l2_depth_3/dose_rate_12" --Training.device "0" &> ./output/baseline/l2_depth_3_dose_rate_12.log &
wait
echo "5/6"
python3 train2d.py --config "./config/baseline/l2_depth_3.cfg" --IO.source "dose_rate_16" --IO.tag "l2_depth_3/dose_rate_16" --Training.device "0" &> ./output/baseline/l2_depth_3_dose_rate_16.log &
wait
echo "6/6"
wait
