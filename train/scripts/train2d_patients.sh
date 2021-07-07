#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=train2d_patients
#SBATCH --nodelist=gpu-node008
#SBATCH --cpus-per-task=16
#SBATCH --time=0


cd ..
python3 train2d.py --config "./config/baseline/l2_depth_3_patient.cfg" --IO.train "L291" --IO.tag "l2_depth_3/L291" --Training.device "1" &> ./output/baseline/l2_depth_3_L291.log &
python3 train2d.py --config "./config/baseline/l2_depth_3_patient.cfg" --IO.train "L067" --IO.tag "l2_depth_3/L067" --Training.device "2" &> ./output/baseline/l2_depth_3_L067.log &
python3 train2d.py --config "./config/baseline/l2_depth_3_patient.cfg" --IO.train "L096" --IO.tag "l2_depth_3/L096" --Training.device "3" &> ./output/baseline/l2_depth_3_L096.log &
wait
echo "3/10"
python3 train2d.py --config "./config/baseline/l2_depth_3_patient.cfg" --IO.train "L109" --IO.tag "l2_depth_3/L109" --Training.device "1" &> ./output/baseline/l2_depth_3_L109.log &
python3 train2d.py --config "./config/baseline/l2_depth_3_patient.cfg" --IO.train "L143" --IO.tag "l2_depth_3/L143" --Training.device "2" &> ./output/baseline/l2_depth_3_L143.log &
python3 train2d.py --config "./config/baseline/l2_depth_3_patient.cfg" --IO.train "L192" --IO.tag "l2_depth_3/L192" --Training.device "3" &> ./output/baseline/l2_depth_3_L192.log &
wait
echo "6/10"
python3 train2d.py --config "./config/baseline/l2_depth_3_patient.cfg" --IO.train "L286" --IO.tag "l2_depth_3/L286" --Training.device "1" &> ./output/baseline/l2_depth_3_L286.log &
python3 train2d.py --config "./config/baseline/l2_depth_3_patient.cfg" --IO.train "L310" --IO.tag "l2_depth_3/L310" --Training.device "2" &> ./output/baseline/l2_depth_3_L310.log &
python3 train2d.py --config "./config/baseline/l2_depth_3_patient.cfg" --IO.train "L333" --IO.tag "l2_depth_3/L333" --Training.device "3" &> ./output/baseline/l2_depth_3_L333.log &
wait
echo "9/10"
python3 train2d.py --config "./config/baseline/l2_depth_3_patient.cfg" --IO.train "L506" --IO.tag "l2_depth_3/L506" --Training.device "1" &> ./output/baseline/l2_depth_3_L506.log &
wait
