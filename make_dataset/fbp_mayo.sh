#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=fbp_mayo
#SBATCH --nodelist=gpu-node008
#SBATCH --cpus-per-task=16
#SBATCH --time=0


echo "1/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L067_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20201221_0.log &
echo "2/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L067_full_sino" --dose_rate "2" --device "1" &>> fbp_mayo_20201221_1.log &
echo "3/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_3" --name "L067_full_sino" --dose_rate "3" --device "2" &>> fbp_mayo_20201221_2.log &
echo "4/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L067_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "4/160"
echo "5/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_5" --name "L067_full_sino" --dose_rate "5" --device "0" &>> fbp_mayo_20201221_0.log &
echo "6/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L067_full_sino" --dose_rate "6" --device "1" &>> fbp_mayo_20201221_1.log &
echo "7/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_7" --name "L067_full_sino" --dose_rate "7" --device "2" &>> fbp_mayo_20201221_2.log &
echo "8/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L067_full_sino" --dose_rate "8" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "8/160"
echo "9/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_9" --name "L067_full_sino" --dose_rate "9" --device "0" &>> fbp_mayo_20201221_0.log &
echo "10/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_10" --name "L067_full_sino" --dose_rate "10" --device "1" &>> fbp_mayo_20201221_1.log &
echo "11/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_11" --name "L067_full_sino" --dose_rate "11" --device "2" &>> fbp_mayo_20201221_2.log &
echo "12/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L067_full_sino" --dose_rate "12" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "12/160"
echo "13/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_13" --name "L067_full_sino" --dose_rate "13" --device "0" &>> fbp_mayo_20201221_0.log &
echo "14/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_14" --name "L067_full_sino" --dose_rate "14" --device "1" &>> fbp_mayo_20201221_1.log &
echo "15/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_15" --name "L067_full_sino" --dose_rate "15" --device "2" &>> fbp_mayo_20201221_2.log &
echo "16/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L067_full_sino" --dose_rate "16" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "16/160"
echo "17/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L096_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20201221_0.log &
echo "18/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L096_full_sino" --dose_rate "2" --device "1" &>> fbp_mayo_20201221_1.log &
echo "19/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_3" --name "L096_full_sino" --dose_rate "3" --device "2" &>> fbp_mayo_20201221_2.log &
echo "20/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L096_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "20/160"
echo "21/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_5" --name "L096_full_sino" --dose_rate "5" --device "0" &>> fbp_mayo_20201221_0.log &
echo "22/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L096_full_sino" --dose_rate "6" --device "1" &>> fbp_mayo_20201221_1.log &
echo "23/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_7" --name "L096_full_sino" --dose_rate "7" --device "2" &>> fbp_mayo_20201221_2.log &
echo "24/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L096_full_sino" --dose_rate "8" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "24/160"
echo "25/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_9" --name "L096_full_sino" --dose_rate "9" --device "0" &>> fbp_mayo_20201221_0.log &
echo "26/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_10" --name "L096_full_sino" --dose_rate "10" --device "1" &>> fbp_mayo_20201221_1.log &
echo "27/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_11" --name "L096_full_sino" --dose_rate "11" --device "2" &>> fbp_mayo_20201221_2.log &
echo "28/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L096_full_sino" --dose_rate "12" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "28/160"
echo "29/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_13" --name "L096_full_sino" --dose_rate "13" --device "0" &>> fbp_mayo_20201221_0.log &
echo "30/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_14" --name "L096_full_sino" --dose_rate "14" --device "1" &>> fbp_mayo_20201221_1.log &
echo "31/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_15" --name "L096_full_sino" --dose_rate "15" --device "2" &>> fbp_mayo_20201221_2.log &
echo "32/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L096_full_sino" --dose_rate "16" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "32/160"
echo "33/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L109_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20201221_0.log &
echo "34/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L109_full_sino" --dose_rate "2" --device "1" &>> fbp_mayo_20201221_1.log &
echo "35/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_3" --name "L109_full_sino" --dose_rate "3" --device "2" &>> fbp_mayo_20201221_2.log &
echo "36/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L109_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "36/160"
echo "37/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_5" --name "L109_full_sino" --dose_rate "5" --device "0" &>> fbp_mayo_20201221_0.log &
echo "38/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L109_full_sino" --dose_rate "6" --device "1" &>> fbp_mayo_20201221_1.log &
echo "39/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_7" --name "L109_full_sino" --dose_rate "7" --device "2" &>> fbp_mayo_20201221_2.log &
echo "40/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L109_full_sino" --dose_rate "8" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "40/160"
echo "41/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_9" --name "L109_full_sino" --dose_rate "9" --device "0" &>> fbp_mayo_20201221_0.log &
echo "42/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_10" --name "L109_full_sino" --dose_rate "10" --device "1" &>> fbp_mayo_20201221_1.log &
echo "43/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_11" --name "L109_full_sino" --dose_rate "11" --device "2" &>> fbp_mayo_20201221_2.log &
echo "44/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L109_full_sino" --dose_rate "12" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "44/160"
echo "45/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_13" --name "L109_full_sino" --dose_rate "13" --device "0" &>> fbp_mayo_20201221_0.log &
echo "46/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_14" --name "L109_full_sino" --dose_rate "14" --device "1" &>> fbp_mayo_20201221_1.log &
echo "47/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_15" --name "L109_full_sino" --dose_rate "15" --device "2" &>> fbp_mayo_20201221_2.log &
echo "48/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L109_full_sino" --dose_rate "16" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "48/160"
echo "49/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L143_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20201221_0.log &
echo "50/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L143_full_sino" --dose_rate "2" --device "1" &>> fbp_mayo_20201221_1.log &
echo "51/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_3" --name "L143_full_sino" --dose_rate "3" --device "2" &>> fbp_mayo_20201221_2.log &
echo "52/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L143_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "52/160"
echo "53/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_5" --name "L143_full_sino" --dose_rate "5" --device "0" &>> fbp_mayo_20201221_0.log &
echo "54/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L143_full_sino" --dose_rate "6" --device "1" &>> fbp_mayo_20201221_1.log &
echo "55/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_7" --name "L143_full_sino" --dose_rate "7" --device "2" &>> fbp_mayo_20201221_2.log &
echo "56/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L143_full_sino" --dose_rate "8" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "56/160"
echo "57/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_9" --name "L143_full_sino" --dose_rate "9" --device "0" &>> fbp_mayo_20201221_0.log &
echo "58/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_10" --name "L143_full_sino" --dose_rate "10" --device "1" &>> fbp_mayo_20201221_1.log &
echo "59/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_11" --name "L143_full_sino" --dose_rate "11" --device "2" &>> fbp_mayo_20201221_2.log &
echo "60/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L143_full_sino" --dose_rate "12" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "60/160"
echo "61/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_13" --name "L143_full_sino" --dose_rate "13" --device "0" &>> fbp_mayo_20201221_0.log &
echo "62/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_14" --name "L143_full_sino" --dose_rate "14" --device "1" &>> fbp_mayo_20201221_1.log &
echo "63/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_15" --name "L143_full_sino" --dose_rate "15" --device "2" &>> fbp_mayo_20201221_2.log &
echo "64/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L143_full_sino" --dose_rate "16" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "64/160"
echo "65/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L192_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20201221_0.log &
echo "66/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L192_full_sino" --dose_rate "2" --device "1" &>> fbp_mayo_20201221_1.log &
echo "67/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_3" --name "L192_full_sino" --dose_rate "3" --device "2" &>> fbp_mayo_20201221_2.log &
echo "68/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L192_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "68/160"
echo "69/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_5" --name "L192_full_sino" --dose_rate "5" --device "0" &>> fbp_mayo_20201221_0.log &
echo "70/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L192_full_sino" --dose_rate "6" --device "1" &>> fbp_mayo_20201221_1.log &
echo "71/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_7" --name "L192_full_sino" --dose_rate "7" --device "2" &>> fbp_mayo_20201221_2.log &
echo "72/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L192_full_sino" --dose_rate "8" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "72/160"
echo "73/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_9" --name "L192_full_sino" --dose_rate "9" --device "0" &>> fbp_mayo_20201221_0.log &
echo "74/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_10" --name "L192_full_sino" --dose_rate "10" --device "1" &>> fbp_mayo_20201221_1.log &
echo "75/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_11" --name "L192_full_sino" --dose_rate "11" --device "2" &>> fbp_mayo_20201221_2.log &
echo "76/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L192_full_sino" --dose_rate "12" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "76/160"
echo "77/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_13" --name "L192_full_sino" --dose_rate "13" --device "0" &>> fbp_mayo_20201221_0.log &
echo "78/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_14" --name "L192_full_sino" --dose_rate "14" --device "1" &>> fbp_mayo_20201221_1.log &
echo "79/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_15" --name "L192_full_sino" --dose_rate "15" --device "2" &>> fbp_mayo_20201221_2.log &
echo "80/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L192_full_sino" --dose_rate "16" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "80/160"
echo "81/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L286_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20201221_0.log &
echo "82/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L286_full_sino" --dose_rate "2" --device "1" &>> fbp_mayo_20201221_1.log &
echo "83/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_3" --name "L286_full_sino" --dose_rate "3" --device "2" &>> fbp_mayo_20201221_2.log &
echo "84/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L286_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "84/160"
echo "85/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_5" --name "L286_full_sino" --dose_rate "5" --device "0" &>> fbp_mayo_20201221_0.log &
echo "86/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L286_full_sino" --dose_rate "6" --device "1" &>> fbp_mayo_20201221_1.log &
echo "87/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_7" --name "L286_full_sino" --dose_rate "7" --device "2" &>> fbp_mayo_20201221_2.log &
echo "88/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L286_full_sino" --dose_rate "8" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "88/160"
echo "89/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_9" --name "L286_full_sino" --dose_rate "9" --device "0" &>> fbp_mayo_20201221_0.log &
echo "90/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_10" --name "L286_full_sino" --dose_rate "10" --device "1" &>> fbp_mayo_20201221_1.log &
echo "91/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_11" --name "L286_full_sino" --dose_rate "11" --device "2" &>> fbp_mayo_20201221_2.log &
echo "92/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L286_full_sino" --dose_rate "12" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "92/160"
echo "93/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_13" --name "L286_full_sino" --dose_rate "13" --device "0" &>> fbp_mayo_20201221_0.log &
echo "94/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_14" --name "L286_full_sino" --dose_rate "14" --device "1" &>> fbp_mayo_20201221_1.log &
echo "95/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_15" --name "L286_full_sino" --dose_rate "15" --device "2" &>> fbp_mayo_20201221_2.log &
echo "96/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L286_full_sino" --dose_rate "16" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "96/160"
echo "97/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L291_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20201221_0.log &
echo "98/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L291_full_sino" --dose_rate "2" --device "1" &>> fbp_mayo_20201221_1.log &
echo "99/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_3" --name "L291_full_sino" --dose_rate "3" --device "2" &>> fbp_mayo_20201221_2.log &
echo "100/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L291_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "100/160"
echo "101/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_5" --name "L291_full_sino" --dose_rate "5" --device "0" &>> fbp_mayo_20201221_0.log &
echo "102/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L291_full_sino" --dose_rate "6" --device "1" &>> fbp_mayo_20201221_1.log &
echo "103/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_7" --name "L291_full_sino" --dose_rate "7" --device "2" &>> fbp_mayo_20201221_2.log &
echo "104/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L291_full_sino" --dose_rate "8" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "104/160"
echo "105/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_9" --name "L291_full_sino" --dose_rate "9" --device "0" &>> fbp_mayo_20201221_0.log &
echo "106/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_10" --name "L291_full_sino" --dose_rate "10" --device "1" &>> fbp_mayo_20201221_1.log &
echo "107/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_11" --name "L291_full_sino" --dose_rate "11" --device "2" &>> fbp_mayo_20201221_2.log &
echo "108/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L291_full_sino" --dose_rate "12" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "108/160"
echo "109/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_13" --name "L291_full_sino" --dose_rate "13" --device "0" &>> fbp_mayo_20201221_0.log &
echo "110/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_14" --name "L291_full_sino" --dose_rate "14" --device "1" &>> fbp_mayo_20201221_1.log &
echo "111/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_15" --name "L291_full_sino" --dose_rate "15" --device "2" &>> fbp_mayo_20201221_2.log &
echo "112/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L291_full_sino" --dose_rate "16" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "112/160"
echo "113/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L310_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20201221_0.log &
echo "114/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L310_full_sino" --dose_rate "2" --device "1" &>> fbp_mayo_20201221_1.log &
echo "115/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_3" --name "L310_full_sino" --dose_rate "3" --device "2" &>> fbp_mayo_20201221_2.log &
echo "116/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L310_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "116/160"
echo "117/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_5" --name "L310_full_sino" --dose_rate "5" --device "0" &>> fbp_mayo_20201221_0.log &
echo "118/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L310_full_sino" --dose_rate "6" --device "1" &>> fbp_mayo_20201221_1.log &
echo "119/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_7" --name "L310_full_sino" --dose_rate "7" --device "2" &>> fbp_mayo_20201221_2.log &
echo "120/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L310_full_sino" --dose_rate "8" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "120/160"
echo "121/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_9" --name "L310_full_sino" --dose_rate "9" --device "0" &>> fbp_mayo_20201221_0.log &
echo "122/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_10" --name "L310_full_sino" --dose_rate "10" --device "1" &>> fbp_mayo_20201221_1.log &
echo "123/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_11" --name "L310_full_sino" --dose_rate "11" --device "2" &>> fbp_mayo_20201221_2.log &
echo "124/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L310_full_sino" --dose_rate "12" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "124/160"
echo "125/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_13" --name "L310_full_sino" --dose_rate "13" --device "0" &>> fbp_mayo_20201221_0.log &
echo "126/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_14" --name "L310_full_sino" --dose_rate "14" --device "1" &>> fbp_mayo_20201221_1.log &
echo "127/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_15" --name "L310_full_sino" --dose_rate "15" --device "2" &>> fbp_mayo_20201221_2.log &
echo "128/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L310_full_sino" --dose_rate "16" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "128/160"
echo "129/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L333_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20201221_0.log &
echo "130/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L333_full_sino" --dose_rate "2" --device "1" &>> fbp_mayo_20201221_1.log &
echo "131/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_3" --name "L333_full_sino" --dose_rate "3" --device "2" &>> fbp_mayo_20201221_2.log &
echo "132/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L333_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "132/160"
echo "133/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_5" --name "L333_full_sino" --dose_rate "5" --device "0" &>> fbp_mayo_20201221_0.log &
echo "134/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L333_full_sino" --dose_rate "6" --device "1" &>> fbp_mayo_20201221_1.log &
echo "135/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_7" --name "L333_full_sino" --dose_rate "7" --device "2" &>> fbp_mayo_20201221_2.log &
echo "136/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L333_full_sino" --dose_rate "8" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "136/160"
echo "137/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_9" --name "L333_full_sino" --dose_rate "9" --device "0" &>> fbp_mayo_20201221_0.log &
echo "138/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_10" --name "L333_full_sino" --dose_rate "10" --device "1" &>> fbp_mayo_20201221_1.log &
echo "139/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_11" --name "L333_full_sino" --dose_rate "11" --device "2" &>> fbp_mayo_20201221_2.log &
echo "140/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L333_full_sino" --dose_rate "12" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "140/160"
echo "141/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_13" --name "L333_full_sino" --dose_rate "13" --device "0" &>> fbp_mayo_20201221_0.log &
echo "142/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_14" --name "L333_full_sino" --dose_rate "14" --device "1" &>> fbp_mayo_20201221_1.log &
echo "143/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_15" --name "L333_full_sino" --dose_rate "15" --device "2" &>> fbp_mayo_20201221_2.log &
echo "144/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L333_full_sino" --dose_rate "16" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "144/160"
echo "145/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L506_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20201221_0.log &
echo "146/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L506_full_sino" --dose_rate "2" --device "1" &>> fbp_mayo_20201221_1.log &
echo "147/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_3" --name "L506_full_sino" --dose_rate "3" --device "2" &>> fbp_mayo_20201221_2.log &
echo "148/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L506_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "148/160"
echo "149/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_5" --name "L506_full_sino" --dose_rate "5" --device "0" &>> fbp_mayo_20201221_0.log &
echo "150/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L506_full_sino" --dose_rate "6" --device "1" &>> fbp_mayo_20201221_1.log &
echo "151/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_7" --name "L506_full_sino" --dose_rate "7" --device "2" &>> fbp_mayo_20201221_2.log &
echo "152/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L506_full_sino" --dose_rate "8" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "152/160"
echo "153/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_9" --name "L506_full_sino" --dose_rate "9" --device "0" &>> fbp_mayo_20201221_0.log &
echo "154/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_10" --name "L506_full_sino" --dose_rate "10" --device "1" &>> fbp_mayo_20201221_1.log &
echo "155/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_11" --name "L506_full_sino" --dose_rate "11" --device "2" &>> fbp_mayo_20201221_2.log &
echo "156/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L506_full_sino" --dose_rate "12" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "156/160"
echo "157/160" &>> fbp_mayo_20201221_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_13" --name "L506_full_sino" --dose_rate "13" --device "0" &>> fbp_mayo_20201221_0.log &
echo "158/160" &>> fbp_mayo_20201221_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_14" --name "L506_full_sino" --dose_rate "14" --device "1" &>> fbp_mayo_20201221_1.log &
echo "159/160" &>> fbp_mayo_20201221_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_15" --name "L506_full_sino" --dose_rate "15" --device "2" &>> fbp_mayo_20201221_2.log &
echo "160/160" &>> fbp_mayo_20201221_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "2e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L506_full_sino" --dose_rate "16" --device "3" &>> fbp_mayo_20201221_3.log &
wait
echo "160/160"
wait
cat fbp_mayo_20201221_0.log fbp_mayo_20201221_1.log fbp_mayo_20201221_2.log fbp_mayo_20201221_3.log > fbp_mayo_20201221.log
rm fbp_mayo_20201221_0.log
rm fbp_mayo_20201221_1.log
rm fbp_mayo_20201221_2.log
rm fbp_mayo_20201221_3.log
