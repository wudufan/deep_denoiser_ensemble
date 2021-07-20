#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=fbp_mayo
#SBATCH --nodelist=gpu-node008
#SBATCH --cpus-per-task=16
#SBATCH --time=0


echo "1/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L067_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20210707_0.log &
echo "2/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L067_full_sino" --dose_rate "2" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "2/70"
echo "3/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L067_full_sino" --dose_rate "4" --device "0" &>> fbp_mayo_20210707_0.log &
echo "4/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L067_full_sino" --dose_rate "6" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "4/70"
echo "5/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L067_full_sino" --dose_rate "8" --device "0" &>> fbp_mayo_20210707_0.log &
echo "6/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L067_full_sino" --dose_rate "12" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "6/70"
echo "7/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L067_full_sino" --dose_rate "16" --device "0" &>> fbp_mayo_20210707_0.log &
echo "8/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L096_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "8/70"
echo "9/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L096_full_sino" --dose_rate "2" --device "0" &>> fbp_mayo_20210707_0.log &
echo "10/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L096_full_sino" --dose_rate "4" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "10/70"
echo "11/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L096_full_sino" --dose_rate "6" --device "0" &>> fbp_mayo_20210707_0.log &
echo "12/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L096_full_sino" --dose_rate "8" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "12/70"
echo "13/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L096_full_sino" --dose_rate "12" --device "0" &>> fbp_mayo_20210707_0.log &
echo "14/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L096_full_sino" --dose_rate "16" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "14/70"
echo "15/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L109_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20210707_0.log &
echo "16/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L109_full_sino" --dose_rate "2" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "16/70"
echo "17/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L109_full_sino" --dose_rate "4" --device "0" &>> fbp_mayo_20210707_0.log &
echo "18/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L109_full_sino" --dose_rate "6" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "18/70"
echo "19/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L109_full_sino" --dose_rate "8" --device "0" &>> fbp_mayo_20210707_0.log &
echo "20/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L109_full_sino" --dose_rate "12" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "20/70"
echo "21/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L109_full_sino" --dose_rate "16" --device "0" &>> fbp_mayo_20210707_0.log &
echo "22/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L143_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "22/70"
echo "23/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L143_full_sino" --dose_rate "2" --device "0" &>> fbp_mayo_20210707_0.log &
echo "24/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L143_full_sino" --dose_rate "4" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "24/70"
echo "25/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L143_full_sino" --dose_rate "6" --device "0" &>> fbp_mayo_20210707_0.log &
echo "26/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L143_full_sino" --dose_rate "8" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "26/70"
echo "27/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L143_full_sino" --dose_rate "12" --device "0" &>> fbp_mayo_20210707_0.log &
echo "28/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L143_full_sino" --dose_rate "16" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "28/70"
echo "29/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L192_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20210707_0.log &
echo "30/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L192_full_sino" --dose_rate "2" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "30/70"
echo "31/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L192_full_sino" --dose_rate "4" --device "0" &>> fbp_mayo_20210707_0.log &
echo "32/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L192_full_sino" --dose_rate "6" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "32/70"
echo "33/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L192_full_sino" --dose_rate "8" --device "0" &>> fbp_mayo_20210707_0.log &
echo "34/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L192_full_sino" --dose_rate "12" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "34/70"
echo "35/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L192_full_sino" --dose_rate "16" --device "0" &>> fbp_mayo_20210707_0.log &
echo "36/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L286_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "36/70"
echo "37/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L286_full_sino" --dose_rate "2" --device "0" &>> fbp_mayo_20210707_0.log &
echo "38/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L286_full_sino" --dose_rate "4" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "38/70"
echo "39/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L286_full_sino" --dose_rate "6" --device "0" &>> fbp_mayo_20210707_0.log &
echo "40/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L286_full_sino" --dose_rate "8" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "40/70"
echo "41/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L286_full_sino" --dose_rate "12" --device "0" &>> fbp_mayo_20210707_0.log &
echo "42/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L286_full_sino" --dose_rate "16" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "42/70"
echo "43/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L291_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20210707_0.log &
echo "44/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L291_full_sino" --dose_rate "2" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "44/70"
echo "45/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L291_full_sino" --dose_rate "4" --device "0" &>> fbp_mayo_20210707_0.log &
echo "46/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L291_full_sino" --dose_rate "6" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "46/70"
echo "47/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L291_full_sino" --dose_rate "8" --device "0" &>> fbp_mayo_20210707_0.log &
echo "48/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L291_full_sino" --dose_rate "12" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "48/70"
echo "49/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L291_full_sino" --dose_rate "16" --device "0" &>> fbp_mayo_20210707_0.log &
echo "50/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L310_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "50/70"
echo "51/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L310_full_sino" --dose_rate "2" --device "0" &>> fbp_mayo_20210707_0.log &
echo "52/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L310_full_sino" --dose_rate "4" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "52/70"
echo "53/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L310_full_sino" --dose_rate "6" --device "0" &>> fbp_mayo_20210707_0.log &
echo "54/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L310_full_sino" --dose_rate "8" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "54/70"
echo "55/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L310_full_sino" --dose_rate "12" --device "0" &>> fbp_mayo_20210707_0.log &
echo "56/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L310_full_sino" --dose_rate "16" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "56/70"
echo "57/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L333_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20210707_0.log &
echo "58/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L333_full_sino" --dose_rate "2" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "58/70"
echo "59/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L333_full_sino" --dose_rate "4" --device "0" &>> fbp_mayo_20210707_0.log &
echo "60/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L333_full_sino" --dose_rate "6" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "60/70"
echo "61/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L333_full_sino" --dose_rate "8" --device "0" &>> fbp_mayo_20210707_0.log &
echo "62/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L333_full_sino" --dose_rate "12" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "62/70"
echo "63/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L333_full_sino" --dose_rate "16" --device "0" &>> fbp_mayo_20210707_0.log &
echo "64/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_1" --name "L506_full_sino" --dose_rate "1" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "64/70"
echo "65/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_2" --name "L506_full_sino" --dose_rate "2" --device "0" &>> fbp_mayo_20210707_0.log &
echo "66/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_4" --name "L506_full_sino" --dose_rate "4" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "66/70"
echo "67/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L506_full_sino" --dose_rate "6" --device "0" &>> fbp_mayo_20210707_0.log &
echo "68/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_8" --name "L506_full_sino" --dose_rate "8" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "68/70"
echo "69/70" &>> fbp_mayo_20210707_0.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_12" --name "L506_full_sino" --dose_rate "12" --device "0" &>> fbp_mayo_20210707_0.log &
echo "70/70" &>> fbp_mayo_20210707_1.log
python3 fbp_mayo.py --input_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/lowdoseCTsets/" --geometry "/home/local/PARTNERS/dw640/deep_denoiser_ensemble/geometry_mayo.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/deep_denoiser_ensemble/data/mayo/dose_rate_16" --name "L506_full_sino" --dose_rate "16" --device "0" &>> fbp_mayo_20210707_1.log &
wait
echo "70/70"
wait
cat fbp_mayo_20210707_0.log fbp_mayo_20210707_1.log > fbp_mayo_20210707.log
rm fbp_mayo_20210707_0.log
rm fbp_mayo_20210707_1.log
