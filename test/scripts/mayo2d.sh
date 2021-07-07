#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=mayo2d
#SBATCH --nodelist=gpu-node008
#SBATCH --cpus-per-task=16
#SBATCH --time=0


cd ..

echo "24/39"
python3 test2d_mayo.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --train_dir "/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --N0 "1e5" --prj "/home/dwu/data/lowdoseCTsets/L291_full_sino.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/mayo_2d_3_layer_mean/dose_rate_6/l2/L291" --checkpoint "l2_depth_3/dose_rate_6/25.h5" --dose_rate "6" --device "2" &>> ./outputs/mayo2d_0.log &
python3 test2d_mayo.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --train_dir "/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --N0 "1e5" --prj "/home/dwu/data/lowdoseCTsets/L143_full_sino.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/mayo_2d_3_layer_mean/dose_rate_6/l2/L143" --checkpoint "l2_depth_3/dose_rate_6/25.h5" --dose_rate "6" --device "3" &>> ./outputs/mayo2d_1.log &
wait
echo "26/39"
python3 test2d_mayo.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --train_dir "/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --N0 "1e5" --prj "/home/dwu/data/lowdoseCTsets/L067_full_sino.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/mayo_2d_3_layer_mean/dose_rate_6/l2/L067" --checkpoint "l2_depth_3/dose_rate_6/25.h5" --dose_rate "6" --device "2" &>> ./outputs/mayo2d_0.log &

wait

echo "36/39"
python3 test2d_mayo.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --train_dir "/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --N0 "1e5" --prj "/home/dwu/data/lowdoseCTsets/L291_full_sino.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/mayo_2d_3_layer_mean/dose_rate_6/wgan/L291" --checkpoint "l2_depth_3_wgan/dose_rate_6/25.h5" --dose_rate "6" --device "2" &>> ./outputs/mayo2d_0.log &
python3 test2d_mayo.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --train_dir "/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --N0 "1e5" --prj "/home/dwu/data/lowdoseCTsets/L143_full_sino.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/mayo_2d_3_layer_mean/dose_rate_6/wgan/L143" --checkpoint "l2_depth_3_wgan/dose_rate_6/25.h5" --dose_rate "6" --device "3" &>> ./outputs/mayo2d_1.log &
wait
echo "38/39"
python3 test2d_mayo.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --train_dir "/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --N0 "1e5" --prj "/home/dwu/data/lowdoseCTsets/L067_full_sino.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/mayo_2d_3_layer_mean/dose_rate_6/wgan/L067" --checkpoint "l2_depth_3_wgan/dose_rate_6/25.h5" --dose_rate "6" --device "2" &>> ./outputs/mayo2d_0.log &
wait
cat ./outputs/mayo2d_0.log ./outputs/mayo2d_1.log > ./outputs/mayo2d.log
rm ./outputs/mayo2d_0.log
rm ./outputs/mayo2d_1.log
