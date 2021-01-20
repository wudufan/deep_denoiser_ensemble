#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=dect2d
#SBATCH --nodelist=gpu-node008
#SBATCH --cpus-per-task=16
#SBATCH --time=0


cd ..
python3 test2d_ensemble_dect.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/dect_2d_3_layer_mean/geometry.cfg" --train_dir "/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --checkpoint "25" --tags "l2_depth_3/dose_rate_2,l2_depth_3/dose_rate_4,l2_depth_3/dose_rate_8,l2_depth_3/dose_rate_16" --checkpoint_smooth "l2_depth_3/dose_rate_16/25.h5" --N0_ref "1.25e4" --prj "/home/dwu/data/DECT/sinogram/sino_35_2.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/dect_2d_3_layer_mean/half/ensemble/35" --energy "a" --device "2" &>> ./outputs/dect2d_0.log &
python3 test2d_ensemble_dect.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/dect_2d_3_layer_mean/geometry.cfg" --train_dir "/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --checkpoint "25" --tags "l2_depth_3/dose_rate_2,l2_depth_3/dose_rate_4,l2_depth_3/dose_rate_8,l2_depth_3/dose_rate_16" --checkpoint_smooth "l2_depth_3/dose_rate_16/25.h5" --N0_ref "1.25e4" --prj "/home/dwu/data/DECT/sinogram/sino_35_2.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/dect_2d_3_layer_mean/half/ensemble/35" --energy "b" --device "3" &>> ./outputs/dect2d_1.log &
wait
echo "2/14"
python3 test2d_fbp_dect.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/dect_2d_3_layer_mean/geometry.cfg" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --prj "/home/dwu/data/DECT/sinogram/sino_35_1.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/dect_2d_3_layer_mean/full/fbp/35" --device "2" --energy "a" &>> ./outputs/dect2d_0.log &
python3 test2d_fbp_dect.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/dect_2d_3_layer_mean/geometry.cfg" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --prj "/home/dwu/data/DECT/sinogram/sino_35_1.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/dect_2d_3_layer_mean/full/fbp/35" --device "3" --energy "b" &>> ./outputs/dect2d_1.log &
wait
echo "4/14"
python3 test2d_fbp_dect.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/dect_2d_3_layer_mean/geometry.cfg" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --prj "/home/dwu/data/DECT/sinogram/sino_35_2.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/dect_2d_3_layer_mean/half/fbp/35" --device "2" --energy "a" &>> ./outputs/dect2d_0.log &
python3 test2d_fbp_dect.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/dect_2d_3_layer_mean/geometry.cfg" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --prj "/home/dwu/data/DECT/sinogram/sino_35_2.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/dect_2d_3_layer_mean/half/fbp/35" --device "3" --energy "b" &>> ./outputs/dect2d_1.log &
wait
echo "6/14"
python3 test2d_dect.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/dect_2d_3_layer_mean/geometry.cfg" --train_dir "/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --prj "/home/dwu/data/DECT/sinogram/sino_35_2.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/dect_2d_3_layer_mean/half/l2/35_all" --checkpoint "l2_depth_3/all/5.h5" --device "2" --energy "a" &>> ./outputs/dect2d_0.log &
python3 test2d_dect.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/dect_2d_3_layer_mean/geometry.cfg" --train_dir "/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --prj "/home/dwu/data/DECT/sinogram/sino_35_2.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/dect_2d_3_layer_mean/half/l2/35_all" --checkpoint "l2_depth_3/all/5.h5" --device "3" --energy "b" &>> ./outputs/dect2d_1.log &
wait
echo "8/14"
python3 test2d_dect.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/dect_2d_3_layer_mean/geometry.cfg" --train_dir "/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --prj "/home/dwu/data/DECT/sinogram/sino_35_2.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/dect_2d_3_layer_mean/half/l2/35" --checkpoint "l2_depth_3/dose_rate_4/25.h5" --device "2" --energy "a" &>> ./outputs/dect2d_0.log &
python3 test2d_dect.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/dect_2d_3_layer_mean/geometry.cfg" --train_dir "/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --prj "/home/dwu/data/DECT/sinogram/sino_35_2.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/dect_2d_3_layer_mean/half/l2/35" --checkpoint "l2_depth_3/dose_rate_4/25.h5" --device "3" --energy "b" &>> ./outputs/dect2d_1.log &
wait
echo "10/14"
python3 test2d_dect.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/dect_2d_3_layer_mean/geometry.cfg" --train_dir "/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --prj "/home/dwu/data/DECT/sinogram/sino_35_2.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/dect_2d_3_layer_mean/half/wgan/35_all" --checkpoint "l2_depth_3_wgan/all/5.h5" --device "2" --energy "a" &>> ./outputs/dect2d_0.log &
python3 test2d_dect.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/dect_2d_3_layer_mean/geometry.cfg" --train_dir "/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --prj "/home/dwu/data/DECT/sinogram/sino_35_2.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/dect_2d_3_layer_mean/half/wgan/35_all" --checkpoint "l2_depth_3_wgan/all/5.h5" --device "3" --energy "b" &>> ./outputs/dect2d_1.log &
wait
echo "12/14"
python3 test2d_dect.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/dect_2d_3_layer_mean/geometry.cfg" --train_dir "/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --prj "/home/dwu/data/DECT/sinogram/sino_35_2.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/dect_2d_3_layer_mean/half/wgan/35" --checkpoint "l2_depth_3_wgan/dose_rate_4/25.h5" --device "2" --energy "a" &>> ./outputs/dect2d_0.log &
python3 test2d_dect.py --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/dect_2d_3_layer_mean/geometry.cfg" --train_dir "/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/" --nslice_mean "3" --img_norm "0.019" --islices "0" "-1" --filter "hann" --margin "96" --vmin "-0.16" --vmax "0.24" --prj "/home/dwu/data/DECT/sinogram/sino_35_2.mat" --output "/home/dwu/trainData/deep_denoiser_ensemble/test/dect_2d_3_layer_mean/half/wgan/35" --checkpoint "l2_depth_3_wgan/dose_rate_4/25.h5" --device "3" --energy "b" &>> ./outputs/dect2d_1.log &
wait
echo "14/14"
wait
cat ./outputs/dect2d_0.log ./outputs/dect2d_1.log > ./outputs/dect2d.log
rm ./outputs/dect2d_0.log
rm ./outputs/dect2d_1.log
