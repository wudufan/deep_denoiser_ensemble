#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Test ensemble network on mayo dataset
'''


# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import os
import sys
import SimpleITK as sitk
import glob
import h5py
import configparser


# In[3]:


sys.path.append('..')
import CTProjector.projector.ct_projector as ct_projector
import model.unet as unet


# In[4]:


import argparse
import configparser
parser = argparse.ArgumentParser()
parser.add_argument('--prj', default = '/home/dwu/data/lowdoseCTsets/L291_full_sino.mat')
parser.add_argument('--output', default = '/home/dwu/trainData/deep_denoiser_ensemble/test/mayo_2d_3_layer_mean/debug/L291')
parser.add_argument('--geometry', default = '/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg')
parser.add_argument('--islices', type = int, nargs = 2, default = [0,-1])
parser.add_argument('--nslice_mean', type = int, default=3)
# Note that 1e5 is for single slice noise insertion
parser.add_argument('--N0', type = float, default = 1e5)
parser.add_argument('--dose_rate', type=float, default = 4)

parser.add_argument('--device', type = int, default = 0)
parser.add_argument('--img_norm', type = float, default = 0.019)
parser.add_argument('--train_dir', default = '/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/')
parser.add_argument('--checkpoint', type=int, default = 25)
parser.add_argument('--tags', default = 'l2_depth_3/dose_rate_2,l2_depth_3/dose_rate_4,l2_depth_3/dose_rate_8,l2_depth_3/dose_rate_16')

parser.add_argument('--checkpoint_smooth', default = 'l2_depth_3/dose_rate_16/25.h5')

parser.add_argument('--vmin', type=float, default = -0.16)
parser.add_argument('--vmax', type=float, default = 0.24)
# for single slice
parser.add_argument('--N0_ref', type=float, default = 2.5e4)
parser.add_argument('--filter', default = 'hann')
parser.add_argument('--margin', type=int, default=96)

if sys.argv[0] == 'ipykernel_launcher.py':
    showplot = True
    args = parser.parse_args(['--device', '0',
                              '--prj', '/home/dwu/data/lowdoseCTsets/L291_quarter_sino.mat',
                              '--N0', '-1',
                              '--dose_rate', '8',
#                               '--islices', '30','32',
#                               '--tags', 'l2_depth_3_wgan/dose_rate_2,l2_depth_3_wgan/dose_rate_4,l2_depth_3_wgan/dose_rate_8,l2_depth_3_wgan/dose_rate_16',
                             ])
else:
    showplot = False
    args = parser.parse_args()

for k in vars(args):
    print (k, '=', getattr(args, k))


# In[5]:


os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%args.device
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# In[6]:


print ('[Geometry]')
projector = ct_projector.ct_projector()
projector.from_file(args.geometry)
projector.nv = 1
for k in vars(projector):
    print (k, '=', getattr(projector, k))


# In[7]:


print ('Loading data...', end='', flush=True)
with h5py.File(args.prj, 'r') as f:
    prj = f['sino']
    if args.islices[0] < args.islices[1]:
        prj = np.copy(prj[args.islices[0] * args.nslice_mean:args.islices[1]*args.nslice_mean])
    else:
        prj = np.copy(prj[:prj.shape[0]//args.nslice_mean*args.nslice_mean])
print ('Done', flush=True)

# take layer-means
prj = prj.reshape([-1, args.nslice_mean, prj.shape[1], prj.shape[2]]).mean(1)
# reshape
prj = prj[:, :, np.newaxis, :] / args.img_norm

if args.N0 > 0:
    print ('Adding noise...', end='', flush=True)
    
    N0 = args.N0 * args.nslice_mean
    prj = prj + np.sqrt((1 - 1 / args.dose_rate) * args.dose_rate * np.exp(prj * args.img_norm) / N0) * np.random.normal(size = prj.shape) / args.img_norm
    prj = prj.astype(np.float32)
    
    print ('Done', flush=True)
else:
    print ('Use original projection', flush=True)


# In[8]:


def fbp_with_noise_estimation(prj, projector, args):
    '''
    Do FBP and use Noise2Noise to estimate the noise pattern
    '''
    angles = projector.get_angles()
    
    inds_offset = np.random.randint(0, 2, prj.shape[1]//2)
    inds1 = np.arange(0, prj.shape[1], 2) + inds_offset
    inds2 = np.arange(0, prj.shape[1], 2) + (1 - inds_offset)
    fprj = projector.ramp_filter(prj, args.filter)
    img1 = projector.fbp_fan_bp(np.copy(fprj[:, inds1, ...], 'C'), angles[inds1]) * 2
    img2 = projector.fbp_fan_bp(np.copy(fprj[:, inds2, ...], 'C'), angles[inds2]) * 2
    
    fbp = (img1 + img2) / 2 - 1
    noise = (img1 - img2) / 2
    
    return fbp, noise


# In[9]:


def estimate_smooth_image(fbp, args):
    model = tf.keras.models.load_model(os.path.join(args.train_dir, args.checkpoint_smooth))
    smooth = model.predict(fbp[:, 0, :, :, np.newaxis])[..., 0]
    smooth = smooth[:, np.newaxis, :, :]
    
    return smooth


# In[10]:


def estimate_noise_level(smooth, noise, projector, args):
    '''
    Estimate the noise level using forward projection
    '''
    # estimate the noise level of reference image
    angles = projector.get_angles()
    fp = projector.siddon_fan_fp(smooth + 1, angles)
    N0 = args.N0_ref * args.nslice_mean / np.mean(np.exp(-fp * args.img_norm))
    
    # noise insertion
    fp_noisy = -np.log(np.random.poisson(N0 * np.exp(-fp * args.img_norm)) / N0) / args.img_norm
    
    # noise level estimation
    recon_noiseless = projector.fbp_fan_bp(projector.ramp_filter(fp, args.filter), angles) - 1
    recon_noisy = projector.fbp_fan_bp(projector.ramp_filter(fp_noisy, args.filter), angles) - 1
    
    m = args.margin
    noise_ratio = np.linalg.norm((recon_noisy - recon_noiseless)[..., m:-m, m:-m]) / np.linalg.norm(noise[..., m:-m, m:-m])
    
    return noise_ratio


# In[11]:


def pred_meta_images(img, args):
    '''
    Predict the meta images to ensemble
    '''
    metas = [img]
    for tag in args.tags.split(','):
        model = tf.keras.models.load_model(os.path.join(args.train_dir, tag, '%d.h5'%args.checkpoint))
        pred = model.predict(img[:, 0, :, :, np.newaxis])[..., 0]
        pred = pred[:, np.newaxis, :, :]
        
        metas.append(pred)
    
    return np.array(metas)


# In[12]:


def preprocess(x, args):
    return x[..., args.margin:-args.margin, args.margin:-args.margin]

def ensemble_l2(target, metas, args):
    '''
    ensemble metas to target by minimizing the l2 distance
    ''' 
    
    y = preprocess(target, args)
    xs = [preprocess(m, args) for m in metas]
    
    M = np.zeros([len(metas), len(metas)])
    b = np.zeros([len(metas)])
    
    for k in range(len(xs)):
        b[k] = np.sum(y * xs[k])
        for j in range(len(xs)):
            M[k, j] = np.sum(xs[k] * xs[j])
    
    c = np.linalg.solve(M, b)
    
    return c.astype(np.float32)


# In[13]:


K.clear_session()
np.random.seed(0)

preds = []
coefs = []

print (len(prj), flush=True)
for islice in range(prj.shape[0]):
    print (islice, end=',', flush=True)
    
    fbp, noise = fbp_with_noise_estimation(prj[[islice], ...], projector, args)
    smooth = estimate_smooth_image(fbp, args)
    noise_ratio = estimate_noise_level(smooth, noise, projector, args)
    
    # predict the meta images
    fbp_metas = pred_meta_images(fbp, args)
    smooth_metas = pred_meta_images(smooth + noise, args)
    
    # get the combination coefficients
    # c = ensemble_l2(smooth + noise_ratio * noise, smooth_metas, args)
    # average
    c = np.ones(len(fbp_metas)) / len(fbp_metas)
    
    # combine the fbp metas to get the result
    pred = np.sum(fbp_metas * c[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis], 0)
    
    preds.append(pred)
    coefs.append(c)
print ('')

preds = np.array(preds)
coefs = np.array(coefs)


# In[15]:


# save the results
print ('Writing results...', end='', flush=True)

if not os.path.exists(os.path.dirname(args.output)):
    os.makedirs(os.path.dirname(args.output))

# running configuration
cfg = configparser.ConfigParser()
cfg.add_section('Test')
for k in vars(args):
    cfg['Test'][k] = str(getattr(args, k))
with open(args.output + '.cfg', 'w') as f:
    cfg.write(f)

# coefficients
np.save(args.output + '_coefs.npy', coefs)

# image
sitk_img = sitk.GetImageFromArray((preds.squeeze() * 1000).astype(np.int16))
sitk_img.SetSpacing([float(projector.dx), float(projector.dy), float(projector.dz * args.nslice_mean)])
sitk.WriteImage(sitk_img, args.output + '.nii')

print ('Done', flush=True)


# In[16]:


if showplot:
    plt.figure(figsize=[16,8])
    plt.subplot(121);plt.imshow(fbp[0,0,args.margin:-args.margin, args.margin:-args.margin], 'gray', vmin = args.vmin, vmax = args.vmax)
    plt.subplot(122);plt.imshow(pred[0,0,args.margin:-args.margin, args.margin:-args.margin], 'gray', vmin = args.vmin, vmax = args.vmax)


# In[ ]:




