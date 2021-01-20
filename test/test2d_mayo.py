#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Test single network on Mayo dataset
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


# In[24]:


import argparse
import configparser
parser = argparse.ArgumentParser()
parser.add_argument('--prj', default = '/home/dwu/data/lowdoseCTsets/L291_full_sino.mat')
parser.add_argument('--output', default = '/home/dwu/trainData/deep_denoiser_ensemble/test/mayo_2d_3_layer_mean/debug_single/L291')
parser.add_argument('--geometry', default = '/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg')
parser.add_argument('--islices', type = int, nargs = 2, default = [0,-1])
parser.add_argument('--nslice_mean', type = int, default=3)
# Note that 1e5 is for single slice noise insertion
parser.add_argument('--N0', type = float, default = 1e5)
parser.add_argument('--dose_rate', type=float, default = 4)

parser.add_argument('--device', type = int, default = 0)
parser.add_argument('--img_norm', type = float, default = 0.019)
parser.add_argument('--train_dir', default = '/home/dwu/trainData/deep_denoiser_ensemble/train/mayo_2d_3_layer_mean/')
parser.add_argument('--checkpoint', default = 'l2_depth_3/all/5.h5')

parser.add_argument('--vmin', type=float, default = -0.16)
parser.add_argument('--vmax', type=float, default = 0.24)
parser.add_argument('--filter', default = 'hann')
parser.add_argument('--margin', type=int, default=96)

if sys.argv[0] != 'test2d_mayo.py':
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


# In[25]:


os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%args.device
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# In[26]:


print ('[Geometry]')
projector = ct_projector.ct_projector()
projector.from_file(args.geometry)
projector.nv = 1
for k in vars(projector):
    print (k, '=', getattr(projector, k))


# In[27]:


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


# In[28]:


K.clear_session()
model = tf.keras.models.load_model(os.path.join(args.train_dir, args.checkpoint))

preds = []

print (len(prj), flush=True)
for islice in range(prj.shape[0]):
    print (islice, end=',', flush=True)
    
    angles = projector.get_angles()
    fprj = projector.ramp_filter(prj[[islice]], args.filter)
    fbp = projector.fbp_fan_bp(fprj, angles) - 1
    
    pred = model.predict(fbp[:, 0, :, :, np.newaxis])[..., 0]
    pred = pred[:, np.newaxis, :, :]
    
    preds.append(pred)    
print ('')

preds = np.array(preds)


# In[29]:


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

# image
sitk_img = sitk.GetImageFromArray((preds.squeeze() * 1000).astype(np.int16))
sitk_img.SetSpacing([float(projector.dx), float(projector.dy), float(projector.dz * args.nslice_mean)])
sitk.WriteImage(sitk_img, args.output + '.nii')

print ('Done', flush=True)


# In[30]:


if showplot:
    plt.figure(figsize=[16,8])
    plt.subplot(121);plt.imshow(fbp[0,0,args.margin:-args.margin, args.margin:-args.margin], 'gray', vmin = args.vmin, vmax = args.vmax)
    plt.subplot(122);plt.imshow(pred[0,0,args.margin:-args.margin, args.margin:-args.margin], 'gray', vmin = args.vmin, vmax = args.vmax)


# In[ ]:




