# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# generate the scripts to process the datasets


# %%
import glob
import os
import copy
import pathlib
from datetime import datetime


# %%
job_name = 'train2d_wgan'
devices = ['1', '2', '3']
nprocesses = 3
args = {'config': './config/wgan/l2_depth_3.cfg', 
        }
logprefix = './output/wgan/l2_depth_3'
slurm_header = """#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=%s
#SBATCH --nodelist=gpu-node008
#SBATCH --cpus-per-task=16
#SBATCH --time=0
"""%job_name


# %%
# identify the input directories
# dose_rates = [2,4,8,12,16]
dose_rates = [6]
cmds = []
logs = []
for dose_rate in dose_rates:
    cmd = copy.deepcopy(args)
    cmd['IO.source'] = 'dose_rate_%d'%dose_rate
    cmd['IO.tag'] = 'l2_depth_3_wgan/dose_rate_%d'%dose_rate
    cmd['Training.device'] = devices[len(cmds) % len(devices)]
    cmds.append(cmd)
    
    logs.append(logprefix + '_dose_rate_%d'%dose_rate)

# %%
# generate scripts
with open('scripts/%s.sh'%job_name, 'w') as f:
    # slurm
    f.write(slurm_header + '\n\n')
    f.write('cd ..\n')

    for k, cmd in enumerate(cmds):
        argstr = ' '.join(['--%s "%s"'%(name, cmd[name]) for name in cmd])
        logstr = '&> %s.log'%(logs[k])
        f.write('python3 %s.py '%job_name + argstr + ' ' + logstr + ' &\n')
        if (k+1)%nprocesses == 0:
            f.write('wait\n')
            f.write('echo "%d/%d"\n'%(k+1, len(cmds)))
    f.write('wait\n')


# %%



