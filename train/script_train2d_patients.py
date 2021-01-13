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
job_name = 'train2d_patients'
devices = ['1', '2', '3']
nprocesses = 3
args = {'config': './config/baseline/l2_depth_3_patient.cfg', 
        }
logprefix = './output/baseline/l2_depth_3'
slurm_header = """#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=%s
#SBATCH --nodelist=gpu-node008
#SBATCH --cpus-per-task=16
#SBATCH --time=0
"""%job_name


# %%
# identify the input directories
patients = ['L291', 'L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L310', 'L333', 'L506']
cmds = []
logs = []
for patient in patients:
    cmd = copy.deepcopy(args)
    cmd['IO.train'] = patient
    cmd['IO.tag'] = 'l2_depth_3/%s'%patient
    cmd['Training.device'] = devices[len(cmds) % len(devices)]
    cmds.append(cmd)
    
    logs.append(logprefix + '_' + patient)

# %%
# generate scripts
with open('scripts/%s.sh'%job_name, 'w') as f:
    # slurm
    f.write(slurm_header + '\n\n')
    f.write('cd ..\n')

    for k, cmd in enumerate(cmds):
        argstr = ' '.join(['--%s "%s"'%(name, cmd[name]) for name in cmd])
        logstr = '&> %s.log'%(logs[k])
        f.write('python3 train2d.py ' + argstr + ' ' + logstr + ' &\n')
        if (k+1)%nprocesses == 0:
            f.write('wait\n')
            f.write('echo "%d/%d"\n'%(k+1, len(cmds)))
    f.write('wait\n')


# %%



