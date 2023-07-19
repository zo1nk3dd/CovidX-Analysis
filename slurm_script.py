import os

cmds = [
    '#!/bin/bash',
    '#SBATCH --partition=vgpu',
    '#SBATCH --gres=gpu:1',
    '#SBATCH --job-name=covidx_model',
    '#SBATCH --output=covidx_model.out',
    'conda activate torch2',
    f'python ./train.py'
]

with open('job.sh', 'w') as f:
    f.write('\n'.join(cmds))

os.system('sbatch job.sh')
