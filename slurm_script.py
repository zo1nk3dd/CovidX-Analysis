import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--beta', type=float, default=1, help='kl-divergence weighting')
parser.add_argument('--alpha_y', type=float, default=1, help='weighting for class predictions')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--latent_dim', type=int, default=100, help='latent dimension')

args = parser.parse_args()

cmds = [
    '#!/bin/bash',
    '#SBATCH --partition=vgpu',
    '#SBATCH --gres=gpu:1',
    '#SBATCH --job-name=covidx_model',
    'conda activate torch2',
    f'python ./train.py --beta={args.beta} --alpha_y={args.alpha_y} --epochs={args.epochs} --latent_dim={args.latent_dim}'
]

with open('job.sh', 'w') as f:
    f.write('\n'.join(cmds))

os.system('sbatch job.sh')