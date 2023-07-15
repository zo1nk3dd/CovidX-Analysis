print("File running")

import pytorch_lightning as pl

from model import VAE_Classifier
from data import CovidXDataModule

from pytorch_lightning.loggers import WandbLogger
import wandb
import argparse

print("Importing finished")

parser = argparse.ArgumentParser()

parser.add_argument('--beta', type=float, default=1, help='kl-divergence weighting')
parser.add_argument('--alpha_y', type=float, default=1, help='weighting for class predictions')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--latent_dim', type=int, default=100, help='latent dimension')
parser.add_argument('--img_dir', type=str, default='D:\Datasets\COVIDX\Data', help='path to image directory')

args = parser.parse_args()

model = VAE_Classifier(latent_dim=args.latent_dim, beta=args.beta, alpha_y=args.alpha_y)

dm = CovidXDataModule(img_dir=args.img_dir)

wandb.login(key='99ecdbb4fcebc379c7df8b8f11b69c805e9f3f5d')
logger = WandbLogger(project='CovidX', name='VAE/tune_hparams', log_model='all')

# checkpoint_callback = ModelCheckpoint(monitor='val.loss', mode='min')

trainer = pl.Trainer(max_epochs=args.epochs, accelerator="gpu", devices=1, logger=logger, callbacks=[])

print("Training")

trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=dm)
