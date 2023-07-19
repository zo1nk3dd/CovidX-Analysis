print("File running")

import pytorch_lightning as pl

from model import VAE_Classifier
from data import CovidXDataModule

from pytorch_lightning.loggers import WandbLogger
import wandb


'''
TRAINING PARAMETERS
'''
img_dir = 'D:/Datasets/COVIDX/Data'
epochs = 50

latent_dimension = [25, 50, 100]

beta = [0.1, 1, 10]
alpha_y = [1, 100, 10000]


for latent_dim in latent_dimension:
    for beta in beta:
        for alpha_y in alpha_y:
            model = VAE_Classifier(latent_dim=latent_dimension, beta=beta, alpha_y=alpha_y)

            dm = CovidXDataModule(img_dir)

            wandb.login(key='99ecdbb4fcebc379c7df8b8f11b69c805e9f3f5d')
            logger = WandbLogger(project='CovidX-hparams', name=f'l_dim: {latent_dimension}, beta: {beta}, alpha_y: {alpha_y}', log_model='all')

            trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices=1, logger=logger, callbacks=[])

            trainer.fit(model, datamodule=dm)

            trainer.test(model, datamodule=dm)
