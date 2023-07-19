import pytorch_lightning as pl

from model import VAE_Classifier
from data import CovidXDataModule

from pytorch_lightning.loggers import WandbLogger
import wandb


'''
TRAINING PARAMETERS
'''
img_dir = 'D:/Datasets/COVIDX/Data'
epochs = 10

latent_dimension = 100

beta = 0
alpha_y = 1

model = VAE_Classifier(latent_dimension, beta, alpha_y)

dm = CovidXDataModule(img_dir, batch_size=16)

wandb.login(key='99ecdbb4fcebc379c7df8b8f11b69c805e9f3f5d')
logger = WandbLogger(project='CovidX-pic_quality', name=f'model_loss_paper, beta: {beta}', log_model='all')

trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices=1, logger=logger, callbacks=[])

trainer.fit(model, datamodule=dm)

trainer.test(model, datamodule=dm)
