import pytorch_lightning as pl

from model_domain_aware import DIVA
from data import CovidXDataModule

from pytorch_lightning.loggers import WandbLogger
import wandb

pl.seed_everything(42, workers=True)
'''
TRAINING PARAMETERS
'''
img_dir = 'D:/Datasets/COVIDX/ResizedData'
epochs = 10

latent_dimension = 100

beta = 0.01
alpha_y = 1
alpha_d = 1

dm = CovidXDataModule(img_dir, batch_size=16, domain_aware=True)

print(f'wandb login: {wandb.login(key="99ecdbb4fcebc379c7df8b8f11b69c805e9f3f5d")}')

model = DIVA(latent_dimension, beta, alpha_y, alpha_d)

logger = WandbLogger(project='CovidX-domain-aware', name=f'DIVA')

trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices=1, logger=logger, callbacks=[])

trainer.fit(model, datamodule=dm)

trainer.test(model, datamodule=dm)

wandb.finish()
