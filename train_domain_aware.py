import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model_domain_aware import DIVA
from data import CovidXDataModule

from pytorch_lightning.loggers import WandbLogger
import wandb

class BetaScheduler(pl.Callback):
    def on_train_epoch_end(self, trainer, module):
        if module.beta < 1:
            module.beta += 0.01

pl.seed_everything(42, workers=True)
'''
TRAINING PARAMETERS
'''
img_dir = 'D:/Datasets/COVIDX/ResizedData'

zy_dim = 64
zd_dim = 64
zx_dim = 64

beta = 1.0
alpha_y = 7500
alpha_d = 10000
reconstruction_penalty = 10000

dm = CovidXDataModule(img_dir, batch_size=32, domain_aware=True)

print(f'wandb login: {wandb.login(key="99ecdbb4fcebc379c7df8b8f11b69c805e9f3f5d")}')

model = DIVA(zy_dim, zd_dim, zx_dim, beta, alpha_y, alpha_d, reconstruction_penalty)

logger = WandbLogger(project='CovidX-domain-aware', name=f'DIVA-smaller-latents', log_model=True)

trainer = pl.Trainer( 
    accelerator="gpu", 
    devices=1, 
    logger=logger, 
    callbacks=[
        BetaScheduler(),
        EarlyStopping(monitor='val/loss', patience=10, mode='min'),
        ModelCheckpoint(monitor='val/loss', mode='min')
    ])

trainer.fit(model, datamodule=dm)

trainer.test(model, datamodule=dm)

trainer.predict(model, datamodule=dm)

wandb.finish()
