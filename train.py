import wandb
import pytorch_lightning as pl

from model import VAE_Classifier
from data import CovidXDataModule

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

model = VAE_Classifier(100)

dm = CovidXDataModule()

logger = WandbLogger(project='CovidX', name='VAE/test_new_logs', log_model='all')

# checkpoint_callback = ModelCheckpoint(monitor='val.loss', mode='min')

trainer = pl.Trainer(max_epochs=10, accelerator="gpu", devices=1, logger=logger, callbacks=[])

trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=dm)