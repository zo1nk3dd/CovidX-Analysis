import pytorch_lightning as pl
from model import Classifier
from data import CovidXDataModule
from pytorch_lightning.loggers import WandbLogger

model = Classifier()

dm = CovidXDataModule()

logger = WandbLogger(log_model='all')

checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min')

trainer = pl.Trainer(max_epochs=10, accelerator="gpu", devices=1, logger=logger, callbacks=[checkpoint_callback])

trainer.fit(model, datamodule=dm)