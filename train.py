import pytorch_lightning as pl

from model_compressed import VAE_Classifier
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

beta = 0
alpha_y = 1

dm = CovidXDataModule(img_dir, batch_size=16)

print(f'wandb login: {wandb.login(key="99ecdbb4fcebc379c7df8b8f11b69c805e9f3f5d")}')

for ld in [100, 1000, 10000]:
    for mse in [1, 100, 10000]:
        model = VAE_Classifier(ld, mse, alpha_y)

        logger = WandbLogger(project='CovidX-pic_quality', name='Autoencoder-resizeconv-mseloss-{mse}-latent-{ld}', log_model='all')

        trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices=1, logger=logger, callbacks=[])

        trainer.fit(model, datamodule=dm)

        trainer.test(model, datamodule=dm)
