from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_accuracy
import pytorch_lightning as pl
import wandb

from data import CLASS_LABELS


class VAE_Classifier(pl.LightningModule):
    def __init__(self, latent_dim, y_dim=3, d_dim=-1):
        super().__init__()
        self.metric = multiclass_accuracy
        self.beta = 0.1
        self.alpha_y = 1

        self.latent_dim = latent_dim
        self.y_dim = y_dim  
        self.d_dim = d_dim

        self.model = VAE(latent_dim, y_dim, d_dim)

    def training_step(self, batch, batch_idx):
        x, y = batch
        recons, yhat, z_q, qz = self.model(x)

        recon_loss, kl_div = self.reconstruction_information(x, recons, z_q, qz)
        class_loss, class_acc = self.classification_information(y, yhat)
        loss = recon_loss - self.beta * kl_div + self.alpha_y * class_loss

        # Log loss and metric
        self.log('train_loss', loss)
        self.log('train_recon_loss', recon_loss)
        self.log('train_kl_div', kl_div)
        self.log('train_class_loss', class_loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        recons, yhat, z_q, qz = self.model(x)

        recon_loss, kl_div = self.reconstruction_information(x, recons, z_q, qz)
        class_loss, class_acc = self.classification_information(y, yhat)

        loss = recon_loss - self.beta * kl_div + self.alpha_y * class_loss
        # Log loss and metric
        self.log('val_loss', loss)
        # self.log({'val': {f"{label}_acc": class_acc[i] for i, label in enumerate(CLASS_LABELS)}})
        
        if batch_idx == 0:
            recon = wandb.Image(recons[0], caption="Reconstruction")
            image = wandb.Image(x[0], caption="Original")
            self.logger.experiment.log({"visualisations": {"reconstruction": recon, "original": image}})

        return recons
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        recons, yhat, z_q, qz = self.model(x)

        if batch_idx == 0:
            self._y = [y]
            self._yhat = [yhat]
        else:
            self._y.append(y)
            self._yhat.append(yhat)

    def on_test_end(self):
        gt = torch.cat(self._y).cpu().numpy()
        preds = torch.cat(self._yhat).cpu().numpy()

        cm = wandb.plot.confusion_matrix(
            probs=preds,
            y_true=gt,
            class_names=CLASS_LABELS
        )

        self._y = None
        self._yhat = None

        self.logger.experiment.log({'confidence_matrix': cm})
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def reconstruction_information(self, inputs, reconstructions, z_q, qz):
        # Reconstruction loss
        recon_loss = F.mse_loss(inputs, reconstructions)

        z_p_loc, z_p_scale = torch.zeros(z_q.size()[0], self.latent_dim).cuda(), torch.ones(z_q.size()[0], self.latent_dim).cuda()
        pz = torch.distributions.Normal(z_p_loc, z_p_scale)

        kl_div = torch.sum(pz.log_prob(z_q) - qz.log_prob(z_q))

        return recon_loss, kl_div
    
    def classification_information(self, y, yhat):
        class_loss = F.cross_entropy(yhat, y)
        class_acc = []
        # for i in range(self.y_dim):
        #     preds = yhat[y==i]
        #     targets = y[y==i]
        #     acc = torch.sum(preds == targets) / len(targets)
        #     class_acc.append(acc)
        return class_loss, class_acc
    

class VAE(nn.Module):
    def __init__(self, latent_dim, y_dim, d_dim):
        super().__init__()
        self.metric = multiclass_accuracy
        self.beta = 0.1
        self.alpha_y = 1

        self.z_dim = latent_dim
        self.y_dim = y_dim
        self.d_dim = d_dim

        self.encoder = Encoder(self.z_dim)
        self.decoder = Decoder(self.z_dim)  
        self.classifier = MLPClassifier(self.z_dim, self.y_dim) 

    def encode(self, x):
        mu, log_var = self.encoder(x)
        return mu, log_var
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        qz = torch.distributions.Normal(mu, std)
        return qz

    def forward(self, x):
        mu, log_var = self.encode(x)
        qz = self.reparametrize(mu, log_var) # Distribution of z given the encoder
        z_q = qz.rsample()

        reconstruction = self.decode(z_q)
        y_predicted = self.classifier(z_q)

        return reconstruction, y_predicted, z_q, qz


# A classifier that takes a latent vector and reduces it to a single class
class MLPClassifier(nn.Module):
    def __init__(self, latent_dim, class_dim) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, class_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        z = self.relu(z)
        return self.fc(z)   


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    

class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    

# Encoder that maps a 299x299 grayscale image to a latent dimension
class Encoder(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()
        self.c1 = ConvBlock(1, 16, 5, 3, 0)
        self.c2 = ConvBlock(16, 32, 3, 2, 0)
        self.c3 = ConvBlock(32, 64, 3, 2, 0)
        self.c4 = ConvBlock(64, 64, 2, 2, 0)
        self.c5 = ConvBlock(64, 64, 3, 1, 1)
        
        self.mu = nn.Linear(12*12*64, latent_dim)
        self.log_var = nn.Linear(12*12*64, latent_dim)
        
    def forward(self, x):
        z = self.c1(x)
        z = self.c2(z)
        z = self.c3(z)
        z = self.c4(z)
        z = self.c5(z)

        z = z.view(-1, 12*12*64)
        mu = self.mu(z)
        log_var = self.log_var(z)
        return mu, log_var


# An encoder to reverse the process of the encoder
class Decoder(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 12*12*64)
        self.d1 = DeConvBlock(64, 64, 3, 1, 1)
        self.d2 = DeConvBlock(64, 64, 2, 2, 0)
        self.d3 = DeConvBlock(64, 32, 3, 2, 0)
        self.d4 = DeConvBlock(32, 16, 3, 2, 0)
        self.d5 = DeConvBlock(16, 1, 5, 3, 0)
        
    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 64, 12, 12)
        z = self.d1(z)
        z = self.d2(z)
        z = self.d3(z)
        z = self.d4(z)
        z = self.d5(z)
        return z