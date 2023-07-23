from typing import Any, Optional
import torch
from torch import nn
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_accuracy
import pytorch_lightning as pl
import wandb

from data import CLASS_LABELS, DOMAIN_LABELS


class DIVA(pl.LightningModule):
    def __init__(self, latent_dim, beta, alpha_y, y_dim=3, d_dim=8):
        super().__init__()
        self.metric = multiclass_accuracy
        self.beta = beta
        self.alpha_y = alpha_y

        self.latent_dim = latent_dim
        self.y_dim = y_dim  
        self.d_dim = d_dim

        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.latent_dim)

        self.y_classifier = MLPClassifier(self.latent_dim, self.y_dim)
        self.d_classifier = MLPClassifier(self.latent_dim, self.d_dim)

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, (y, d) = batch
        
        z = self.encoder(x)

        r = self.decoder(z)    

        yhat = self.y_classifier(z)
        dhat = self.d_classifier(z)

        recon_loss, kl_div = self.reconstruction_information(x, r, None, None)
        class_loss, class_acc = self.classification_information(y, yhat)
        domain_loss, domain_acc = self.classification_information(d, dhat)
        loss = recon_loss + self.alpha_y * class_loss # - self.beta * kl_div 

        # Log loss and metric
        self.log('train/loss', loss)
        self.log('train/recon_loss', recon_loss)
        # self.log('train_kl_div', kl_div)
        self.log('train/class_loss', class_loss)
        self.log('train/domain_loss', domain_loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, (y, d) = batch
        
        z = self.encoder(x)

        r = self.decoder(z)    

        yhat = self.y_classifier(z)
        dhat = self.d_classifier(z)

        recon_loss, kl_div = self.reconstruction_information(x, r, None, None)
        class_loss, class_acc = self.classification_information(y, yhat)
        domain_loss, domain_acc = self.classification_information(d, dhat)
        loss = recon_loss + self.alpha_y * class_loss # - self.beta * kl_div 

        # Log loss and metric
        self.log('val/loss', loss)
        self.log('val/recon_loss', recon_loss)
        # self.log('train_kl_div', kl_div)
        self.log('val/class_loss', class_loss)
        self.log('val/domain_loss', domain_loss)
        
        if batch_idx == 0:
            recon = wandb.Image(r[0], caption="Reconstruction")
            image = wandb.Image(x[0], caption="Original")
            self.logger.experiment.log({"visualisations": {"reconstruction": recon, "original": image}})

        return r
    
    def test_step(self, batch, batch_idx):
        x, (y, d) = batch
        
        z = self.encoder(x)

        r = self.decoder(z)    

        yhat = self.y_classifier(z)
        dhat = self.d_classifier(z)

        if batch_idx == 0:
            self._y = [y]
            self._yhat = [yhat]
            self._d = [d]
            self._dhat = [dhat]
        else:
            self._y.append(y)
            self._yhat.append(yhat)
            self._d.append(d)
            self._dhat.append(dhat)

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

        self.logger.experiment.log({'class_scores': cm})

        gt = torch.cat(self._d).cpu().numpy()
        preds = torch.cat(self._dhat).cpu().numpy()

        cm = wandb.plot.confusion_matrix(
            probs=preds,
            d_true=gt,
            class_names=DOMAIN_LABELS
        )

        self._d = None
        self._dhat = None

        self.logger.experiment.log({'domain_scores': cm})
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def reconstruction_information(self, inputs, reconstructions, z_q, qz):
        # Reconstruction loss
        recon_loss = F.mse_loss(inputs, reconstructions)

        # z_p_loc, z_p_scale = torch.zeros(z_q.size()[0], self.latent_dim).cuda(), torch.ones(z_q.size()[0], self.latent_dim).cuda()
        # pz = torch.distributions.Normal(z_p_loc, z_p_scale)

        # kl_div = torch.sum(pz.log_prob(z_q) - qz.log_prob(z_q))

        return recon_loss, None # kl_div
    
    def classification_information(self, y, yhat):
        class_loss = F.cross_entropy(yhat, y)
        class_acc = []
        # for i in range(self.y_dim):
        #     preds = yhat[y==i]
        #     targets = y[y==i]
        #     acc = torch.sum(preds == targets) / len(targets)
        #     class_acc.append(acc)
        return class_loss, class_acc
    
    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        qz = torch.distributions.Normal(mu, std)
        return qz


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
        self.mp = nn.MaxPool2d((2, 2))

    def forward(self, x):
        return self.mp(self.relu(self.bn(self.conv(x))))


class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.up(self.relu(self.bn(self.conv(x))))

# Encoder that maps a 299x299 grayscale image to a latent dimension
class Encoder(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()
        # n x 1 x 128 x 128
        self.c1 = ConvBlock(1, 32, 3, 1, 1)
        # n x 16 x 64 x 64
        self.c2 = ConvBlock(32, 64, 3, 1, 1)
        # n x 32 x 32 x 32
        self.c3 = ConvBlock(64, 64, 3, 1, 1)
        # n x 64 x 16 x 16
        self.c4 = ConvBlock(64, 64, 3, 1, 1)
        # n x 64 x 8 x 8
        # self.c5 = ConvBlock(64, 64, 3, 1, 1)
        # n x 64 x 4 x 4
        
        self.mu = nn.Linear(4096, latent_dim)
        # self.log_var = nn.Linear(12*12*64, latent_dim)
        
    def forward(self, x):
        z = self.c1(x)
        z = self.c2(z)
        z = self.c3(z)
        z = self.c4(z)
        # z = self.c5(z)

        z = z.view(-1, 4096)
        mu = self.mu(z)
        # log_var = self.log_var(z)
        return mu, None #mu , log_var


# An encoder to reverse the process of the encoder
class Decoder(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 4096)
        self.d1 = DeConvBlock(64, 64, 3, 1, 1)
        self.d2 = DeConvBlock(64, 32, 3, 1, 1)
        self.d3 = DeConvBlock(32, 16, 3, 1, 1)
        self.d4 = DeConvBlock(16, 1, 3, 1, 1)
        # self.d5 = DeConvBlock(16, 1, 3, 1, 1)
        
        # self.fc = nn.Linear(latent_dim, 4*4*64)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 64, 8, 8)
        z = self.d1(z)
        z = self.d2(z)
        z = self.d3(z)
        z = self.d4(z)
        # z = self.d5(z)
        return z
