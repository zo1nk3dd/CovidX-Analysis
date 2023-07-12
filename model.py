from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy
import pytorch_lightning as pl


class VAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = VAE(100, 50, 50, 50, 10)
        self.loss = nn.CrossEntropyLoss()
        self.metric = MulticlassAccuracy()
        self.beta = 1
        self.gamma = 1

    def training_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        overall_loss, recon_loss, kl_div, class_loss = loss

        # Log loss and metric
        self.log('train_recon_loss', recon_loss)
        self.log('train_kl_div', kl_div)
        self.log('train_class_loss', class_loss)
        return overall_loss
    
    def validation_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val_loss', loss[0])
        self.log('val_accuracy', acc)
        
        return preds
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        recons, yhat, qz, pz, z_q = self.model(x)

        # Reconstruction loss
        recon_loss = self.model.reconstruction_loss(x, recons)

        # KL divergence
        kl_div = self.model.kl_divergence(pz, qz, z_q)

        # Classification loss
        class_loss = self.model.classification_loss(y, yhat)

        loss = recon_loss - self.beta * kl_div + self.gamma * class_loss

        return yhat, (loss, recon_loss, kl_div, class_loss), self.metric(yhat, y)

        



class BasicModel(nn.Module):
    def __init__(self, input_size: int = 299 * 299) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(16, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4, stride=2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*3*32, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 3)
        )

    def forward(self, x):
        return self.model(x)
    
    

class VAE(nn.Module):
    def __init__(self, z_dim, x_dim, y_dim, d_dim, beta) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.d_dim = d_dim
        self.beta = beta

        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)  
        self.classifier = MLPClassifier(z_dim, y_dim)

    def reconstruction_loss(self, x, recons):
        return F.mse_loss(recons, x)

    def kl_divergence(self, pz, qz, z_q):
        return torch.sum(pz.log_prob(z_q) - qz.log_prob(z_q))    
    
    def classification_loss(self, y, y_hat):
        return F.cross_entropy(y_hat, y)


    def forward(self, x):
        mu, log_var = self.encoder(x)

        std = torch.exp(0.5 * log_var)

        qz = torch.distributions.Normal(mu, std)
        z_q = qz.rsample()

        z_p_loc, z_p_scale = torch.zeros(z_q.size()[0], self.z_dim).cuda(), torch.ones(z_q.size()[0], self.z_dim).cuda()

        # Reparameterization trick
        pz = torch.distributions.Normal(z_p_loc, z_p_scale)

        recons = self.decoder(z_q)

        yhat = self.classifier(z_q)

        return recons, yhat, qz, pz, z_q


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

