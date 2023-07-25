from typing import Any, Optional
import torch
from torch import nn
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_accuracy
import pytorch_lightning as pl
import wandb

from data import CLASS_LABELS, DOMAIN_LABELS


class DIVA(pl.LightningModule):
    def __init__(self, zy_dim, zd_dim, zx_dim, beta, alpha_y, alpha_d, reconstruction_penalty, y_dim=3, d_dim=8):
        super().__init__()
        self.metric = multiclass_accuracy
        self.beta = 0.1 # Beta scheduling used
        self.alpha_y = alpha_y
        self.alpha_d = alpha_d
        self.recon_pen = reconstruction_penalty

        self.zy_dim = zy_dim
        self.zd_dim = zd_dim
        self.zx_dim = zx_dim
        self.y_dim = y_dim  
        self.d_dim = d_dim

        self.qzy = Encoder(zy_dim)
        self.qzd = Encoder(zd_dim)
        self.qzx = Encoder(zx_dim)

        self.px = Decoder(zy_dim + zd_dim + zx_dim)

        self.prior_y = Prior(self.y_dim, self.zy_dim)
        self.prior_d = Prior(self.d_dim, self.zd_dim)

        self.y_classifier = MLPClassifier(self.zy_dim, self.y_dim)
        self.d_classifier = MLPClassifier(self.zd_dim, self.d_dim)

        self.save_hyperparameters()

    def forward(self, x, y, d):
        y_mu, y_log_var = self.qzy(x)
        d_mu, d_log_var = self.qzd(x)
        x_mu, x_log_var = self.qzx(x)
        
        zy = self.reparametrize(y_mu, y_log_var)
        zd = self.reparametrize(d_mu, d_log_var)
        zx = self.reparametrize(x_mu, x_log_var)

        y_z = zy.rsample()
        d_z = zd.rsample()
        x_z = zx.rsample()

        r = self.px(torch.cat((y_z, d_z, x_z), dim=1))    

        yhat = self.y_classifier(y_z)
        dhat = self.d_classifier(d_z)

        return r, yhat, dhat, zy, y_z, zd, d_z, zx, x_z
    
    def loss_values(self, batch):
        x, l = batch
        y, d = l

        r, yhat, dhat, zy, y_z, zd, d_z, zx, x_z = self.forward(x, y, d)

        recon_loss = F.mse_loss(x, r)

        y_kl_div = self.kl_div(zy, y_z, y, type='y')
        d_kl_div = self.kl_div(zd, d_z, d, type='d')
        x_kl_div = self.kl_div(zx, x_z, None)

        kl_div = y_kl_div + d_kl_div + x_kl_div

        class_loss, class_acc = self.classification_information(y, yhat)
        domain_loss, domain_acc = self.classification_information(d, dhat)

        loss = self.recon_pen * recon_loss + self.alpha_y * class_loss + self.alpha_d * domain_loss - self.beta * kl_div

        return r, loss, recon_loss, kl_div, class_loss, domain_loss, class_acc, domain_acc

    def training_step(self, batch, batch_idx):
        recon, loss, recon_loss, kl_div, class_loss, domain_loss, class_acc, domain_acc = self.loss_values(batch)

        # Log loss and metric
        self.log('train/loss', loss)
        self.log('train/recon_loss', recon_loss)
        self.log('train/kl_div', kl_div)
        self.log('train/class_loss', class_loss)
        self.log('train/domain_loss', domain_loss)
        # self.log('train/acc', class_acc)
        # self.log('train/domain_acc', domain_acc)

        return loss
    
    def validation_step(self, batch, batch_idx):
        recon, loss, recon_loss, kl_div, class_loss, domain_loss, class_acc, domain_acc = self.loss_values(batch)

        # Log loss and metric
        self.log('val/loss', loss)
        self.log('val/recon_loss', recon_loss)
        self.log('val/kl_div', kl_div)
        self.log('val/class_loss', class_loss)
        self.log('val/domain_loss', domain_loss)
        # self.log('val/class_acc', class_acc)
        # self.log('val/domain_acc', domain_acc)
        
        if batch_idx == 0:
            recon = wandb.Image(recon[0], caption="Reconstruction")
            image = wandb.Image(batch[0][0], caption="Original")
            self.logger.experiment.log({"visualisations": {"reconstruction": recon, "original": image}})

        return recon
    
    def test_step(self, batch, batch_idx):
        x, l = batch
        y, d = l
        
        r, yhat, dhat, zy, y_z, zd, d_z, zx, x_z = self.forward(x, y, d)

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
            y_true=gt,
            class_names=DOMAIN_LABELS
        )

        self._d = None
        self._dhat = None

        self.logger.experiment.log({'domain_scores': cm})


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if batch_idx == 0:
            num_images = 16
            x, l = batch
            y, d = l

            batch_size = x.shape[0]

            table = wandb.Table(columns=['Original'] + CLASS_LABELS)

            r, yhat, dhat, zy, y_z, zd, d_z, zx, x_z = self.forward(x, y, d)

            mu, log_var = self.prior_y(torch.tensor([i for i in range(self.y_dim)]).cuda())
            print(mu.shape)

            mu = torch.stack(batch_size * [mu])
            print(mu.shape)

            mu = mu.permute(1, 0, 2)
            print(mu.shape)

            latents = (torch.cat((mu[i], d_z, x_z), dim=1) for i in range(self.y_dim))

            # Class only
            images = [self.px(latent) for latent in latents]

            for i in range(num_images):
                table.add_data(
                    wandb.Image(x[i], caption=f"{CLASS_LABELS[y[i].data]}"),
                    wandb.Image(images[0][i]),
                    wandb.Image(images[1][i]),
                    wandb.Image(images[2][i])
                )                      

            self.logger.experiment.log({'COVID Variance': table})

                
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def kl_div(self, qz, z_q, label, type=None):
        if type == 'y':
            mu, log_var = self.prior_y(label)
            pz = self.reparametrize(mu, log_var)
            
        elif type == 'd':
            mu, log_var = self.prior_d(label)
            pz = self.reparametrize(mu, log_var)
        else:
            z_p_loc, z_p_scale = torch.zeros(z_q.size()[0], self.zx_dim).cuda(), torch.ones(z_q.size()[0], self.zx_dim).cuda()
            pz = torch.distributions.Normal(z_p_loc, z_p_scale)
        kl_div = torch.sum(pz.log_prob(z_q) - qz.log_prob(z_q))

        return kl_div
    
    def classification_information(self, y, yhat):
        class_loss = F.cross_entropy(yhat, y)
        class_acc = []

        return class_loss, class_acc
    
    def reparametrize(self, mu, log_var):
        if torch.isnan(mu).any():
            print("mu is nan")
            print(mu)
        elif torch.isnan(log_var).any():
            print("log_var is nan")
            print(log_var)
        std = torch.exp(0.5 * log_var)
        qz = torch.distributions.Normal(mu, std)
        return qz
    
    
class Prior(nn.Module):
    def __init__(self, classes, latent_dim):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(classes, latent_dim, bias=False), nn.BatchNorm1d(latent_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(latent_dim, latent_dim))
        self.fc22 = nn.Sequential(nn.Linear(latent_dim, latent_dim))

        self.classes = classes

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        self.fc1[1].weight.data.fill_(1)
        self.fc1[1].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, c):
        x = F.one_hot(c, num_classes=self.classes).float()
        hidden = self.fc1(x)
        mu = self.fc21(hidden)
        log_var = self.fc22(hidden)

        return mu, log_var


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
        self.c5 = ConvBlock(64, 64, 3, 1, 1)
        # n x 64 x 4 x 4
        
        self.mu = nn.Linear(1024, latent_dim)
        self.log_var = nn.Linear(1024, latent_dim)
        
    def forward(self, x):
        z = self.c1(x)
        z = self.c2(z)
        z = self.c3(z)
        z = self.c4(z)
        z = self.c5(z)

        z = z.view(-1, 1024)
        mu = self.mu(z)
        log_var = self.log_var(z)
        return mu , log_var


# An encoder to reverse the process of the encoder
class Decoder(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 4096)
        self.d1 = DeConvBlock(64, 64, 3, 1, 1)
        self.d2 = DeConvBlock(64, 64, 3, 1, 1)
        self.d3 = DeConvBlock(64, 32, 3, 1, 1)
        self.d4 = DeConvBlock(32, 16, 3, 1, 1)
        self.d5 = DeConvBlock(16, 1, 3, 1, 1)
        
    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 64, 4, 4)
        z = self.d1(z)
        z = self.d2(z)
        z = self.d3(z)
        z = self.d4(z)
        z = self.d5(z)
        return z
