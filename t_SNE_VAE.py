from model import VAE, reparameterization_trick
import torch.nn as nn
import torch


class TSNE_VAE(nn.Module):
    def __init__(self, base_vae, latent_features):
        super(TSNE_VAE, self).__init__()
        self.base_vae = base_vae

        self.flatten_size = latent_features * 4 * 4;

        # Thêm lớp chiếu về 2D cho t-SNE
        self.fc_mu = nn.Linear(self.flatten_size, 2)
        self.fc_logvar = nn.Linear(self.flatten_size, 2)

        # Thêm lớp khôi phục từ 2D về lại kích thước ban đầu cho Decoder
        self.fc_decode = nn.Linear(2, self.flatten_size)

    def encode(self, x):
        x0 = self.base_vae.encoder.conv0(x)
        x1 = self.base_vae.encoder.resnet1(x0)
        x2 = self.base_vae.encoder.conv2(x1)
        x3 = self.base_vae.encoder.resnet2(x2)
        x4 = self.base_vae.encoder.conv4(x3)
        x5 = self.base_vae.encoder.resnet3(x4)
        x6 = self.base_vae.encoder.conv6(x5)

        mu_conv, log_var_conv = torch.chunk(x6, chunks=2, dim=1)

        # Ép phẳng và đưa về 2D
        mu_flat = torch.flatten(mu_conv, start_dim=1)
        logvar_flat = torch.flatten(log_var_conv, start_dim=1)

        mu_2d = self.fc_mu(mu_flat)
        logvar_2d = self.fc_logvar(logvar_flat)

        return mu_2d, logvar_2d
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = reparameterization_trick(mu, log_var)
        output = self.decode(z)
        return output, mu, log_var
    
    def freeze_encoder(self):
        for param in self.base_vae.encoder.parameters():
            param.requires_grad = False
        for param in self.fc_mu.parameters():
            param.requires_grad = False
        for param in self.fc_logvar.parameters():
            param.requires_grad = False
        print("Encoder đã được đóng băng.")

    