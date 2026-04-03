from utils import run_training, vae_loss_fn_ver1, vae_loss_fn_ver2, vae_loss_fn_ver3
import matplotlib.pyplot as plt
from model import VAE

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import yaml
import torch

# Đọc dữ liệu config
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Chia dữ liệu

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Device: ", device)

latent_features = config['latent_features'] # 32

# Khởi tạo model
model_vae = VAE(latent_features=latent_features)
# Huấn luyện với 3 hàm loss khác nhau
# loss 1 
model_1, history_1 = run_training(model=model_vae, train_loader=, val_loader=, config=config, device=device, loss_fn=vae_loss_fn_ver1)