from utils import run_training, vae_loss_fn_ver1, vae_loss_fn_ver2, vae_loss_fn_ver3
import matplotlib.pyplot as plt

from model import VAE
from data import MNISTDataset, CIFARDataset, data_split, create_tsne_dataloader

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import yaml
import torch
import torch.optim as optim

from utils import train_tsne_decoder, train_tsne_encoder, visualize_tsne_latent_space, compare_constructions
from t_SNE_VAE import TSNE_VAE 
from cifar_model import CIFAR_VAE

# Đọc dữ liệu config
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Chia dữ liệu
full_train_dataset = CIFARDataset(train=True, download=False)
train_loader, val_loader = data_split(full_train_dataset=full_train_dataset, config=config)

print("\n")

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Device: ", device)

latent_features = config['latent_features'] # 32
in_channels = config['in_channels']

data_set = "CIFAR10"

# Huấn luyện với 3 hàm loss khác nhau
# loss 1 
model_vae_1 = CIFAR_VAE(latent_features=latent_features, in_channels=in_channels)
name_loss_1 = "loss_fn_1"
model_1, history_1 = run_training(model=model_vae_1, train_loader=train_loader, val_loader=val_loader, config=config, device=device, dataset=data_set, name_loss=name_loss_1, loss_fn=vae_loss_fn_ver1)
print("\n")
# Loss 2
model_vae_2 = CIFAR_VAE(latent_features=latent_features, in_channels=in_channels)
name_loss_2 = "loss_fn_2"
model_2, history_2 = run_training(model=model_vae_2, train_loader=train_loader, val_loader=val_loader, config=config, device=device,dataset=data_set, name_loss=name_loss_2, loss_fn=vae_loss_fn_ver2)
print("\n")
# Loss 3
model_vae_3 = CIFAR_VAE(latent_features=latent_features, in_channels=in_channels)
name_loss_3 = "loss_fn_3"
model_3, history_3 = run_training(model=model_vae_3, train_loader=train_loader, val_loader=val_loader, config=config, device=device,dataset=data_set, name_loss=name_loss_3, loss_fn=vae_loss_fn_ver3)
print("\n")

epochs_range_1 = range(1, len(history_1['train_loss']) + 1)
epochs_range_2 = range(1, len(history_2['train_loss']) + 1)
epochs_range_3 = range(1, len(history_3['train_loss']) + 1)

plt.figure(figsize=(12, 6))

# Vẽ đồ thị Train Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range_1, history_1['train_loss'], label='Train Loss MAE', color='blue', linestyle='-')
plt.plot(epochs_range_2, history_2['train_loss'], label='Train Loss MAE and SSIM', color='red', linestyle='-')
plt.plot(epochs_range_3, history_3['train_loss'], label='Train Loss MSE and SSIM', color = 'green', linestyle='-')
plt.title('So sánh Train Loss trên tập CIFAR-10')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Vẽ đồ thị Validation Loss
plt.subplot(1, 2, 2)
val_range_1 = range(1, len(history_1['val_loss']) + 1)
val_range_2 = range(1, len(history_2['val_loss']) + 1)
val_range_3 = range(1, len(history_3['val_loss']) + 1)

plt.plot(val_range_1, history_1['val_loss'], label='Val Loss MAE', color='blue', linestyle='-')
plt.plot(val_range_2, history_2['val_loss'], label='Val Loss MAE and SSIM', color='red', linestyle='-')
plt.plot(val_range_3, history_3['val_loss'], label='Val Loss MSE and SSIM', color = 'green', linestyle='-')
plt.title('So sánh Validation Loss trên tập CIFAR-10')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.show()

# Lưu ảnh:
plt.savefig('loss_comparison_CIFAR.png', dpi=300, bbox_inches='tight')
print("\nĐã lưu biểu đồ thành công vào file 'loss_comparison_CIFAR_10.png'")

### TSNE
# base_vae = VAE(latent_features=latent_features)
# tsne_model = TSNE_VAE(base_vae=base_vae, latent_features=latent_features)

# # Dataloader
# tsne_loader = create_tsne_dataloader(train_loader=train_loader, batch_size=config['batch_size'], device=device)

# # Huấn luyện Encoder
# encoder_optimzer = optim.Adam(tsne_model.parameters(), lr=config['learning_rate'])
# train_tsne_encoder(
#     model=tsne_model,
#     dataloader=tsne_loader,
#     optimizer=encoder_optimzer,
#     epochs=10,
#     device=device
# )

# # Đóng băng Encoder sau khi train xong
# tsne_model.freeze_encoder()

# # Huấn luyện decoder
# decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, tsne_model.parameters()), lr=config['learning_rate'])
# train_tsne_decoder(
#     model=tsne_model,
#     dataloader=tsne_loader,
#     optimizer=decoder_optimizer,
#     epochs=10,
#     device=device
# )

# print("\nHoàn tất quá trình huấn luyện t-SNE VAE!")

# # Trực quan không gian tiềm ẩn 
# visualize_tsne_latent_space(
#     model=tsne_model,
#     dataloader=val_loader,
#     device=device,
#     num_samples=config['num_samples']
# )
# # So sánh kết quả tái tạo ảnh gốc
# print("\n=== SO SÁNH CHẤT LƯỢNG ẢNH TÁI TẠO ===")
# # Bạn có thể dùng model_1, model_2, hoặc model_3 làm đại diện cho Vanilla VAE
# compare_constructions(
#     tsne_model=tsne_model,
#     dataloader=val_loader,
#     device=device,
#     n_images=15 
# )