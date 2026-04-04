# file: utils.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_msssim import ssim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Hàm loss
def vae_loss_fn_ver1(model, batch, lambda_rec = 1.0, lambda_kl = 1.0, lambda_ssim=0.84):
    # Chạy model
    output, mu, log_var = model(batch)
    # Reconstruction loss: Mean Absolute Loss
    reconstruction_loss = F.l1_loss(output, batch, reduction='mean')
    # KL Loss
    kl_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1, dim=(1, 2, 3))
    kl_loss = torch.mean(kl_loss) # Lấy trung bình trên toàn batch
    kl_loss = kl_loss / (4 * 4 * 4)

    # SSIM Loss
    ssim_loss = 0.0

    # Tổng hợp Loss
    total_loss = lambda_rec * reconstruction_loss + lambda_kl * kl_loss + lambda_ssim * ssim_loss
    
    return total_loss

# Hàm loss
def vae_loss_fn_ver2(model, batch, lambda_rec = 1.0, lambda_kl = 1.0, lambda_ssim=0.84):
    # Chạy model
    output, mu, log_var = model(batch)
    # Reconstruction loss: Mean Absolute Loss
    reconstruction_loss = F.l1_loss(output, batch, reduction='mean')
    # KL Loss
    kl_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1, dim=(1, 2, 3))
    kl_loss = torch.mean(kl_loss) # Lấy trung bình trên toàn batch
    kl_loss = kl_loss / (4 * 4 * 4)

    # SSIM Loss
    ssim_val = ssim(output, batch, data_range=1.0)
    ssim_loss = 1.0 - ssim_val.mean()
    # Tổng hợp Loss
    total_loss = lambda_rec * reconstruction_loss + lambda_kl * kl_loss + lambda_ssim * ssim_loss
    
    return total_loss

# Hàm loss
def vae_loss_fn_ver3(model, batch, lambda_rec = 1.0, lambda_kl = 1.0, lambda_ssim=0.84):
    # Chạy model
    output, mu, log_var = model(batch)
    # Reconstruction loss: Mean Squared Loss
    reconstruction_loss = F.mse_loss(output, batch, reduction='mean')
    # KL Loss
    kl_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1, dim=(1, 2, 3))
    kl_loss = torch.mean(kl_loss) # Lấy trung bình trên toàn batch
    kl_loss = kl_loss / (4 * 4 * 4)

    # SSIM Loss
    ssim_val = ssim(output, batch, data_range=1.0)
    ssim_loss = 1.0 - ssim_val.mean()
    # Tổng hợp Loss
    total_loss = lambda_rec * reconstruction_loss + lambda_kl * kl_loss + lambda_ssim * ssim_loss
    
    return total_loss


# Vòng lặp huấn luyện
def run_training(model, train_loader, val_loader, config, device, loss_fn = vae_loss_fn_ver1):
    model.train()
    model.to(device)

    # Khởi tạo optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # Dictinary để lưu trung bình loss
    history = {
        "train_loss": [],
        "val_loss": []
    }

    # Vòng lặp epoch
    for epoch_idx in range(config['num_epochs']):
        
        model.train()

        epoch_losses = []

        # tqdm để tạo thanh tiến trình 
        progess_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch_idx+1}/{config['num_epochs']}")

        # Vòng lặp batch
        for i, batch in progess_bar:
            x = batch['image'].to(device)
            # Xóa sạch đạo hàm
            optimizer.zero_grad()
            # Tính loss
            loss = loss_fn(
                model=model,
                batch=x,
                lambda_rec=config['lambda_rec'],
                lambda_kl=config['lambda_kl'],
                lambda_ssim=config['lambda_ssim']
            )
            # Backpropagation
            loss.backward()

            # Cập nhật trọng số
            optimizer.step()

            current_loss = loss.item()
            epoch_losses.append(current_loss)

            # Cập nhật thanh tiến trình
            progess_bar.set_postfix(loss=f"{current_loss:.4f}")
        
        # Tính trung bình loss sau mỗi epoch
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        # print(f"Epoch {epoch_idx+1} | Train Loss trung bình: {avg_epoch_loss:.4f}\n")
        # Lưu vào history
        history['train_loss'].append(avg_epoch_loss)

        # Validate model
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device=device)
                # Tính loss
                val_loss = loss_fn(
                    model=model,
                    batch=images,
                    lambda_rec=config['lambda_rec'],
                    lambda_kl=config['lambda_kl'],
                    lambda_ssim=config['lambda_ssim']
                ) 
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / len(val_loader)
        # print(f"Epoch {epoch_idx+1} | Val Loss trung bình: {average_val_loss:.4f}\n")
        # Lưu vào history
        history['val_loss'].append(average_val_loss)

        print(f"Epoch {epoch_idx+1} | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {average_val_loss:.4f}\n")

    print("Huấn luyện xong")
    return model, history


###### T-SNE VAE 
def train_tsne_encoder(model, dataloader, optimizer, epochs, device):
    print("\n[Giai đoạn 1] Huấn luyện t-SNE Encoder...")
    model.train()
    model.to(device)

    for epoch in range(epochs):
        overall_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Encoder Epoch {epoch+1}/{epochs}")
        
        for i, (images, targets) in progress_bar:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            mu, _ = model.encode(images) # Chỉ lấy output dự đoán 2D
            
            # Loss: Mean Squared Error so với tọa độ t-SNE chuẩn
            loss = F.mse_loss(mu, targets)
            loss.backward()
            optimizer.step()
            
            overall_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.5f}")

def train_tsne_decoder(model, dataloader, optimizer, epochs, device):
    print("\n[Giai đoạn 2] Huấn luyện t-SNE Decoder...")
    model.train()
    model.to(device)

    for epoch in range(epochs):
        overall_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Decoder Epoch {epoch+1}/{epochs}")
        
        for i, (images, _) in progress_bar:
            images = images.to(device)
            
            optimizer.zero_grad()
            # Bước 1: Lấy đặc trưng 2D từ Encoder (đã bị đóng băng)
            with torch.no_grad():
                mu, _ = model.encode(images)
            
            # Bước 2: Khôi phục ảnh từ tọa độ 2D
            prediction = model.decode(mu)
            
            # Loss: MSE/BCE Loss cho quá trình tái tạo ảnh (Reconstruction)
            loss = F.mse_loss(prediction, images, reduction='mean')
            
            loss.backward()
            optimizer.step()
            
            overall_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.5f}")

        
def visualize_tsne_latent_space(model, dataloader, device, num_samples=10000):
    print("\nĐang trích xuất không gian tiềm ẩn để vẽ đồ thị...")
    model.eval() # Chuyển model sang chế độ đánh giá
    all_mu = []
    all_labels = []

    with torch.no_grad(): # Không tính gradient để tiết kiệm bộ nhớ
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            
            # Chỉ sử dụng phần encode để lấy tọa độ 2D (mu)
            mu, _ = model.encode(images)
            
            all_mu.append(mu.cpu().numpy())
            all_labels.append(labels)
            
            # Dừng lại nếu đã đủ số lượng mẫu (giúp biểu đồ không bị quá tải)
            if sum(len(l) for l in all_labels) >= num_samples:
                break
    
    # Gộp tất cả các batch lại
    all_mu = np.concatenate(all_mu, axis=0)[:num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:num_samples]
    
    # --- Bắt đầu vẽ biểu đồ ---
    plt.figure(figsize=(10, 8))
    
    # Vẽ scatter plot: x = mu[:, 0], y = mu[:, 1], màu sắc c = labels
    # Sử dụng colormap 'tab10' rất phù hợp cho 10 class (chữ số MNIST)
    scatter = plt.scatter(all_mu[:, 0], all_mu[:, 1], c=all_labels, cmap='tab10', alpha=0.7, s=15)
    
    # Thêm thanh chú thích màu sắc
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label('Nhãn dữ liệu (Labels)')
    
    # Trang trí biểu đồ
    plt.title('Không gian tiềm ẩn (Latent Space) 2D từ t-SNE VAE', fontsize=14, fontweight='bold')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def compare_constructions(tsne_model, dataloader, device, n_images=10):
    print("\nĐang tạo ảnh so sánh...")
    
    # Lấy 1 batch dữ liệu từ dataloader
    batch = next(iter(dataloader))
    
    # Lấy đủ số lượng ảnh cho 2 hàng (nrows=2)
    images = batch['image'].to(device)[:n_images*2] 
    
    # 1. Chuẩn bị Ảnh Gốc (Ground truth)
    # Loại bỏ kênh màu (channel) để vẽ: (B, 1, 32, 32) -> (B, 32, 32)
    ground_truth = images.cpu().squeeze(1).numpy()

    # 3. Chuẩn bị Ảnh từ t-SNE VAE
    tsne_model.eval()
    with torch.no_grad():
        # Gọi thẳng tsne_model(images) nó sẽ trả về (output, mu, logvar) theo class TSNE_VAE
        preds_tsne, _, _ = tsne_model(images)
        preds_tsne = preds_tsne.cpu().squeeze(1).numpy()

    # Hàm hỗ trợ vẽ lưới ảnh
    def plot_images(axes, images_to_plot, title, nrows=2, ncols=n_images):
        for i in range(nrows):
            for j in range(ncols):
                idx = i * ncols + j
                ax = axes[i, j]
                if idx < len(images_to_plot):
                    ax.imshow(images_to_plot[idx], cmap='Greys')
                ax.axis('off')
    
    # --- Vẽ biểu đồ ---
    # Vẽ Ground Truth
    fig1, axes1 = plt.subplots(2, n_images, figsize=(15, 3))
    fig1.suptitle('Ảnh gốc (Ground Truths)', fontsize=16)
    plot_images(axes1, ground_truth, 'Ảnh gốc (Ground Truths)')
    plt.show()

    # Vẽ t-SNE VAE
    fig3, axes3 = plt.subplots(2, n_images, figsize=(15, 3))
    fig3.suptitle('Tái tạo từ t-SNE VAE', fontsize=16)
    plot_images(axes3, preds_tsne, 'Tái tạo từ t-SNE VAE')
    plt.show()