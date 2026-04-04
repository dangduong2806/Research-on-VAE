# file: utils.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_msssim import ssim
from tqdm import tqdm

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