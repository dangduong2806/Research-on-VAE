from torch.utils.data import DataLoader, random_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

# MNIST
class MNISTDataset(Dataset):
    def __init__(self, root="./data", train=True, download=False):
        # Đổi kích thước lên 32*32 và chuyển thành Tensor [0, 1]
        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor()
        ])
        self.dataset = datasets.MNIST(root=root, train=train, download=download, transform=self.transform)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, label = self.dataset[index]
        return {'image': img, 'label': label}
    

## CIFAR-10
class CIFARDataset(Dataset):
    def __init__(self, root="./data", train=True, download=False):
        self.transform = transforms.Compose([
            transforms.Resize(32), # CIFAR-10 vốn đã 32x32, nhưng giữ lại để đồng bộ
            transforms.ToTensor()
        ])
        # Gọi dataset CIFAR10 thay vì MNIST
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download, transform=self.transform)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, label = self.dataset[index]
        return {'image': img, 'label': label}

# Tạo hàm hỗ trợ chọn Dataset
def get_dataset(config):
    dataset_name = config.get('dataset', 'MNIST')
    if dataset_name == 'CIFAR10':
        print("Đang tải dữ liệu CIFAR-10...")
        return CIFARDataset(train=True, download=True)
    else:
        print("Đang tải dữ liệu MNIST...")
        return MNISTDataset(train=True, download=True)
    

def data_split(full_train_dataset, config):
    # Chia thành tập train và validate
    total_size = len(full_train_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
    print("=== PHÂN CHIA DỮ LIỆU ===")
    print(f"Tổng số ảnh gốc:     {total_size}")
    print(f"Số ảnh tập Train:    {len(train_subset)}")
    print(f"Số ảnh tập Validate: {len(val_subset)}\n") 
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True) # Shuffle tập train
    val_loader = DataLoader(dataset=val_subset, batch_size=config['batch_size'], shuffle=False) # Ko shuffle tập val 

    return train_loader, val_loader


### T-SNE
from sklearn.manifold import TSNE
import numpy as np
import torch

def create_tsne_dataloader(train_loader, batch_size, device):
    print("\n--- ĐANG CHUẨN BỊ DỮ LIỆU t-SNE ---")
    all_images = []
    
    # Rút toàn bộ ảnh từ dataloader hiện tại
    for batch in train_loader:
        all_images.append(batch['image'])
        
    X_tensor = torch.cat(all_images, dim=0)
    # Chuyển ảnh thành vector 1D để chạy thuật toán t-SNE gốc
    X_flat = X_tensor.view(X_tensor.size(0), -1).numpy()
    
    print(f"Bắt đầu chạy thuật toán t-SNE trên {X_flat.shape[0]} mẫu dữ liệu. Quá trình này có thể mất vài phút...")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_reduced_data = tsne.fit_transform(X_flat)
    print("Hoàn tất t-SNE!")
    
    # Scale dữ liệu (chia 40) theo như công thức bạn đã cung cấp
    tsne_target = (torch.tensor(tsne_reduced_data, dtype=torch.float32) / 40.0)
    
    # Tạo DataLoader mới chứa cặp (Ảnh gốc, Tọa độ t-SNE)
    tsne_dataset = TensorDataset(X_tensor, tsne_target)
    tsne_loader = DataLoader(tsne_dataset, batch_size=batch_size, shuffle=True)
    
    return tsne_loader