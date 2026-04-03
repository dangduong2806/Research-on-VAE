from torch.utils.data import DataLoader, random_split
# Xây dựng Wrapper cho Dataset
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

# Khởi tạo Dataset và Dataloader
full_train_dataset = MNISTDataset(train=True, download=False)
# Chia thành tập train và validate
total_size = len(full_train_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
print("=== PHÂN CHIA DỮ LIỆU ===")
print(f"Tổng số ảnh gốc:     {total_size}")
print(f"Số ảnh tập Train:    {len(train_subset)}")
print(f"Số ảnh tập Validate: {len(val_subset)}\n")  