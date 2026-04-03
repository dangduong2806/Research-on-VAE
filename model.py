import torch 
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, channels: int):
        # Khởi tạo khối ResNet với channels: là số chiều của ảnh đầu vào
        super(ResNetBlock, self).__init__()
        self.channels = channels
        self.num_groups = 8
        # Khởi tạo cho khối thứ 1
        self.norm1 = nn.GroupNorm(num_groups=self.num_groups, num_channels=self.channels)
        self.act1 = nn.SiLU()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3,
                               stride=1, padding=1)
        # Khởi tạo cho khối thứ 2
        self.norm2 = nn.GroupNorm(num_groups=self.num_groups, num_channels=self.channels)
        self.act2 = nn.SiLU()

        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, 
                               stride=1, padding=1)

        
    def forward(self, x):
        residual = x
        # Khối 1
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        # Khối 2
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)

        return x + residual
    
# Cài đặt Encoder
class Encoder(nn.Module):
    def __init__(self, latent_features: int):
        super(Encoder, self).__init__()
        self.latent_features = latent_features
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.resnet1 = ResNetBlock(channels=32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resnet2 = ResNetBlock(channels=64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.resnet3 = ResNetBlock(channels=128)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels= self.latent_features * 2, kernel_size=3, stride=2, padding = 1)
        
    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.resnet1(x0)

        x2 = self.conv2(x1)
        x3 = self.resnet2(x2)

        x4 = self.conv4(x3)
        x5 = self.resnet3(x4)
        
        x6 = self.conv6(x5)
        # Sử dụng torch.chunk để chia đôi tensor.
        muy, log_var = torch.chunk(x6, chunks=2, dim= 1)
        # Giải thích: mu và log_var sẽ có shape là (B, latent_features, 32, 32)
        return muy, log_var
    
# Latent space and reparameterization trick
def reparameterization_trick(mu: torch.Tensor, log_var: torch.Tensor):
    # 1. Lấy độ lệch chuẩn
    stdev = torch.exp(0.5 * log_var)

    # 2. Lấy mẫu nhiễu epsilon từ phân phối chuẩn tắc N(0, 1)
    # Hàm torch.randn_like cực kỳ tiện lợi: nó tự động tạo ra một tensor chứa nhiễu
    # có cùng kích thước (shape), cùng kiểu dữ liệu (dtype) và 
    # nằm trên cùng thiết bị (CPU hay GPU) với tensor `mu`.
    epsilon = torch.randn_like(mu)

    z = mu + stdev * epsilon

    return z

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_features: int):
        super(Decoder, self).__init__()
        self.latent_features = latent_features
        self.sigmoid = nn.Sigmoid()

        self.conv0 = nn.ConvTranspose2d(in_channels=self.latent_features, out_channels=128, kernel_size=4, stride=2,
                                        padding=1)
        self.resnet1 = ResNetBlock(channels=128)

        self.conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2,
                                        padding=1)
        self.resnet2 = ResNetBlock(channels=64)

        self.conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2,
                                        padding=1)
        self.resnet3 = ResNetBlock(channels=32)

        self.conv6 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, 
                                        padding=1)
        
        
    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.resnet1(x0)

        x2 = self.conv2(x1)
        x3 = self.resnet2(x2)

        x4 = self.conv4(x3)
        x5 = self.resnet3(x4)
        
        logits = self.conv6(x5) # Shape: (B, 3, 256, 256)
        output = self.sigmoid(logits) # Ép giá trị về [0, 1]

        return output
    
# VAE class
class VAE(nn.Module):
    def __init__(self, latent_features: int):
        super(VAE, self).__init__()
        # Khởi tạo encoder và decoder
        self.encoder = Encoder(latent_features=latent_features)
        self.decoder = Decoder(latent_features=latent_features)
    
    def forward(self, x):
        # Encoder
        mu, log_var = self.encoder(x)
        # Reparameterization trick
        z = reparameterization_trick(mu=mu, log_var=log_var)
        # Decoder
        output = self.decoder(z)
        return output, mu, log_var
    
