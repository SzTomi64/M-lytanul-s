import torch.nn as nn
import torch


class Level(nn.Module):
    def __init__(self, inch, och, dim):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inch)
        self.conv1 = nn.Conv2d(inch, och, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(och)
        self.conv2 = nn.Conv2d(och, och, 3, 1, 1)
        self.act = nn.ReLU()
        self.time_embedding = nn.Sequential(
            Sinusoidal_embedding(dim),
            nn.Linear(dim, och),
            nn.ReLU()
        )
    
    def __call__(self, x, t):
        return self.forward(x, t)
    
    def forward(self, x, t):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.act(x)

        embedding = self.time_embedding(t)
        embedding = embedding[(..., ) + (None, ) * 2]
        x = x+embedding

        x = self.bn2(x)
        x = self.conv2(x)
        x = self.act(x)

        return x

class Upconv(nn.Module):
    def __init__(self, inch, och):
        super().__init__()
        self.bn = nn.BatchNorm2d(inch)
        self.conv = nn.ConvTranspose2d(inch, och, 2, 2)
        self.act = nn.ReLU()

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.act(x)

        return x
    
class Sinusoidal_embedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, t):
        return self.forward(t)
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim//2
        exp_sin = torch.linspace(0, 1, half_dim, device=device)
        exp_cos = torch.linspace(0, 1, half_dim, device=device)
        exp_sin = torch.sin(t*10000**exp_sin)
        exp_cos = torch.cos(t*10000**exp_cos)
        embedding = torch.cat((exp_sin, exp_cos), dim=-1)
        return embedding


class Unet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.downlevel1 = Level(3, 64, dim)
        self.downlevel2 = Level(64, 128, dim)
        self.downlevel3 = Level(128, 256, dim)
        self.downlevel4 = Level(256, 512, dim)
        self.downlevel5 = Level(512, 1024, dim)
        self.uplevel1 = Level(1024, 512, dim)
        self.uplevel2 = Level(512, 256, dim)
        self.uplevel3 = Level(256, 128, dim)
        self.uplevel4 = Level(128, 64, dim)
        self.pool = nn.MaxPool2d(2, 2)
        self.upconv1 = Upconv(1024, 512)
        self.upconv2 = Upconv(512, 256)
        self.upconv3 = Upconv(256, 128)
        self.upconv4 = Upconv(128, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.lastconv = nn.Conv2d(64, 3, 1)
        self.bn2 = nn.BatchNorm2d(3)
        self.act = nn.Sigmoid()
    
    def __call__(self, x, t):
        return self.forward(x, t)
    
    def forward(self, x, t):
        x = self.downlevel1(x, t)
        residual1 = x
        x = self.pool(x)
        
        x = self.downlevel2(x, t)
        residual2 = x
        x = self.pool(x)

        x = self.downlevel3(x, t)
        residual3 = x
        x = self.pool(x)

        x = self.downlevel4(x, t)
        residual4 = x
        x = self.pool(x)

        x = self.downlevel5(x, t)

        x = self.upconv1(x)
        x = torch.cat([residual4, x], 1)
        x = self.uplevel1(x, t)

        x = self.upconv2(x)
        x = torch.cat([residual3, x], 1)
        x = self.uplevel2(x, t)
        
        x = self.upconv3(x)
        x = torch.cat([residual2, x], 1)
        x = self.uplevel3(x, t)

        x = self.upconv4(x)
        x = torch.cat([residual1, x], 1)
        x = self.uplevel4(x, t)

        x = self.bn1(x)
        x = self.lastconv(x)


        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.avg_pool(x).view(B, C)  # B, C, 1, 1 -> B, C
        y = self.fc(y).view(B, C, 1, 1)  # B, C -> B, C, 1, 1
        return x * y.expand_as(x)  # Element-wise multiplication

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling and Max Pooling along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # B, 1, H, W
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # B, 1, H, W
        # Concatenate and pass through convolution
        y = torch.cat([avg_out, max_out], dim=1)  # B, 2, H, W
        y = self.conv(y)
        y = (self.sigmoid(y)+1)/2
        return x * y  # Apply spatial attention


class AttentionLevel(Level):
    def __init__(self, inch, och, dim, heads=8):
        super().__init__(inch, och, dim)
        self.channel_attention = ChannelAttention(och)
        self.spatial_attention = SpatialAttention()

    def forward(self, x, t):
        x = super().forward(x, t)
        x = self.channel_attention(x)  # Apply channel attention
        x = self.spatial_attention(x)  # Apply spatial attention
        return x


class AttentionUnet(Unet):
    def __init__(self, dim,):
        super().__init__(dim)
        self.downlevel1 = AttentionLevel(3, 64, dim)
        self.downlevel2 = AttentionLevel(64, 128, dim)
        self.downlevel3 = AttentionLevel(128, 256, dim)
        self.downlevel4 = AttentionLevel(256, 512, dim)
        self.downlevel5 = AttentionLevel(512, 1024, dim)
        self.uplevel1 = AttentionLevel(1024, 512, dim)
        self.uplevel2 = AttentionLevel(512, 256, dim)
        self.uplevel3 = AttentionLevel(256, 128, dim)
        self.uplevel4 = AttentionLevel(128, 64, dim)