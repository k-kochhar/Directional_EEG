import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        channel_attention = torch.sigmoid(avg_out + max_out).view(b, c, 1)
        return x * channel_attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn

class EEGAnglePredictionModel(nn.Module):
    def __init__(self, input_channels=64, sample_length=128, num_filters=32):
        super(EEGAnglePredictionModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.pool1 = nn.AvgPool1d(kernel_size=2)
        self.ca1 = ChannelAttention(num_filters)
        self.sa1 = SpatialAttention(kernel_size=7)

        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(num_filters*2)
        self.pool2 = nn.AvgPool1d(kernel_size=2)
        self.ca2 = ChannelAttention(num_filters*2)
        self.sa2 = SpatialAttention(kernel_size=5)

        self.conv3 = nn.Conv1d(num_filters*2, num_filters*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_filters*4)
        self.pool3 = nn.AvgPool1d(kernel_size=2)

        self.feature_size = self._get_feature_size(input_channels, sample_length)
        self.fc1 = nn.Linear(self.feature_size, 128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)

    def _get_feature_size(self, channels, length):
        dummy_input = torch.zeros(1, channels, length)
        x = self.pool1(self.sa1(self.ca1(F.leaky_relu(self.bn1(self.conv1(dummy_input))))))
        x = self.pool2(self.sa2(self.ca2(F.leaky_relu(self.bn2(self.conv2(x))))))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x))))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool1(self.sa1(self.ca1(F.leaky_relu(self.bn1(self.conv1(x))))))
        x = self.pool2(self.sa2(self.ca2(F.leaky_relu(self.bn2(self.conv2(x))))))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.drop1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)
