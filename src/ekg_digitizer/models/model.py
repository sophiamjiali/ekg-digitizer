# ==============================================================================
# File Name : model.py
# Purpose   : Defines the model architecture and utilities
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ekg_digitizer.config import Config


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        
        # Channel attention
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv_spatial(spatial_att))
        x = x * spatial_att
        
        return x


class ResidualBlock(nn.Module):
    """Enhanced residual block with CBAM"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


class MultiScaleFusion(nn.Module):
    """Multi-scale feature fusion"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, 1)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        fused = torch.cat([b1, b2, b3], dim=1)
        
        return self.fusion(fused)


class AdvancedTCN(nn.Module):
    """Advanced Temporal Convolutional Network"""
    def __init__(self, input_dim, output_length, num_channels=[512, 512, 384, 256]):
        super().__init__()
        
        layers = []
        in_channels = input_dim
        
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ))
            in_channels = out_channels
        
        self.tcn = nn.Sequential(*layers)
        
        self.upsampler = nn.Sequential(
            nn.Conv1d(num_channels[-1], num_channels[-1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels[-1], 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, kernel_size=1)
        )
        
        self.output_length = output_length
    
    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(-1)
        x = F.interpolate(x, size=self.output_length, mode='linear', align_corners=False)
        x = self.tcn(x)
        x = self.upsampler(x)
        
        return x.squeeze(1)


class EnhancedEKGNet(nn.Module):
    """Enhanced EKG reconstruction network"""
    def __init__(self, config: Config):
        super().__init__()
        
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(config.img_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Encoder layers with increasing channels
        self.layer1 = self._make_layer(64, 128, 3, stride=1)
        self.layer2 = self._make_layer(128, 256, 3, stride=2)
        self.layer3 = self._make_layer(256, 512, 4, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)
        
        # Multi-scale fusion
        self.fusion = MultiScaleFusion(512, 512)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        
        # Lead-specific decoders
        self.lead_decoders = nn.ModuleDict()
        
        for lead in ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                     'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
            target_len = int(config.base_fs * config.lead_ii_duration) if lead == 'II' else int(config.base_fs * config.other_leads_duration)
            self.lead_decoders[lead] = AdvancedTCN(512, target_len)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fusion(x)
        
        features = self.global_pool(x)
        features = features.view(features.size(0), -1)
        features = self.dropout(features)
        
        outputs = {}
        
        for lead_name, decoder in self.lead_decoders.items():
            outputs[lead_name] = decoder(features)
        
        return outputs