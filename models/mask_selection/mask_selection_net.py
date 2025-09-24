import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import os
import glob

from tqdm import tqdm


class MaskSelectionNet(nn.Module):
    """各パッチに対して候補マスクの選択確率を出力するモデル"""
    
    def __init__(self, num_candidates, patch_grid_size=32):
        super().__init__()
        self.num_candidates = num_candidates
        self.patch_grid_size = patch_grid_size  # 32x32のパッチグリッド
        
        # 特徴抽出器（ResNetベースのエンコーダー）
        self.feature_extractor = nn.Sequential(
            # 初期畳み込み層
            nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 256x256 -> 64x64
            
            # ResBlockライクな構造
            self._make_layer(64, 128, 2, stride=2),   # 64x64 -> 32x32
            self._make_layer(128, 256, 2, stride=1),  # 32x32を維持
            self._make_layer(256, 256, 2, stride=1),  # 32x32を維持
        )
        
        # パッチレベルの特徴を強化
        self.patch_enhancer = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # マスク選択ヘッド
        self.mask_selector = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_candidates, kernel_size=1),  # [B, num_candidates, 32, 32]
        )
        
        # 初期化
        self.apply(self._init_weights)
    
    def _make_layer(self, in_planes, planes, num_blocks, stride):
        """ResNetスタイルのブロック作成"""
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        """重み初期化"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, 16, 256, 256] - 入力ビデオフレーム
        
        Returns:
            mask_logits: [B, num_candidates, 32, 32] - 各パッチでの候補マスクの選択ログ確率
            mask_probs: [B, num_candidates, 32, 32] - 各パッチでの候補マスクの選択確率
        """
        # 特徴抽出
        features = self.feature_extractor(x)  # [B, 256, 32, 32]
        
        # パッチレベル特徴強化
        enhanced_features = self.patch_enhancer(features)  # [B, 256, 32, 32]
        
        # マスク選択ロジット
        mask_logits = self.mask_selector(enhanced_features)  # [B, num_candidates, 32, 32]
        
        # ソフトマックスで確率に変換（パッチごとに独立）
        mask_probs = F.softmax(mask_logits, dim=1)  # [B, num_candidates, 32, 32]
        
        return mask_logits, mask_probs

class BasicBlock(nn.Module):
    """ResNetの基本ブロック"""
    
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

