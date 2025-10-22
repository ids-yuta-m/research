import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskSelectionNetDirect(nn.Module):
    """
    Direct mask selection network that predicts the next mask from:
    - Current used mask (what was used for compression)
    - Compressed image (the result of applying the mask)

    This simplifies the pipeline by removing the need for video reconstruction.
    """

    def __init__(self, num_candidates, patch_grid_size=32):
        """
        Args:
            num_candidates: Number of candidate masks to choose from
            patch_grid_size: Grid size for patch-wise mask selection (default: 32x32)
        """
        super().__init__()
        self.num_candidates = num_candidates
        self.patch_grid_size = patch_grid_size  # 32x32 patch grid

        # Encoder for compressed image
        # Input: [B, 1, 256, 256] - single compressed image
        self.image_encoder = nn.Sequential(
            # Initial convolution
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 256x256 -> 64x64

            # ResBlock-like structure
            self._make_layer(64, 128, 2, stride=2),   # 64x64 -> 32x32
            self._make_layer(128, 256, 2, stride=1),  # maintain 32x32
        )

        # Encoder for used mask
        # Input: [B, 16, 256, 256] - mask used for compression (16 temporal positions)
        self.mask_encoder = nn.Sequential(
            # Initial convolution
            nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 256x256 -> 64x64

            # ResBlock-like structure
            self._make_layer(64, 128, 2, stride=2),   # 64x64 -> 32x32
            self._make_layer(128, 256, 2, stride=1),  # maintain 32x32
        )

        # Fusion module to combine image and mask features
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 256 + 256 = 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Patch-level feature enhancement
        self.patch_enhancer = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Mask selection head
        # Output: [B, num_candidates, 32, 32] - selection logits for each patch
        self.mask_selector = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_candidates, kernel_size=1),
        )

        # Weight initialization
        self.apply(self._init_weights)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        """Create ResNet-style blocks"""
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        """Weight initialization"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, compressed_image, used_mask):
        """
        Args:
            compressed_image: [B, 1, 256, 256] - Compressed image from current capture
            used_mask: [B, 16, 256, 256] - Mask used for the current compression

        Returns:
            mask_logits: [B, num_candidates, 32, 32] - Selection logits for each patch
            mask_probs: [B, num_candidates, 32, 32] - Selection probabilities for each patch
        """
        # Encode compressed image
        image_features = self.image_encoder(compressed_image)  # [B, 256, 32, 32]

        # Encode used mask
        mask_features = self.mask_encoder(used_mask)  # [B, 256, 32, 32]

        # Fuse features
        combined_features = torch.cat([image_features, mask_features], dim=1)  # [B, 512, 32, 32]
        fused_features = self.fusion(combined_features)  # [B, 256, 32, 32]

        # Enhance patch-level features
        enhanced_features = self.patch_enhancer(fused_features)  # [B, 256, 32, 32]

        # Generate mask selection logits
        mask_logits = self.mask_selector(enhanced_features)  # [B, num_candidates, 32, 32]

        # Convert to probabilities with softmax (independent per patch)
        mask_probs = F.softmax(mask_logits, dim=1)  # [B, num_candidates, 32, 32]

        return mask_logits, mask_probs


class BasicBlock(nn.Module):
    """Basic ResNet block"""

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


class MaskSelectionNetDirectWithHistory(nn.Module):
    """
    Enhanced version with temporal history support.

    This version can optionally take multiple previous compressed images and masks
    to make more informed decisions about the next mask.
    """

    def __init__(self, num_candidates, patch_grid_size=32, history_length=3):
        """
        Args:
            num_candidates: Number of candidate masks to choose from
            patch_grid_size: Grid size for patch-wise mask selection (default: 32x32)
            history_length: Number of previous captures to consider (default: 3)
        """
        super().__init__()
        self.num_candidates = num_candidates
        self.patch_grid_size = patch_grid_size
        self.history_length = history_length

        # Encoder for compressed images (can handle multiple history frames)
        # Input: [B, history_length, 256, 256]
        self.image_encoder = nn.Sequential(
            nn.Conv2d(history_length, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 256x256 -> 64x64

            self._make_layer(64, 128, 2, stride=2),   # 64x64 -> 32x32
            self._make_layer(128, 256, 2, stride=1),  # maintain 32x32
        )

        # Encoder for used masks (can handle multiple history frames)
        # Input: [B, 16 * history_length, 256, 256]
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(16 * history_length, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 256x256 -> 64x64

            self._make_layer(64, 128, 2, stride=2),   # 64x64 -> 32x32
            self._make_layer(128, 256, 2, stride=1),  # maintain 32x32
        )

        # Fusion module
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Patch-level feature enhancement
        self.patch_enhancer = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Mask selection head
        self.mask_selector = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_candidates, kernel_size=1),
        )

        # Weight initialization
        self.apply(self._init_weights)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        """Create ResNet-style blocks"""
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        """Weight initialization"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, compressed_images, used_masks):
        """
        Args:
            compressed_images: [B, history_length, 256, 256] - Previous compressed images
            used_masks: [B, 16 * history_length, 256, 256] - Previous masks used

        Returns:
            mask_logits: [B, num_candidates, 32, 32] - Selection logits for each patch
            mask_probs: [B, num_candidates, 32, 32] - Selection probabilities for each patch
        """
        # Encode compressed images
        image_features = self.image_encoder(compressed_images)  # [B, 256, 32, 32]

        # Encode used masks
        mask_features = self.mask_encoder(used_masks)  # [B, 256, 32, 32]

        # Fuse features
        combined_features = torch.cat([image_features, mask_features], dim=1)  # [B, 512, 32, 32]
        fused_features = self.fusion(combined_features)  # [B, 256, 32, 32]

        # Enhance patch-level features
        enhanced_features = self.patch_enhancer(fused_features)  # [B, 256, 32, 32]

        # Generate mask selection logits
        mask_logits = self.mask_selector(enhanced_features)  # [B, num_candidates, 32, 32]

        # Convert to probabilities
        mask_probs = F.softmax(mask_logits, dim=1)  # [B, num_candidates, 32, 32]

        return mask_logits, mask_probs
