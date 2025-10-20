import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import os
import glob
from pathlib import Path
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# CUDA設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# パス設定
MASK_DIR = "/home/yuta-m/research-github/data/masks/candidates/128masks"
GT_DIR = "/home/yuta-m/research-github/data/raw/256x256/gt_gray"  # 256x256データから切り取り

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x

class MaskAdaptiveUNet_8x8(nn.Module):
    """8x8画像用の軽量UNet - 1層構造"""
    def __init__(self, in_ch, out_ch):
        super(MaskAdaptiveUNet_8x8, self).__init__()
        
        # 8x8は非常に小さいため、ダウンサンプリングなしの単純な構造
        self.conv1 = double_conv(in_ch, 32)
        self.conv2 = double_conv(32, 32)
        self.conv_last = nn.Conv2d(32, out_ch, 1)
        self.afn_last = nn.Tanh()

    def forward(self, x):
        inputs = x
        
        x = self.conv1(x)      # [B, 32, 8, 8]
        x = self.conv2(x)      # [B, 32, 8, 8]
        x = self.conv_last(x)  # [B, out_ch, 8, 8]
        x = self.afn_last(x)
        
        # 残差接続
        if inputs.shape[1] == x.shape[1]:
            out = x + inputs
        else:
            out = x
        
        return out

class UniversalADMM_net_8x8(nn.Module):
    def __init__(self, t=16, s=8, patch_size=8):
        super().__init__()
        self.t = t
        self.s = s
        self.patch_size = patch_size
        
        # マスクエンコーダ（簡略化）
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(t, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        
        # 軽量UNet（9段階）
        self.unet_stages = nn.ModuleList([
            MaskAdaptiveUNet_8x8(t + 16, t) for _ in range(9)
        ])
        
        # 適応的ガンマパラメータ予測器（簡略化）
        self.gamma_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 9),
            nn.Softplus()
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        """重み初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        # マスク特徴抽出
        mask_feat = self.mask_encoder(mask.float())
        adaptive_gammas = self.gamma_predictor(mask_feat)
        
        # 測定生成
        y = torch.sum(x * mask, dim=1)  # [B, 8, 8]
        
        # ADMM反復初期化
        x_list = []
        theta = self.At(y, mask)
        b = torch.zeros_like(x)
        
        # 9段階のADMM反復
        for i, unet in enumerate(self.unet_stages):
            gamma = adaptive_gammas[:, i:i+1, None, None]  # [B, 1, 1, 1]
            
            # ADMMステップ
            yb = self.A(theta + b, mask)
            Phi_s = torch.sum(mask, dim=1, keepdim=True)  # [B, 1, 8, 8]
            
            Phi_s_safe = Phi_s + 1e-8
            gamma_expanded = gamma.expand_as(Phi_s_safe)
            
            residual = torch.div(y.unsqueeze(1) - yb.unsqueeze(1), Phi_s_safe + gamma_expanded)
            x = theta + b + self.At(residual.squeeze(1), mask)
            x1 = x - b
            
            # マスク情報を含む入力
            mask_aware_input = torch.cat([x1, mask_feat], dim=1)
            
            # UNet処理
            theta = unet(mask_aware_input) + x1
            b = b - (x - theta)
            x_list.append(theta)
        
        return x_list
    
    def A(self, x, mask):
        """Forward operator: x -> y"""
        temp = x * mask
        y = torch.sum(temp, dim=1)
        return y

    def At(self, y, mask):
        """Transpose operator: y -> x"""
        if y.dim() == 3:  # [B, H, W]
            temp = torch.unsqueeze(y, 1).repeat(1, mask.shape[1], 1, 1)
        else:
            temp = y.repeat(1, mask.shape[1], 1, 1) if y.shape[1] == 1 else y
        
        x = temp * mask
        return x

def load_candidate_masks(mask_dir):
    """候補マスクを読み込む"""
    mask_files = glob.glob(os.path.join(mask_dir, "*.mat"))
    candidate_masks = []
    
    print(f"Loading {len(mask_files)} candidate masks...")
    for mask_file in tqdm(mask_files):
        mask_data = sio.loadmat(mask_file)
        key = [k for k in mask_data.keys() if not k.startswith('_')][0]
        mask = mask_data[key]  # [16, 8, 8]
        
        mask = np.array(mask, dtype=np.float32)
        if mask.shape != (16, 8, 8):
            if mask.ndim == 3:
                if mask.shape == (8, 8, 16):
                    mask = mask.transpose(2, 0, 1)
                elif mask.shape == (8, 16, 8):
                    mask = mask.transpose(1, 0, 2)
            print(f"Warning: Mask shape was {mask_data[key].shape}, reshaped to {mask.shape}")
        
        candidate_masks.append(torch.tensor(mask, dtype=torch.float32))
    
    return candidate_masks

def create_8x8_mask(candidate_masks):
    """8x8用マスクを作成（1つのパッチマスクをそのまま使用）"""
    selected_mask = random.choice(candidate_masks)  # [16, 8, 8]
    return selected_mask

def load_ground_truth_data(gt_dir, max_files=None):
    """教師データを読み込む（256x256から8x8を切り取り）"""
    gt_files = glob.glob(os.path.join(gt_dir, "*.mat"))
    if max_files:
        gt_files = gt_files[:max_files]
    
    gt_data = []
    print(f"Loading {len(gt_files)} ground truth files...")
    for gt_file in tqdm(gt_files):
        data = sio.loadmat(gt_file)
        key = [k for k in data.keys() if not k.startswith('_')][0]
        video = data[key]  # [16, 256, 256]
        
        video = np.array(video, dtype=np.float32)
        if video.shape != (16, 256, 256):
            if video.ndim == 3:
                if video.shape == (256, 256, 16):
                    video = video.transpose(2, 0, 1)
                elif video.shape == (256, 16, 256):
                    video = video.transpose(1, 0, 2)
        
        # 正規化（0-1範囲）
        video = (video - video.min()) / (video.max() - video.min() + 1e-8)
        gt_data.append(torch.tensor(video, dtype=torch.float32))
    
    return gt_data

def extract_8x8_patch(video_256):
    """256x256動画からランダムに8x8パッチを切り取る
    
    Args:
        video_256: [16, 256, 256] のテンソル
    
    Returns:
        patch_8x8: [16, 8, 8] のテンソル
    """
    # ランダムな開始位置を選択（0～248の範囲）
    start_h = random.randint(0, 256 - 8)
    start_w = random.randint(0, 256 - 8)
    
    # 8x8パッチを切り取り
    patch = video_256[:, start_h:start_h+8, start_w:start_w+8]
    
    return patch

def calculate_psnr(img1, img2, mask=None):
    """PSNR計算"""
    if mask is not None:
        mse = torch.mean(((img1 - img2) * mask) ** 2)
        mask_ratio = torch.mean(mask)
        mse = mse / (mask_ratio + 1e-8)
    else:
        mse = torch.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def train_adaptive_admm_net_8x8():
    """8x8用適応的ADMM_netの学習"""
    # データ読み込み
    candidate_masks = load_candidate_masks(MASK_DIR)
    gt_data = load_ground_truth_data(GT_DIR)
    
    print(f"Loaded {len(candidate_masks)} candidate masks")
    print(f"Loaded {len(gt_data)} ground truth videos")
    
    if len(candidate_masks) > 0:
        print(f"Candidate mask shape: {candidate_masks[0].shape}")
        print(f"Candidate mask value range: {candidate_masks[0].min():.3f} - {candidate_masks[0].max():.3f}")
    if len(gt_data) > 0:
        print(f"GT data shape: {gt_data[0].shape}")
        print(f"GT data value range: {gt_data[0].min():.3f} - {gt_data[0].max():.3f}")
    
    # モデル初期化
    model = UniversalADMM_net_8x8(t=16, s=8, patch_size=8).to(device)
    
    # パラメータ数を表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 再構成ネットワークのみ最適化（マスク学習なし）
    recon_lr = 0.0001  # 8x8は小さいので学習率を少し高めに
    
    optimizer = torch.optim.Adam(model.parameters(), lr=recon_lr, weight_decay=1e-4)
    
    # 学習率スケジューラ
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    
    # MSE損失
    criterion = nn.MSELoss()
    
    # 学習ログ
    train_losses = []
    train_psnrs = []
    
    num_epochs = 20000
    batch_size = 8  # 8x8は小さいのでバッチサイズを増やす
    
    print("Starting training...")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0
        start_time = time.time()
        
        print(f"\rTraining Epoch: {epoch}/{num_epochs}", end="", flush=True)
        
        # バッチ処理
        for batch_start in range(0, len(gt_data), batch_size):
            batch_end = min(batch_start + batch_size, len(gt_data))
            batch_gt_256 = gt_data[batch_start:batch_end]
            
            # 256x256から8x8パッチを切り取り
            batch_gt_8x8 = []
            for video_256 in batch_gt_256:
                patch_8x8 = extract_8x8_patch(video_256)
                batch_gt_8x8.append(patch_8x8)
            
            # バッチをGPUに移動
            batch_gt = torch.stack(batch_gt_8x8).to(device)
            
            # 各サンプルに対してランダムマスクを生成
            batch_masks = []
            for _ in range(len(batch_gt)):
                mask = create_8x8_mask(candidate_masks)
                batch_masks.append(mask)
            batch_masks = torch.stack(batch_masks).to(device)
            
            # 勾配初期化
            optimizer.zero_grad()
            
            # 前向き計算
            reconstructed_list = model(batch_gt, batch_masks)
            
            # 多段階損失
            recon_loss = (torch.sqrt(criterion(reconstructed_list[-1], batch_gt)) + 
                         0.5 * torch.sqrt(criterion(reconstructed_list[-2], batch_gt)) + 
                         0.5 * torch.sqrt(criterion(reconstructed_list[-3], batch_gt)))
            
            total_loss = recon_loss
            
            # 逆伝播
            total_loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # オプティマイザ更新
            optimizer.step()
            
            # メトリクス計算
            with torch.no_grad():
                final_reconstruction = reconstructed_list[-1]
                batch_psnr = 0.0
                for i in range(len(batch_gt)):
                    psnr_val = 10 * torch.log10(1 / criterion(final_reconstruction[i], batch_gt[i]))
                    batch_psnr += psnr_val.item()
                batch_psnr /= len(batch_gt)
            
            epoch_loss += total_loss.item()
            epoch_psnr += batch_psnr
            num_batches += 1
        
        # エポック平均
        avg_loss = epoch_loss / num_batches
        avg_psnr = epoch_psnr / num_batches
        time_taken = time.time() - start_time
        
        train_losses.append(avg_loss)
        train_psnrs.append(avg_psnr)
        
        # 学習率更新
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # ログ出力
        if epoch % 50 == 0:
            print("\n====================================")
            print(f"Loss: {avg_loss:.6f}")
            print(f"PSNR: {avg_psnr:.2f} dB")
            print(f"Time: {time_taken:.2f}s")
            print(f"LR: {current_lr:.6f}")
        
        # モデル保存
        if epoch % 200 == 0:
            os.makedirs(f'./checkpoints/reconstruction/8x8', exist_ok=True)
            model_out_path = f'./checkpoints/reconstruction/8x8/adaptive_admm_8x8_epoch_{epoch}_{avg_psnr:.2f}dB.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'psnr': avg_psnr,
            }, model_out_path)
            print(f"Checkpoint saved to {model_out_path}")
    
    # 学習結果の可視化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss (8x8)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_psnrs)
    plt.title('Training PSNR (8x8)')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress_8x8.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training completed!")
    return model

if __name__ == "__main__":
    # 学習実行
    trained_model = train_adaptive_admm_net_8x8()