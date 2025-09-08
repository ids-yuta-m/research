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
import sys
import time
import datetime

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.utils.time_utils import time2file_name

# CUDA設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# パス設定
MASK_DIR = "/home/yuta-m/research-github/data/masks/candidates/128masks"
GT_DIR = "/home/yuta-m/research-github/data/raw/256x256/gt_gray"

class MaskAdaptiveUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.decoder(feat)

class UniversalADMM_net(nn.Module):
    def __init__(self, t=16, s=256, patch_size=8):
        super().__init__()
        self.t = t
        self.s = s
        self.patch_size = patch_size
        
        # マスク埋め込み層
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(t, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # マスク適応型UNet（9段階）
        self.unet_stages = nn.ModuleList([
            MaskAdaptiveUNet(t + 32, t) for _ in range(9)
        ])
        
        # 適応的ガンマパラメータ予測器
        self.gamma_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 9),
            nn.Softplus()  # 正の値を保証
        )

    def forward(self, x, mask):
        # 入力テンソルの形状確認
        if hasattr(self, '_debug_once'):
            if not self._debug_once:
                print(f"Model input - x shape: {x.shape}, mask shape: {mask.shape}")
                self._debug_once = True
        else:
            self._debug_once = False
        
        # マスク特徴抽出
        mask_feat = self.mask_encoder(mask.float())
        adaptive_gammas = self.gamma_predictor(mask_feat)
        
        if not self._debug_once:
            print(f"Mask features shape: {mask_feat.shape}")
            print(f"Adaptive gammas shape: {adaptive_gammas.shape}")
        
        # 測定生成
        y = torch.sum(x * mask, dim=1)  # [B, 256, 256]
        
        if not self._debug_once:
            print(f"Measurement y shape: {y.shape}")
        
        # ADMM反復初期化
        x_list = []
        theta = self.At(y, mask)
        b = torch.zeros_like(x)
        
        # 9段階のADMM反復
        for i, unet in enumerate(self.unet_stages):
            gamma = adaptive_gammas[:, i:i+1, None, None]  # [B, 1, 1, 1]
            
            # ADMMステップ
            yb = self.A(theta + b, mask)
            Phi_s = torch.sum(mask, dim=1, keepdim=True)  # [B, 1, 256, 256]
            
            # ゼロ除算を防ぐ & gamma形状を合わせる
            Phi_s_safe = Phi_s + 1e-8
            gamma_expanded = gamma.expand_as(Phi_s_safe)  # [B, 1, 256, 256]
            
            residual = torch.div(y.unsqueeze(1) - yb.unsqueeze(1), Phi_s_safe + gamma_expanded)
            x = theta + b + self.At(residual.squeeze(1), mask)
            x1 = x - b
            
            # マスク情報を含めた入力
            mask_aware_input = torch.cat([x1, mask_feat], dim=1)  # [B, 16+32, 256, 256]
            
            # UNet処理（残差接続）
            theta = unet(mask_aware_input) + x1
            b = b - (x - theta)
            x_list.append(theta)
            
            if not self._debug_once and i == 0:
                print(f"Stage {i} - gamma: {gamma.shape}, Phi_s: {Phi_s.shape}")
                print(f"Stage {i} - residual: {residual.shape}, theta: {theta.shape}")
                self._debug_once = True
        
        return x_list
    
    def A(self, x, mask):
        """Forward operator: x -> y"""
        temp = x * mask
        y = torch.sum(temp, dim=1)
        return y

    def At(self, y, mask):
        """Transpose operator: y -> x"""
        # yが[B, 256, 256]の場合とy.unsqueeze(1)で[B, 1, 256, 256]の場合に対応
        if y.dim() == 3:  # [B, H, W]
            temp = torch.unsqueeze(y, 1).repeat(1, mask.shape[1], 1, 1)
        else:  # [B, 1, H, W] または [B, C, H, W]
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
        # .matファイルの構造に応じて適切なキーを選択
        key = [k for k in mask_data.keys() if not k.startswith('_')][0]
        mask = mask_data[key]  # [16, 8, 8]
        
        # データ型とサイズの確認
        mask = np.array(mask, dtype=np.float32)
        if mask.shape != (16, 8, 8):
            # もし形状が違う場合は転置や次元調整を行う
            if mask.ndim == 3:
                # 可能な形状パターンを試す
                if mask.shape == (8, 8, 16):
                    mask = mask.transpose(2, 0, 1)
                elif mask.shape == (8, 16, 8):
                    mask = mask.transpose(1, 0, 2)
            print(f"Warning: Mask shape was {mask_data[key].shape}, reshaped to {mask.shape}")
        
        candidate_masks.append(torch.tensor(mask, dtype=torch.float32))
    
    return candidate_masks

def create_full_size_mask(candidate_masks, patch_size=8, full_size=256):
    """パッチごとにランダムマスクを選択して256x256マスクを作成"""
    num_patches_per_side = full_size // patch_size  # 32
    full_mask = torch.zeros(16, full_size, full_size)
    
    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            # ランダムに候補マスクを選択
            selected_mask = random.choice(candidate_masks)  # [16, 8, 8]
            
            # パッチ位置に配置
            start_i = i * patch_size
            end_i = start_i + patch_size
            start_j = j * patch_size
            end_j = start_j + patch_size
            
            full_mask[:, start_i:end_i, start_j:end_j] = selected_mask
    
    return full_mask

def load_ground_truth_data(gt_dir, max_files=None):
    """教師データを読み込む"""
    gt_files = glob.glob(os.path.join(gt_dir, "*.mat"))
    if max_files:
        gt_files = gt_files[:max_files]
    
    gt_data = []
    print(f"Loading {len(gt_files)} ground truth files...")
    for gt_file in tqdm(gt_files):
        data = sio.loadmat(gt_file)
        # .matファイルの構造に応じて適切なキーを選択
        key = [k for k in data.keys() if not k.startswith('_')][0]
        video = data[key]  # [16, 256, 256]
        
        # データ型とサイズの確認
        video = np.array(video, dtype=np.float32)
        if video.shape != (16, 256, 256):
            # 可能な形状パターンを試す
            if video.ndim == 3:
                if video.shape == (256, 256, 16):
                    video = video.transpose(2, 0, 1)
                elif video.shape == (256, 16, 256):
                    video = video.transpose(1, 0, 2)
            #print(f"Warning: GT shape was {data[key].shape}, reshaped to {video.shape}")
        
        # 正規化（0-1範囲）
        video = (video - video.min()) / (video.max() - video.min() + 1e-8)
        gt_data.append(torch.tensor(video, dtype=torch.float32))
    
    return gt_data

def calculate_psnr(img1, img2, mask=None):
    """PSNR計算"""
    if mask is not None:
        # マスク領域のみでPSNR計算
        mse = torch.mean(((img1 - img2) * mask) ** 2)
        mask_ratio = torch.mean(mask)
        mse = mse / (mask_ratio + 1e-8)
    else:
        mse = torch.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0  # データが正規化されている場合
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def train_adaptive_admm_net():
    """適応的ADMM_netの学習"""
    # データ読み込み
    candidate_masks = load_candidate_masks(MASK_DIR)
    gt_data = load_ground_truth_data(GT_DIR)  # メモリ節約のため最初の100ファイル
    
    print(f"Loaded {len(candidate_masks)} candidate masks")
    print(f"Loaded {len(gt_data)} ground truth videos")
    
    # データ形状の確認とデバッグ情報
    if len(candidate_masks) > 0:
        print(f"Candidate mask shape: {candidate_masks[0].shape}")
        print(f"Candidate mask value range: {candidate_masks[0].min():.3f} - {candidate_masks[0].max():.3f}")
    if len(gt_data) > 0:
        print(f"GT data shape: {gt_data[0].shape}")
        print(f"GT data value range: {gt_data[0].min():.3f} - {gt_data[0].max():.3f}")
    
    # テスト用マスク作成
    test_mask = create_full_size_mask(candidate_masks)
    print(f"Created full size mask shape: {test_mask.shape}")
    print(f"Full mask value range: {test_mask.min():.3f} - {test_mask.max():.3f}")
    
    # モデル初期化
    model = UniversalADMM_net(t=16, s=256, patch_size=8).to(device)
    
    # 既存コードを参考にした学習率設定
    mask_lr = 0.0001
    recon_lr = 0.0001
    
    # オプティマイザの分離（既存コードと同様）
    mask_optimizer = torch.optim.Adam([
        {'params': model.mask_encoder.parameters(), 'lr': mask_lr},
        {'params': model.gamma_predictor.parameters(), 'lr': mask_lr}
    ])
    
    recon_optimizer = torch.optim.Adam([
        {'params': model.unet_stages.parameters(), 'lr': recon_lr}
    ])
    
    # 学習率スケジューラを追加
    mask_scheduler = torch.optim.lr_scheduler.StepLR(mask_optimizer, step_size=600, gamma=0.9)
    recon_scheduler = torch.optim.lr_scheduler.StepLR(recon_optimizer, step_size=600, gamma=0.9)
    
    # 既存コードを参考にしたMSE損失
    criterion = nn.MSELoss()
    
    # 学習ログ
    train_losses = []
    train_psnrs = []
    
    num_epochs = 2000
    batch_size = 4  # 既存コードと同じ
    
    print("Starting training...")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0
        start_time = time.time()
        
        # エポック進行状況を一行で表示（既存コードと同様）
        print("\n====================================")
        print(f"\rTraining Epoch: {epoch}/{num_epochs}", end="", flush=True)
        
        # バッチ処理
        for batch_start in range(0, len(gt_data), batch_size):
            batch_end = min(batch_start + batch_size, len(gt_data))
            batch_gt = gt_data[batch_start:batch_end]
            
            # バッチをGPUに移動
            batch_gt = torch.stack(batch_gt).to(device)
            
            # 各サンプルに対してランダムマスクを生成
            batch_masks = []
            for _ in range(len(batch_gt)):
                mask = create_full_size_mask(candidate_masks)
                batch_masks.append(mask)
            batch_masks = torch.stack(batch_masks).to(device)
            
            # 勾配初期化
            mask_optimizer.zero_grad()
            recon_optimizer.zero_grad()
            
            # 前向き計算
            reconstructed_list = model(batch_gt, batch_masks)
            
            # 既存コードを参考にした多段階損失（重み付きMSE）
            recon_loss = (torch.sqrt(criterion(reconstructed_list[-1], batch_gt)) + 
                         0.5 * torch.sqrt(criterion(reconstructed_list[-2], batch_gt)) + 
                         0.5 * torch.sqrt(criterion(reconstructed_list[-3], batch_gt)))
            
            total_loss = recon_loss
            
            # 逆伝播
            total_loss.backward()
            
            # 勾配クリッピング（安定化）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # オプティマイザ更新
            mask_optimizer.step()
            recon_optimizer.step()
            
            # メトリクス計算
            with torch.no_grad():
                # PSNRを既存コードと同じ方法で計算
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
        
        # 学習率更新
        mask_scheduler.step()
        recon_scheduler.step()
        
        train_losses.append(avg_loss)
        train_psnrs.append(avg_psnr)
        
        # # 学習率減衰（既存コードと同様）
        # if epoch % 6 == 0:
        #     mask_lr = mask_lr * 0.9
        #     recon_lr = recon_lr * 0.9
        #     for param_group in mask_optimizer.param_groups:
        #         param_group['lr'] = mask_lr
        #     for param_group in recon_optimizer.param_groups:
        #         param_group['lr'] = recon_lr
        
        # ログ出力（既存コードのスタイル）
        if epoch % 1 == 0:
            
            print(f"Loss: {avg_loss:.6f}")
            print(f"PSNR: {avg_psnr:.2f} dB")
            print(f"Time: {time_taken:.2f}s")
            print(f"Mask LR: {mask_lr:.6f}, Recon LR: {recon_lr:.6f}")
        
        # モデル保存
        if epoch % 100 == 0:
            # 日時情報を含むディレクトリ名の生成
            date_time = str(datetime.datetime.now())
            date_time = time2file_name(date_time)
            os.makedirs(f'/home/yuta-m/research-github/checkpoints/reconstruction/universal/{date_time}', exist_ok=True)
            model_out_path = f'/home/yuta-m/research-github/checkpoints/reconstruction/universal/{date_time}/adaptive_admm_net_epoch_{epoch}_{avg_psnr:.2f}dB.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'mask_optimizer_state_dict': mask_optimizer.state_dict(),
                'recon_optimizer_state_dict': recon_optimizer.state_dict(),
                'loss': avg_loss,
                'psnr': avg_psnr,
            }, model_out_path)
            print(f"Checkpoint saved to {model_out_path}")
    
    # 学習結果の可視化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_psnrs)
    plt.title('Training PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training completed!")
    return model

def test_model(model_path, candidate_masks, test_data, num_test_samples=5):
    """学習済みモデルのテスト"""
    model = UniversalADMM_net(t=16, s=256, patch_size=8).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_psnrs = []
    
    with torch.no_grad():
        for i in range(min(num_test_samples, len(test_data))):
            gt = test_data[i].unsqueeze(0).to(device)
            mask = create_full_size_mask(candidate_masks).unsqueeze(0).to(device)
            
            reconstructed_list = model(gt, mask)
            final_reconstruction = reconstructed_list[-1]
            
            psnr = calculate_psnr(final_reconstruction[0], gt[0], mask[0])
            test_psnrs.append(psnr)
            
            print(f"Test sample {i+1}: PSNR = {psnr:.2f} dB")
    
    print(f"Average test PSNR: {np.mean(test_psnrs):.2f} ± {np.std(test_psnrs):.2f} dB")

def test_model_during_training(model, candidate_masks, test_data, device):
    """学習中のテスト（既存コードのスタイル）"""
    model.eval()
    criterion = nn.MSELoss()
    psnr_samples = []
    
    with torch.no_grad():
        for i, gt_data in enumerate(test_data[:10]):  # 最初の10サンプルのみテスト
            gt = gt_data.unsqueeze(0).to(device).float()
            mask = create_full_size_mask(candidate_masks).unsqueeze(0).to(device)
            
            # 再構成
            out_pic_list = model(gt, mask)
            out_pic = out_pic_list[-1]
            
            # PSNRを既存コードと同じ方法で計算
            psnr_val = 10 * torch.log10(1 / criterion(out_pic, gt))
            psnr_samples.append(psnr_val.item())
    
    avg_psnr = np.mean(psnr_samples)
    return avg_psnr

if __name__ == "__main__":
    # 学習実行
    trained_model = train_adaptive_admm_net()
    
    # テスト実行（オプション）
    # candidate_masks = load_candidate_masks(MASK_DIR)
    # test_data = load_ground_truth_data(GT_DIR, max_files=10)
    # test_psnr = test_model_during_training(trained_model, candidate_masks, test_data, device)
    # print(f"Final test PSNR: {test_psnr:.2f} dB")