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
import matplotlib.colors as mcolors
import sys
import time
import datetime
from torch.distributions import Categorical


project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 既存のモデルをインポート
from scripts.training.reconstruction.train_universal_recon import UniversalADMM_net
from src.utils.time_utils import time2file_name

# CUDA設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# パス設定
MASK_DIR = "/home/yuta-m/research-github/data/masks/candidates/16masks"
GT_DIR = "/home/yuta-m/research-github/data/raw/256x256/gt_gray"
ADMM_MODEL_PATH = "/home/yuta-m/research-github/checkpoints/reconstruction/universal/2025_09_07_12_18_59/adaptive_admm_net_epoch_2000_26.59dB.pth"  # 事前学習済みUniversalADMM_netのパス

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
        
        candidate_masks.append(torch.tensor(mask, dtype=torch.float32))
    
    return candidate_masks

def create_mask_from_selection(candidate_masks, selection_indices, patch_size=8, full_size=256):
    """
    選択インデックスから完全なマスクを作成
    
    Args:
        candidate_masks: list of [16, 8, 8] candidate masks
        selection_indices: [32, 32] - 各パッチでの選択されたマスクのインデックス
        patch_size: パッチサイズ (default: 8)
        full_size: 完全画像サイズ (default: 256)
    
    Returns:
        full_mask: [16, 256, 256] - 完全なマスク
    """
    num_patches_per_side = full_size // patch_size  # 32
    full_mask = torch.zeros(16, full_size, full_size, device=selection_indices.device)
    
    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            selected_idx = selection_indices[i, j].item()
            selected_mask = candidate_masks[selected_idx].to(selection_indices.device)  # [16, 8, 8]
            
            # パッチ位置に配置
            start_i = i * patch_size
            end_i = start_i + patch_size
            start_j = j * patch_size
            end_j = start_j + patch_size
            
            full_mask[:, start_i:end_i, start_j:end_j] = selected_mask
    
    return full_mask

def sample_masks_from_probs(mask_probs, candidate_masks):
    """
    確率分布からマスクをサンプリング
    
    Args:
        mask_probs: [B, num_candidates, 32, 32] - 各パッチでの選択確率
        candidate_masks: list of candidate masks
    
    Returns:
        sampled_masks: [B, 16, 256, 256] - サンプリングされたマスク
        sampled_indices: [B, 32, 32] - サンプリングされたインデックス
        log_probs: [B, 32, 32] - サンプリングのログ確率
    """
    B, num_candidates, H, W = mask_probs.shape
    sampled_masks = []
    sampled_indices = torch.zeros(B, H, W, dtype=torch.long, device=mask_probs.device)
    log_probs = torch.zeros(B, H, W, device=mask_probs.device)
    
    for b in range(B):
        batch_indices = torch.zeros(H, W, dtype=torch.long, device=mask_probs.device)
        batch_log_probs = torch.zeros(H, W, device=mask_probs.device)
        
        for i in range(H):
            for j in range(W):
                # 各パッチでカテゴリカル分布からサンプリング
                probs = mask_probs[b, :, i, j]
                dist = Categorical(probs)
                sampled_idx = dist.sample()
                
                batch_indices[i, j] = sampled_idx
                batch_log_probs[i, j] = dist.log_prob(sampled_idx)
        
        # 完全なマスクを作成
        full_mask = create_mask_from_selection(candidate_masks, batch_indices)
        sampled_masks.append(full_mask)
        
        sampled_indices[b] = batch_indices
        log_probs[b] = batch_log_probs
    
    sampled_masks = torch.stack(sampled_masks)
    return sampled_masks, sampled_indices, log_probs

def calculate_psnr(img1, img2):
    """PSNR計算"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def load_ground_truth_data(gt_dir, max_files=None):
    """教師データを読み込む"""
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

def train_mask_selection_model():
    """マスク選択モデルの学習"""
    print("Loading data...")
    candidate_masks = load_candidate_masks(MASK_DIR)
    gt_data = load_ground_truth_data(GT_DIR)  # メモリ節約のため制限
    
    num_candidates = len(candidate_masks)
    print(f"Loaded {num_candidates} candidate masks")
    print(f"Loaded {len(gt_data)} ground truth videos")
    
    # モデル初期化
    selection_model = MaskSelectionNet(num_candidates=num_candidates).to(device)
    
    # 事前学習済みUniversalADMM_netを読み込み（推論専用）
    admm_model = UniversalADMM_net(t=16, s=256, patch_size=8).to(device)
    if os.path.exists(ADMM_MODEL_PATH):
        checkpoint = torch.load(ADMM_MODEL_PATH, map_location=device)
        admm_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained ADMM model from {ADMM_MODEL_PATH}")
    else:
        print("Warning: No pretrained ADMM model found. Using randomly initialized model.")
    
    admm_model.eval()  # 推論専用に設定
    
    # オプティマイザー
    optimizer = torch.optim.Adam(selection_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    
    # 学習ログ
    train_rewards = []
    train_losses = []
    
    num_epochs = 1000
    batch_size = 4
    
    # ベースライン（移動平均）
    baseline_reward = 0.0
    baseline_momentum = 0.9
    
    date_time = time2file_name(str(datetime.datetime.now()))
    
    print("Starting mask selection training...")
    
    for epoch in range(1, num_epochs + 1):
        selection_model.train()
        epoch_reward = 0.0
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\rTraining Epoch: {epoch}/{num_epochs}", end="", flush=True)
        
        for batch_start in range(0, len(gt_data), batch_size):
            batch_end = min(batch_start + batch_size, len(gt_data))
            batch_gt = gt_data[batch_start:batch_end]
            batch_gt = torch.stack(batch_gt).to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                # 前回のモデル状態を保存（比較用）
                prev_model_state = {name: param.clone() for name, param in selection_model.named_parameters()}
            
            # 現在のモデルでマスク選択
            mask_logits, mask_probs = selection_model(batch_gt)
            current_masks, current_indices, current_log_probs = sample_masks_from_probs(mask_probs, candidate_masks)
            
            # 現在のマスクでADMM再構成
            with torch.no_grad():
                current_reconstructed_list = admm_model(batch_gt, current_masks)
                current_reconstruction = current_reconstructed_list[-1]
            
            # 前回のモデルでマスク選択（比較用）
            if epoch > 1:  # 最初のエポックは比較対象がないのでランダムマスクを使用
                # 前回のモデル状態を復元
                with torch.no_grad():
                    temp_model = MaskSelectionNet(num_candidates=num_candidates).to(device)
                    temp_model.load_state_dict({name: prev_model_state.get(name, param) 
                                              for name, param in temp_model.named_parameters()})
                    temp_model.eval()
                    
                    _, prev_mask_probs = temp_model(batch_gt)
                    prev_masks, _, _ = sample_masks_from_probs(prev_mask_probs, candidate_masks)
                    
                    prev_reconstructed_list = admm_model(batch_gt, prev_masks)
                    prev_reconstruction = prev_reconstructed_list[-1]
            else:
                # 初回はランダムマスク（パフォーマンス比較用）
                random_masks = []
                for _ in range(len(batch_gt)):
                    random_indices = torch.randint(0, num_candidates, (32, 32), device=device)
                    random_mask = create_mask_from_selection(candidate_masks, random_indices)
                    random_masks.append(random_mask)
                prev_masks = torch.stack(random_masks)
                
                with torch.no_grad():
                    prev_reconstructed_list = admm_model(batch_gt, prev_masks)
                    prev_reconstruction = prev_reconstructed_list[-1]
            
            # 報酬計算（PSNR差分）
            batch_rewards = []
            for i in range(len(batch_gt)):
                current_psnr = calculate_psnr(current_reconstruction[i], batch_gt[i])
                prev_psnr = calculate_psnr(prev_reconstruction[i], batch_gt[i])
                reward = current_psnr - prev_psnr
                batch_rewards.append(reward)
            
            batch_rewards = torch.tensor(batch_rewards, device=device)
            avg_reward = batch_rewards.mean().item()
            
            # ベースライン更新
            baseline_reward = baseline_momentum * baseline_reward + (1 - baseline_momentum) * avg_reward
            
            # REINFORCE損失計算
            advantages = batch_rewards - baseline_reward
            policy_loss = 0.0
            
            for b in range(len(batch_gt)):
                # 各パッチでの対数確率にadvantageを掛けて平均
                patch_losses = -current_log_probs[b] * advantages[b]
                policy_loss += patch_losses.mean()
            
            policy_loss = policy_loss / len(batch_gt)
            
            # 逆伝播
            policy_loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(selection_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_reward += avg_reward
            epoch_loss += policy_loss.item()
            num_batches += 1
        
        # エポック平均
        avg_reward = epoch_reward / num_batches
        avg_loss = epoch_loss / num_batches
        
        scheduler.step()
        
        train_rewards.append(avg_reward)
        train_losses.append(avg_loss)
        
        # ログ出力
        if epoch % 10 == 0:
            print(f"\n====================================")
            print(f"Epoch {epoch}")
            print(f"Average Reward (PSNR improvement): {avg_reward:.4f} dB")
            print(f"Policy Loss: {avg_loss:.6f}")
            print(f"Baseline Reward: {baseline_reward:.4f} dB")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # モデル保存
        if epoch % 100 == 0:
            os.makedirs(f'/home/yuta-m/research-github/checkpoints/mask_selection/{date_time}', exist_ok=True)
            model_out_path = f'/home/yuta-m/research-github/checkpoints/mask_selection/{date_time}/mask_selection_net_epoch_{epoch}_reward_{avg_reward:.4f}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': selection_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward': avg_reward,
                'loss': avg_loss,
                'baseline_reward': baseline_reward,
                'num_candidates': num_candidates,
            }, model_out_path)
            print(f"Checkpoint saved to {model_out_path}")
    
    # 学習結果の可視化
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_rewards)
    plt.title('Training Reward (PSNR Improvement)')
    plt.xlabel('Epoch')
    plt.ylabel('Reward (dB)')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_losses)
    plt.title('Policy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 最終的なマスク選択マップを表示
    plt.subplot(1, 3, 3)
    selection_model.eval()
    with torch.no_grad():
        # テストサンプルで最終的なマスク選択を可視化
        test_sample = gt_data[0].unsqueeze(0).to(device)
        _, final_mask_probs = selection_model(test_sample)
        final_selected = torch.argmax(final_mask_probs[0], dim=0).cpu().numpy()
        
        # カラーマップ作成
        if num_candidates <= 10:
            colors = ['red', 'blue', 'green', 'orange', 'purple', 
                     'brown', 'pink', 'gray', 'olive', 'cyan'][:num_candidates]
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, min(num_candidates, 20)))
        cmap = mcolors.ListedColormap(colors)
        
        im = plt.imshow(final_selected, cmap=cmap, vmin=0, vmax=num_candidates-1)
        plt.title('Final Mask Selection Map')
        plt.xlabel('Patch X')
        plt.ylabel('Patch Y')
        plt.colorbar(im, label='Mask Index')
    
    plt.tight_layout()
    plt.savefig('mask_selection_training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Mask selection training completed!")
    return selection_model

def test_mask_selection_model(selection_model_path, admm_model_path, num_test_samples=10):
    """学習済みマスク選択モデルのテスト"""
    # データ読み込み
    candidate_masks = load_candidate_masks(MASK_DIR)
    test_data = load_ground_truth_data(GT_DIR, max_files=num_test_samples)
    
    # モデル読み込み
    checkpoint = torch.load(selection_model_path, map_location=device)
    num_candidates = checkpoint['num_candidates']
    
    selection_model = MaskSelectionNet(num_candidates=num_candidates).to(device)
    selection_model.load_state_dict(checkpoint['model_state_dict'])
    selection_model.eval()
    
    admm_model = UniversalADMM_net(t=16, s=256, patch_size=8).to(device)
    admm_checkpoint = torch.load(admm_model_path, map_location=device)
    admm_model.load_state_dict(admm_checkpoint['model_state_dict'])
    admm_model.eval()
    
    test_psnrs = []
    
    with torch.no_grad():
        for i, gt_data in enumerate(test_data):
            gt = gt_data.unsqueeze(0).to(device)
            
            # マスク選択
            _, mask_probs = selection_model(gt)
            
            # 確率的サンプリングの代わりにargmaxで決定的選択
            selected_indices = torch.argmax(mask_probs, dim=1)  # [1, 32, 32]
            selected_mask = create_mask_from_selection(candidate_masks, selected_indices[0])
            selected_mask = selected_mask.unsqueeze(0)  # [1, 16, 256, 256]
            
            # ADMM再構成
            reconstructed_list = admm_model(gt, selected_mask)
            final_reconstruction = reconstructed_list[-1]
            
            # PSNR計算
            psnr = calculate_psnr(final_reconstruction[0], gt[0])
            test_psnrs.append(psnr)
            
            print(f"Test sample {i+1}: PSNR = {psnr:.2f} dB")
    
    print(f"Average test PSNR: {np.mean(test_psnrs):.2f} ± {np.std(test_psnrs):.2f} dB")

if __name__ == "__main__":
    # マスク選択モデルの学習実行
    trained_selection_model = train_mask_selection_model()
    
    # テスト実行（オプション）
    # test_mask_selection_model("path/to/mask_selection_model.pth", ADMM_MODEL_PATH)