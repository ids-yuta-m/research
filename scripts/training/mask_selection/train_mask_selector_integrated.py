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
    高速版: torch.multinomial を使って一括サンプリング
    """
    B, num_candidates, H, W = mask_probs.shape

    # (B, num_candidates, H, W) → (B*H*W, num_candidates)
    probs = mask_probs.permute(0, 2, 3, 1).reshape(-1, num_candidates)

    # 一括サンプリング
    sampled_indices = torch.multinomial(probs, 1).squeeze(-1)  # [B*H*W]

    # ログ確率も計算
    log_probs = torch.gather(torch.log(probs + 1e-12), 1, sampled_indices.unsqueeze(-1)).squeeze(-1)  # [B*H*W]

    # 元の形に戻す
    sampled_indices = sampled_indices.view(B, H, W)  # [B, H, W]
    log_probs = log_probs.view(B, H, W)             # [B, H, W]

    # 各サンプルについて [16, 256, 256] のマスクを作成
    sampled_masks = []
    for b in range(B):
        full_mask = create_mask_from_selection(candidate_masks, sampled_indices[b])
        sampled_masks.append(full_mask)
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
    gt_data = load_ground_truth_data(GT_DIR)  
    
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
    
    print("Starting mask selection training...")
    
    for epoch in range(1, num_epochs + 1):
        selection_model.train()
        epoch_reward = 0.0
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\rTraining Epoch: {epoch}/{num_epochs}\n", end="", flush=True)
        
        total_files = len(gt_data)
        
        for batch_start in range(0, len(gt_data), batch_size):
            batch_end = min(batch_start + batch_size, len(gt_data))
            batch_gt = gt_data[batch_start:batch_end]
            batch_gt = torch.stack(batch_gt).to(device)
            
            processed = batch_end
            
            print(f"Epoch {epoch}/{num_epochs}  Progress: {processed}/{total_files}", end="\r", flush=True)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                # 前回のモデル状態を保存（比較用）
                prev_model_state = selection_model.state_dict()
            
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
                    temp_model.load_state_dict(prev_model_state)
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
            date_time = time2file_name(str(datetime.datetime.now()))
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