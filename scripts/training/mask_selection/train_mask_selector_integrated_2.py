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
import math

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 既存のモデルをインポート
from scripts.training.reconstruction.train_universal_recon import UniversalADMM_net
from src.utils.time_utils import time2file_name
from models.mask_selection.mask_selection_net import MaskSelectionNet

# CUDA設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# パス設定
MASK_DIR = "/home/yuta-m/research-github/data/masks/candidates/16masks"
GT_DIR = "/home/yuta-m/research-github/data/raw/256x256/gt_gray"
ADMM_MODEL_PATH = "/home/yuta-m/research-github/checkpoints/reconstruction/universal/2025_09_07_12_18_59/adaptive_admm_net_epoch_2000_26.59dB.pth"


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
    """選択インデックスから完全なマスクを作成"""
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

def sample_masks_with_exploration(mask_probs, candidate_masks, epsilon=0.1, temperature=1.0):
    """
    改良版サンプリング: ε-greedy + temperature scaling for better exploration
    """
    B, num_candidates, H, W = mask_probs.shape
    
    # Temperature scalingを適用
    scaled_logits = torch.log(mask_probs + 1e-12) / temperature
    scaled_probs = F.softmax(scaled_logits, dim=1)
    
    # ε-greedy exploration
    if random.random() < epsilon:
        # Random exploration
        uniform_probs = torch.ones_like(scaled_probs) / num_candidates
        sampled_indices = torch.multinomial(uniform_probs.permute(0, 2, 3, 1).reshape(-1, num_candidates), 1).squeeze(-1)
        log_probs = torch.log(uniform_probs.permute(0, 2, 3, 1).reshape(-1, num_candidates) + 1e-12)
        log_probs = torch.gather(log_probs, 1, sampled_indices.unsqueeze(-1)).squeeze(-1)
    else:
        # Policy-based sampling
        probs = scaled_probs.permute(0, 2, 3, 1).reshape(-1, num_candidates)
        sampled_indices = torch.multinomial(probs, 1).squeeze(-1)
        log_probs = torch.gather(torch.log(probs + 1e-12), 1, sampled_indices.unsqueeze(-1)).squeeze(-1)
    
    # 元の形に戻す
    sampled_indices = sampled_indices.view(B, H, W)
    log_probs = log_probs.view(B, H, W)
    
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

def cosine_annealing_lr(optimizer, epoch, max_epochs, lr_max, lr_min=1e-6):
    """Cosine annealing learning rate schedule"""
    lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_mask_selection_model_fixed_baseline():
    """固定ベースライン（ランダムマスク）との比較でマスク選択モデルを学習"""
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
    
    print("Starting mask selection training with fixed baseline...")
    
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
            
            # # === デバッグ: 勾配追跡のチェック ===
            # print(f"\n=== GRADIENT DEBUG at epoch {epoch}, batch {batch_start//batch_size} ===")
            
            # === 学習モデルでマスク選択 ===
            mask_logits, mask_probs = selection_model(batch_gt)
            
            # print(f"mask_logits requires_grad: {mask_logits.requires_grad}")
            # print(f"mask_probs requires_grad: {mask_probs.requires_grad}")
            
            learned_masks, learned_indices, learned_log_probs = sample_masks_with_exploration(mask_probs, candidate_masks)
            
            # print(f"learned_log_probs requires_grad: {learned_log_probs.requires_grad}")
            # print(f"learned_log_probs grad_fn: {learned_log_probs.grad_fn}")
            
            # === 固定ベースライン（ランダムマスク）を生成 ===
            random_masks = []
            for _ in range(len(batch_gt)):
                random_indices = torch.randint(0, num_candidates, (32, 32), device=device)
                random_mask = create_mask_from_selection(candidate_masks, random_indices)
                random_masks.append(random_mask)
            random_masks = torch.stack(random_masks)
            
            # === 同じビデオ・同じモデルで再構成比較 ===
            with torch.no_grad():
                # 学習モデル選択マスクで再構成
                learned_reconstructed_list = admm_model(batch_gt, learned_masks)
                learned_reconstruction = learned_reconstructed_list[-1]
                
                # ランダムマスクで再構成
                random_reconstructed_list = admm_model(batch_gt, random_masks)
                random_reconstruction = random_reconstructed_list[-1]
            
            # === 報酬計算（各ビデオでの改善度） ===
            batch_rewards = []
            for i in range(len(batch_gt)):
                learned_psnr = calculate_psnr(learned_reconstruction[i], batch_gt[i])
                random_psnr = calculate_psnr(random_reconstruction[i], batch_gt[i])
                reward = learned_psnr - random_psnr  # ランダムベースラインからの改善度
                batch_rewards.append(reward)
            
            batch_rewards = torch.tensor(batch_rewards, device=device)
            
            # print(f"batch_rewards requires_grad: {batch_rewards.requires_grad}")
            # print(f"baseline_reward type: {type(baseline_reward)}, value: {baseline_reward}")
            
            avg_reward = batch_rewards.mean().item()
            
            # ベースライン更新（移動平均）
            baseline_reward = baseline_momentum * baseline_reward + (1 - baseline_momentum) * avg_reward
            
            # === REINFORCEポリシー損失計算 ===
            advantages = batch_rewards - baseline_reward
            
            # print(f"advantages requires_grad: {advantages.requires_grad}")
            # print(f"advantages grad_fn: {advantages.grad_fn}")
            
            policy_loss = 0.0
            patch_loss = 0.0
            
            for b in range(len(batch_gt)):
                
                # print(f"\nSample {b}:")
                # print(f"  learned_log_probs[{b}] requires_grad: {learned_log_probs[b].requires_grad}")
                # print(f"  learned_log_probs[{b}] grad_fn: {learned_log_probs[b].grad_fn}")
                # print(f"  advantages[{b}] requires_grad: {advantages[b].requires_grad}")
                # print(f"  advantages[{b}] value: {advantages[b].item():.6f}")
                
                detached_advantage = advantages[b].detach()
                # print(f"  detached_advantage requires_grad: {detached_advantage.requires_grad}")
                
                # 各パッチでの対数確率にadvantageを掛けて平均
                weighted_log_probs = -learned_log_probs[b] * detached_advantage
                
                # print(f"  weighted_log_probs requires_grad: {weighted_log_probs.requires_grad}")
                # print(f"  weighted_log_probs grad_fn: {weighted_log_probs.grad_fn}")
                
                patch_loss += weighted_log_probs.mean()
                
                # print(f"  patch_loss requires_grad: {patch_loss.requires_grad}")
                # print(f"  patch_loss grad_fn: {patch_loss.grad_fn}")
                
                policy_loss += patch_loss
            
            policy_loss = policy_loss / len(batch_gt)
            
            # print(f"\nFinal policy_loss requires_grad: {policy_loss.requires_grad}")
            # print(f"Final policy_loss grad_fn: {policy_loss.grad_fn}")
            # print(f"Final policy_loss value: {policy_loss.item():.6f}")
            
            
            
            # # デバッグ情報（詳細版）
            # if epoch == 1 and batch_start == 0:
            #     print(f"\nDETAILED DEBUG INFO:")
            #     print(f"Batch rewards: {batch_rewards}")
            #     print(f"Baseline reward: {baseline_reward:.6f}")
            #     print(f"Advantages: {advantages}")
            #     print(f"Log probs shape: {learned_log_probs[0].shape}")
            #     print(f"Log probs range: [{learned_log_probs[0].min().item():.4f}, {learned_log_probs[0].max().item():.4f}]")
            #     print(f"Sample log prob values: {learned_log_probs[0][0, 0]:.4f}, {learned_log_probs[0][15, 15]:.4f}")
                
            #     # 各サンプルの詳細
            #     for i in range(len(batch_gt)):
            #         sample_loss = (-learned_log_probs[i] * advantages[i].detach()).mean()
            #         print(f"Sample {i}: advantage={advantages[i].item():.4f}, sample_loss={sample_loss.item():.6f}")
                
            #     print(f"Policy loss value: {policy_loss.item():.6f}")
            #     print(f"Policy loss requires_grad: {policy_loss.requires_grad}")
            
            # # 追加のログ出力（最初の数エポック）
            # if epoch <= 5:
            #     print(f"Epoch {epoch}, Batch {batch_start//batch_size}: "
            #           f"Avg Reward: {avg_reward:.4f}, "
            #           f"Baseline: {baseline_reward:.4f}, "
            #           f"Policy Loss: {policy_loss.item():.6f}")
            
            # # policy_lossが勾配計算可能かチェック
            # if not policy_loss.requires_grad:
            #     print(f"Warning: policy_loss does not require grad at epoch {epoch}")
            #     continue
            
            # # 6. モデルパラメータの勾配設定チェック
            # print(f"\nModel parameters gradient status:")
            # for name, param in selection_model.named_parameters():
            #     print(f"  {name}: requires_grad={param.requires_grad}")
            #     if not param.requires_grad:
            #         print(f"    WARNING: Parameter {name} does not require grad!")
                    
            # # 最初の数バッチのみデバッグ出力
            # if epoch == 1 and batch_start < 2 * batch_size:
            #     print("="*80)
            #     input("Press Enter to continue...")
            
            # # policy_lossが勾配を持たない場合はスキップ
            # if not policy_loss.requires_grad:
            #     print(f"SKIPPING backward pass: policy_loss does not require grad")
            #     continue
            
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
        if epoch % 1 == 0:
            print(f"\n====================================")
            print(f"Epoch {epoch}")
            print(f"Average Reward (PSNR improvement over random): {avg_reward:.4f} dB")
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
    
    return selection_model

def evaluate_vs_random_baseline(selection_model_path, admm_model_path, num_test_samples=20):
    """学習済みモデルとランダムベースラインの詳細比較"""
    
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
    
    learned_psnrs = []
    random_psnrs = []
    improvements = []
    
    print("Evaluating learned model vs random baseline...")
    
    with torch.no_grad():
        for i, gt_data in enumerate(tqdm(test_data)):
            gt = gt_data.unsqueeze(0).to(device)
            
            # === 学習済みモデルでマスク選択 ===
            _, mask_probs = selection_model(gt)
            selected_indices = torch.argmax(mask_probs, dim=1)  # 決定論的選択
            learned_mask = create_mask_from_selection(candidate_masks, selected_indices[0])
            learned_mask = learned_mask.unsqueeze(0)
            
            # === ランダムマスク生成 ===
            random_indices = torch.randint(0, num_candidates, (32, 32), device=device)
            random_mask = create_mask_from_selection(candidate_masks, random_indices)
            random_mask = random_mask.unsqueeze(0)
            
            # === 同じビデオで再構成比較 ===
            # 学習済みマスクで再構成
            learned_reconstructed_list = admm_model(gt, learned_mask)
            learned_reconstruction = learned_reconstructed_list[-1]
            learned_psnr = calculate_psnr(learned_reconstruction[0], gt[0])
            
            # ランダムマスクで再構成
            random_reconstructed_list = admm_model(gt, random_mask)
            random_reconstruction = random_reconstructed_list[-1]
            random_psnr = calculate_psnr(random_reconstruction[0], gt[0])
            
            # 改善度計算
            improvement = learned_psnr - random_psnr
            
            learned_psnrs.append(learned_psnr)
            random_psnrs.append(random_psnr)
            improvements.append(improvement)
            
            print(f"Video {i+1:2d}: Learned={learned_psnr:.2f}dB, Random={random_psnr:.2f}dB, Improvement={improvement:+.2f}dB")
    
    # 統計サマリー
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Learned Model PSNR:  {np.mean(learned_psnrs):.2f} ± {np.std(learned_psnrs):.2f} dB")
    print(f"Random Baseline PSNR: {np.mean(random_psnrs):.2f} ± {np.std(random_psnrs):.2f} dB")
    print(f"Average Improvement:  {np.mean(improvements):.2f} ± {np.std(improvements):.2f} dB")
    print(f"Improvement Range:    {np.min(improvements):.2f} to {np.max(improvements):.2f} dB")
    print(f"Positive Improvements: {np.sum(np.array(improvements) > 0)}/{len(improvements)} videos")
    
    # 改善度の分布
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(random_psnrs, learned_psnrs, alpha=0.7)
    plt.plot([min(random_psnrs), max(random_psnrs)], [min(random_psnrs), max(random_psnrs)], 'r--', label='Equal Performance')
    plt.xlabel('Random Baseline PSNR (dB)')
    plt.ylabel('Learned Model PSNR (dB)')
    plt.title('PSNR Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.hist(improvements, bins=15, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='No Improvement')
    plt.xlabel('PSNR Improvement (dB)')
    plt.ylabel('Frequency')
    plt.title('Improvement Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    video_indices = range(1, len(improvements) + 1)
    plt.bar(video_indices, improvements, alpha=0.7, 
            color=['green' if imp > 0 else 'red' for imp in improvements])
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    plt.xlabel('Video Index')
    plt.ylabel('PSNR Improvement (dB)')
    plt.title('Per-Video Improvement')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learned_vs_random_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'learned_psnrs': learned_psnrs,
        'random_psnrs': random_psnrs,
        'improvements': improvements,
        'mean_improvement': np.mean(improvements),
        'std_improvement': np.std(improvements)
    }

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
    trained_selection_model = train_mask_selection_model_fixed_baseline()
    
    # テスト実行（オプション）
    # test_mask_selection_model("path/to/mask_selection_model.pth", ADMM_MODEL_PATH)