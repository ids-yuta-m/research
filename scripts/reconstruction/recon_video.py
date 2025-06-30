import torch
import numpy as np
import scipy.io as scio
import os
import matplotlib.pyplot as plt
import math
import sys
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.reconstruction.ADMM_net import ADMM_net

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# PSNRを手動で計算する関数
def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

candidate_num = 16
video_label = "aerobatics_2"
# 必要なディレクトリが存在することを確認
output_dir = f'./data/recon/reconstructed_frames/{candidate_num}masks/{video_label}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# デバイスの設定
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# モデルとマスクのパスを設定
model_path = './checkpoints/reconstruction/model-selected_256x256/16masks/aerobatics_2/aerobatics_2_27.58.pth'
mask_path = './data/mask/selected_mask_by_model/16masks/aerobatics_mask.mat'
test_data_path = './data/raw/full_size/test_mat/aerobatics.mat'

# モデルを読み込む
print("Loading model...")

model = ADMM_net().to(device)  # 新しいモデルを作成
model = torch.load(model_path, map_location=device)
#model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()

# マスクを読み込む
print("Loading mask...")
mask_data = scio.loadmat(mask_path)
# 非特殊キーを取得
mask_keys = [k for k in mask_data.keys() if not k.startswith('__')]
if mask_keys:
    mask = torch.from_numpy(mask_data[mask_keys[0]]).to(device).float()
else:
    raise ValueError("No valid mask data found in the .mat file")

mask = mask[16:32]

# マスクをモデルに設定
with torch.no_grad():
    model.mask.weight.copy_(mask)

# テストデータを読み込む
print("Loading test data...")
test_data = scio.loadmat(test_data_path)
# 非特殊キーを取得
test_keys = [k for k in test_data.keys() if not k.startswith('__')]
if test_keys:
    # テストデータを正規化 (0-1の範囲に)
    test_frames = torch.from_numpy(test_data[test_keys[0]] / 255.0).to(device).float()
else:
    raise ValueError("No valid test data found in the .mat file")

test_frames = test_frames[16:32]

# バッチ次元を追加
test_frames = test_frames.unsqueeze(0)

# MSE損失関数を定義
criterion = torch.nn.MSELoss()

# 再構成を行う
print("Reconstructing frames...")
with torch.no_grad():
    reconstructed_frames_list = model(test_frames)
    # モデルの出力は各段階の出力のリストなので、最後の出力を使用
    reconstructed_frames = reconstructed_frames_list[-1]
    
    # PSNR計算
    psnr_value = 10 * torch.log10(1 / criterion(reconstructed_frames, test_frames))
    print(f"PSNR: {psnr_value.item():.4f} dB")

# 再構成された各フレームを保存
print("Saving reconstructed frames...")
reconstructed_frames = reconstructed_frames.squeeze(0).cpu().numpy()
original_frames = test_frames.squeeze(0).cpu().numpy()

# フレーム数を取得
num_frames = reconstructed_frames.shape[0]
print(f"Total frames: {num_frames}")

# 各フレームをPNGとして保存
for i in range(num_frames):
    # 元のフレームと再構成フレームを横に並べて表示
    plt.figure(figsize=(12, 6))
    
    # 元のフレーム
    plt.subplot(1, 2, 1)
    plt.imshow(original_frames[i], cmap='gray')
    plt.title(f"Original Frame {i+1}")
    plt.axis('off')
    
    # 再構成フレーム
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_frames[i], cmap='gray')
    plt.title(f"Reconstructed Frame {i+1}")
    plt.axis('off')
    
    # フレームごとのPSNR計算
    frame_psnr = calculate_psnr(
        original_frames[i], 
        np.clip(reconstructed_frames[i], 0, 1)  # 値を0-1の範囲に制限
    )
    plt.suptitle(f"Frame {i+1} - PSNR: {frame_psnr:.4f} dB")
    
    # 保存
    plt.savefig(f"{output_dir}/frame_{i+1}_comparison.png", bbox_inches='tight')
    plt.close()
    
    # 再構成フレームのみも別に保存
    plt.figure(figsize=(6, 6))
    plt.imshow(reconstructed_frames[i], cmap='gray')
    plt.axis('off')
    plt.savefig(f"{output_dir}/frame_{i+1}_reconstructed.png", bbox_inches='tight')
    plt.close()

# すべてのフレームのPSNRを計算して表示
frame_psnrs = []
for i in range(num_frames):
    frame_psnr = calculate_psnr(
        original_frames[i], 
        np.clip(reconstructed_frames[i], 0, 1)
    )
    frame_psnrs.append(frame_psnr)

avg_frame_psnr = np.mean(frame_psnrs)
print(f"Average PSNR across all frames: {avg_frame_psnr:.4f} dB")
print(f"Individual frame PSNRs: {[f'{p:.4f}' for p in frame_psnrs]}")

print(f"Reconstruction complete. Images saved to {output_dir}")

# 全フレームのサマリー画像も作成
plt.figure(figsize=(16, 16))
for i in range(min(16, num_frames)):
    plt.subplot(4, 4, i+1)
    plt.imshow(reconstructed_frames[i], cmap='gray')
    plt.title(f"Frame {i+1} - PSNR: {frame_psnrs[i]:.2f}dB")
    plt.axis('off')
plt.suptitle(f"All Reconstructed Frames - Avg PSNR: {avg_frame_psnr:.4f} dB")
plt.tight_layout()
plt.savefig(f"{output_dir}/all_frames_summary.png", bbox_inches='tight')
plt.close()