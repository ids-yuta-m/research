import os
import torch
import numpy as np
import scipy.io as scio
from torch.nn.functional import mse_loss
from tqdm import tqdm
import sys
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from models.reconstruction.ADMM_net import ADMM_net

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# モデルロード（固定）
model = ADMM_net().to(DEVICE)
state_dict = torch.load('./pre-train_epoch_12300_state_dict.pth', map_location=DEVICE)
if hasattr(state_dict, "state_dict"):
    model.load_state_dict(state_dict.state_dict())
else:
    model.load_state_dict(state_dict)
model.eval()

# 入力ディレクトリ

input_dir = './data/raw/full_size/gt_mat'

def record_psnr(candidate_num, mask_dir, save_dir):
    # マスク一覧（アルファベット順）
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.mat')])
    mat_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mat')])

    # 各パッチのPSNRリスト（マスクごと）
    psnr_dict = {}

    # 各マスクを適用
    for mask_idx, mask_file in enumerate(mask_files):
        print(f"Processing mask: {mask_file}")
        mask_data = scio.loadmat(os.path.join(mask_dir, mask_file))
        mask_key = [k for k in mask_data.keys() if not k.startswith('__')][0]
        mask_tensor = torch.from_numpy(mask_data[mask_key]).float().to(DEVICE)  # [t, 8, 8] など
        if mask_tensor.dim() == 4:
            mask_tensor = mask_tensor.squeeze(0)  # [t, 8, 8]
        model.mask.weight.data = mask_tensor

        # 全 .mat ファイルで評価
        for mat_file in tqdm(mat_files, desc=f"Mask {mask_idx+1}/{len(mask_files)}"):
            file_path = os.path.join(input_dir, mat_file)
            fname = os.path.splitext(mat_file)[0]
            data = scio.loadmat(file_path)
            keys = [k for k in data.keys() if not k.startswith('__')]
            if len(keys) != 1:
                raise ValueError(f"{mat_file} に期待した1つの主要キーが見つかりません: {keys}")
            tensor = data[keys[0]]  # shape: [t, 256, 256]
            t, h, w = tensor.shape

            for patch_h in range(5):
                for patch_w in range(5):
                    patch = tensor[:, patch_h * 48:(patch_h * 6 + 1) * 8, patch_w * 48:(patch_w * 6 + 1) * 8]
                    for start_frame in range(0, t - 15, 16):
                        sub_patch = patch[start_frame:start_frame + 16]  # [16, 8, 8]
                        input_tensor = torch.from_numpy(sub_patch / 255.).float().unsqueeze(0).to(DEVICE)

                        with torch.no_grad():
                            output = model(input_tensor)[-1]

                        mse = mse_loss(output, input_tensor)
                        psnr = 10 * torch.log10(1.0 / mse).item()
                        #print(psnr)

                        key = f"{fname}_{patch_h * 5 + patch_w}_{int(start_frame / 16)}"
                        if key not in psnr_dict:
                            psnr_dict[key] = []
                        psnr_dict[key].append(psnr)
    # 保存
    np.savez(save_dir, **psnr_dict)

# candidate_num = 32
# for _ in range(3):
#     mask_dir = f'./candidate_mask_2/{candidate_num}masks'
#     save_dir = f'./candidate_psnrs_2/{candidate_num}masks.npz'
#     record_psnr(candidate_num, mask_dir, save_dir)
#     candidate_num = candidate_num*2

candidate_num = 16
mask_dir = f'./data/mask/candidates/{candidate_num}masks'
save_dir = f'./data/psnr_record/{candidate_num}masks.npz'
record_psnr(candidate_num, mask_dir, save_dir)
print("すべてのマスクでのPSNR結果を 'all_mask_psnr.npz' に保存しました。")
