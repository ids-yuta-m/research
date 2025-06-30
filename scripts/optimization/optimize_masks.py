#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
マスク最適化スクリプト
"""

import os
import argparse
import torch
import warnings
import sys
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config_utils import load_config
from src.utils.model_utils import load_model, reset_model_mask
from src.data.mask_data_loader import MaskDataLoader
from src.optimization.mask_optimizer import MaskOptimizer

# 警告を抑制
warnings.filterwarnings("ignore")


def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description='マスク最適化スクリプト')
    parser.add_argument('--config', type=str, default='config/optimization/mask_optimization_config.yaml',
                        help='設定ファイルへのパス')
    parser.add_argument('--group_id', type=int, default=None,
                        help='特定のグループIDのみを処理する場合に指定')
    parser.add_argument('--skip_existing', action='store_true',
                        help='既に処理済みのグループをスキップ')
    return parser.parse_args()


def main():
    """メイン処理"""
    # 引数のパース
    args = parse_args()
    
    # 設定ファイルの読み込み
    config = load_config(args.config)
    
    # GPUの設定
    os.environ["CUDA_VISIBLE_DEVICES"] = config['hardware']['gpu_id']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # 出力ディレクトリの作成
    output_dir = config['io']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # データローダーの初期化
    data_loader = MaskDataLoader(config['io']['group_dir'], device)
    
    # 処理対象のグループIDを取得
    if args.group_id is not None:
        group_ids = [args.group_id]
    else:
        group_ids = data_loader.get_group_ids()
    
    print(f"処理対象グループ数: {len(group_ids)}")
    
    # モデルの読み込み
    model_path = config['io']['model_path']
    model = load_model(model_path, device)
    
    # マスク最適化クラスの初期化
    optimizer = MaskOptimizer(model, config)
    
    # 各グループに対して処理
    for group_id in group_ids:
        print(f"\n🔍 グループ {group_id} の処理中...")
        
        # 既に処理済みの場合はスキップ
        if args.skip_existing and data_loader.should_skip_group(group_id, output_dir):
            print(f"✔ グループ {group_id} は既に処理済み → スキップ")
            continue
        
        try:
            # パッチデータの読み込み
            patches = data_loader.load_group_patches(group_id)
            
            # モデルのマスクをリセット
            t, h, w = config['model']['patch_size']
            reset_model_mask(model, t=t, s=h)  # 正方形パッチを仮定
            
            # マスクの最適化
            optimizer.optimize_mask_for_patch(patches, f"{group_id:04d}")
            
        except Exception as e:
            print(f"⚠ グループ {group_id} の処理中にエラーが発生: {e}")
    
    print("\n✅ すべての処理が完了しました")


if __name__ == "__main__":
    main()