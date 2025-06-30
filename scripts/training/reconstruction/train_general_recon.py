#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
再構成モデルの学習スクリプト
様々なタイプのマスクに対して汎化された再構成モデルを学習
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.io as scio
import json
import datetime
import sys
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 自作モジュールのインポート
from src.data.preprocessing import Imgdataset
from models.reconstruction.ADMM_net import ADMM_net
from src.utils.mask_utils import generate_masks, generate_bit_matrix
from src.training.reconstruction_trainer import ReconstructionTrainer
from src.utils.time_utils import time2file_name

def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description='再構成モデルの学習')
    parser.add_argument('--config', type=str, default='config/training/reconstruction/training_config.yaml',
                        help='学習設定ファイルのパス')
    parser.add_argument('--model_config', type=str, default='config/training/reconstruction/model_config.yaml',
                        help='モデル設定ファイルのパス')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='チェックポイントファイルのパス（学習再開用）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='使用するデバイス（cuda/cpu）')
    return parser.parse_args()

def load_config(config_path):
    """設定ファイルの読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 引数のパース
    args = parse_args()
    
    # 設定ファイルの読み込み
    try:
        training_config = load_config(args.config)
        model_config = load_config(args.model_config)
    except Exception as e:
        print(f"設定ファイルの読み込みに失敗しました: {e}")
        return
    
    # デバイスの設定
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f'Using device: {device}')
    
    # 設定からパラメータを取得
    data_path = training_config['data']['dataset']['train_path']
    test_path = training_config['data']['dataset']['test_path']
    val_path = training_config.get('data', {}).get('dataset', {}).get('val_path', None)
    batch_size = training_config['data']['dataloader']['batch_size']
    max_iter = training_config['training']['num_epochs']
    learning_rate = training_config['optimization']['optimizer']['learning_rate']
    mask_lr = training_config['optimization'].get('mask_lr', 0.0001)
    
    # データセットの準備
    dataset = Imgdataset(data_path, is_8x8=True, type="train")
    train_data_loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=training_config['data']['dataloader']['shuffle'],
        num_workers=training_config['data']['dataloader']['num_workers']
    )
    
    # テストデータローダーの準備
    test_data_loader = None
    if test_path and os.path.exists(test_path):
        test_dataset = Imgdataset(test_path, is_8x8=True, type="test")
        test_data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,  # テスト時は1サンプルずつ
            shuffle=False,
            num_workers=training_config['data']['dataloader']['num_workers']
        )
    
    # 検証データローダーの準備
    val_data_loader = None
    if val_path and os.path.exists(val_path):
        val_dataset = Imgdataset(val_path, is_8x8=True, type="val")
        val_data_loader = DataLoader(
            dataset=val_dataset,
            batch_size=1,  # 検証時は1サンプルずつ
            shuffle=False,
            num_workers=training_config['data']['dataloader']['num_workers']
        )
    
    # モデルの初期化
    network = ADMM_net().to(device)
    
    # チェックポイントからモデルをロード
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        if hasattr(state_dict, "state_dict"):
            network.load_state_dict(state_dict.state_dict())
        else:
            network.load_state_dict(state_dict)
        print(f"チェックポイントをロードしました: {args.checkpoint}")
        
        # ファイル名からエポック数を抽出
        import re
        epoch_match = re.search(r'epoch_(\d+)', args.checkpoint)
        if epoch_match:
            start_epoch = int(epoch_match.group(1))
            print(f"エポック {start_epoch} から学習を再開します")
    
    # マスクの設定
    mask_config = model_config['sampling_config']['mask']
    mask_type = mask_config['type']
    mask_dimensions = mask_config['dimensions']
    
    # マスク生成関数の定義
    def generate_random_mask():
        mask, _ = generate_masks(mask_config['path'], mask_type)
        # モデルの次元に合わせてマスクをリサイズ
        resized_mask = mask[:, 0:mask_dimensions[1], 0:mask_dimensions[2]]
        return generate_bit_matrix(resized_mask.cpu())
    
    # トレーナーの初期化
    checkpoint_dir = training_config['logging']['checkpoint_dir']
    mask_save_dir = training_config['logging']['mask_save_dir']
    recon_save_dir = training_config['logging']['recon_save_dir']
    result_save_dir = training_config['logging']['result_save_dir']
    
    # 日時情報を含むディレクトリ名の生成
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    
    # ディレクトリパスにマスク情報と日時を追加
    checkpoint_dir = os.path.join(checkpoint_dir, date_time)
    mask_save_dir = os.path.join(mask_save_dir, date_time)
    recon_save_dir = os.path.join(recon_save_dir, date_time)
    result_save_dir = os.path.join(result_save_dir, date_time)
    
    # トレーナーの設定
    trainer = ReconstructionTrainer(
        model=network,
        train_loader=train_data_loader,
        test_loader=test_data_loader,
        val_loader=val_data_loader,
        criterion=nn.MSELoss(),
        recon_lr=learning_rate,
        #mask_lr=mask_lr,
        device=device,
        checkpoint_dir=checkpoint_dir,
        #mask_save_dir=mask_save_dir,
        #recon_save_dir=recon_save_dir,
        result_save_dir=result_save_dir
    )
    
    # マスクパラメータの凍結と再構成パラメータの設定
    network.mask.requires_grad_(False)  # マスクパラメータを凍結
    
    # 学習の実行
    result_dict = trainer.train(
        max_iter=max_iter,
        start_epoch=start_epoch,
        save_interval=training_config['training']['save_interval'],
        test_interval=training_config['training']['eval_interval'],
        mask_generator=generate_random_mask,
        lr_decay=training_config['optimization']['scheduler']['decay_rate'],
        lr_decay_interval=training_config['optimization']['scheduler']['decay_steps']
    )
    
    # 結果の保存
    os.makedirs(result_save_dir, exist_ok=True)
    with open(os.path.join(result_save_dir, f'training_results_{date_time}.json'), 'w') as f:
        json.dump(result_dict, f, indent=4)
    
    print(f"学習が完了しました。結果は {result_save_dir} に保存されました。")

if __name__ == '__main__':
    main()