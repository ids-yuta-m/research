#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
マスク選択モデルの学習スクリプト
Experience Replayを用いた強化学習ベースのマスク選択
"""

import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from models.mask_selection.mask_selector import DecisionNetWithExpReplay
from src.data.preprocessing_id import Imgdataset
from src.training.mask_selection_trainer import MaskSelectionTrainer
from src.utils.time_utils import time2file_name

def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description='Train mask selection model')
    parser.add_argument('--config', type=str, 
                       default='config/training/mask_selection/training_config.yaml',
                       help='学習設定ファイルのパス')
    parser.add_argument('--model_config', type=str, 
                       default='config/training/mask_selection/model_config.yaml',
                       help='モデル設定ファイルのパス')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='チェックポイントファイルのパス（学習再開用）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='使用するデバイス（cuda/cpu）')
    return parser.parse_args()

def load_config(config_path):
    """設定ファイルの読み込み"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"設定ファイルが見つかりません: {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"YAML解析エラー: {e}")
        raise

def create_config_object(training_config, model_config):
    """設定辞書からConfigオブジェクトを作成"""
    class Config:
        pass
    
    config = Config()
    config.psnr_results_path = training_config['model_paths']['psnr_results']
    config.num_epochs = training_config['training']['num_epochs']
    config.learning_rate = training_config['optimization']['optimizer']['learning_rate']
    config.replay_batch_size = training_config['optimization']['replay']['replay_batch_size']
    config.min_experiences = training_config['optimization']['replay']['min_experiences']
    config.save_dir = training_config['logging']['save_dir']
    
    return config

def main():
    # 引数のパース
    args = parse_args()
    
    # 設定ファイルの読み込み
    try:
        training_config = load_config(args.config)
        model_config = load_config(args.model_config)
        print(f"設定ファイルを読み込みました:")
        print(f"  Training config: {args.config}")
        print(f"  Model config: {args.model_config}")
    except Exception as e:
        print(f"設定ファイルの読み込みに失敗しました: {e}")
        return
    
    # デバイスの設定
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f'Using device: {device}')
    
    # 設定からパラメータを取得
    data_path = training_config['data']['dataset']['train_path']
    batch_size = training_config['data']['dataloader']['batch_size']
    mask_path = training_config['model_paths']['mask_dir']
    num_masks = model_config['decision_net']['num_masks']
    
    # 設定の表示
    print("Training configuration:")
    print(f"  Data path: {data_path}")
    print(f"  Model directory: {mask_path}")
    print(f"  Number of masks: {num_masks}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of epochs: {training_config['training']['num_epochs']}")
    print(f"  Learning rate: {training_config['optimization']['optimizer']['learning_rate']}")
    
    # 日時情報を含むディレクトリ名の生成
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    
    # 保存ディレクトリの設定
    base_save_dir = training_config['logging']['save_dir']
    save_dir = os.path.join(base_save_dir, date_time)
    os.makedirs(save_dir, exist_ok=True)
    
    # 設定オブジェクトの作成
    config = create_config_object(training_config, model_config)
    config.save_dir = os.path.join("./checkpoints/mask_selection", date_time)  # 日時付きディレクトリに更新
    
    # データセットとデータローダーの設定
    dataset = Imgdataset(data_path)
    train_loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=training_config['data']['dataloader']['shuffle'],
        num_workers=training_config['data']['dataloader']['num_workers']
    )
    
    # 再構築ネットワークモデルのパスを収集
    mask_dirs = [d for d in os.listdir(mask_path) if os.path.isdir(os.path.join(mask_path, d))]
    
    # 指定されたマスク数のディレクトリを探す
    mask_dir = None
    for d in mask_dirs:
        if d.startswith(f"{num_masks}masks"):
            mask_dir = d
            break
    
    if mask_dir is None:
        print(f"Error: Could not find directory for {num_masks} masks")
        return
    
    full_mask_dir = os.path.join(mask_path, mask_dir)
    model_extension = model_config['reconstruction']['mask_candidates']['model_extension']
    model_paths = [os.path.join(full_mask_dir, f) for f in os.listdir(full_mask_dir) 
                   if f.endswith(model_extension)]
    
    if len(model_paths) < num_masks:
        print(f"Warning: Found only {len(model_paths)} models, less than requested {num_masks}")
    
    print(f"Found {len(model_paths)} reconstruction models in {full_mask_dir}")
    
    # トレーナーの初期化
    trainer = MaskSelectionTrainer(config)
    
    # 再構築ネットワークのロード
    recon_nets = trainer.load_recon_models(model_paths[:num_masks])
    print(f"Loaded {len(recon_nets)} reconstruction networks")
    
    # マスク選択モデルの初期化
    decision_net_config = model_config['decision_net']
    decision_net = DecisionNetWithExpReplay(
        num_classes=decision_net_config['num_masks'],
        epsilon_start=decision_net_config['epsilon_start'],
        epsilon_end=decision_net_config['epsilon_end'],
        epsilon_decay=decision_net_config['epsilon_decay'],
        buffer_size=decision_net_config['buffer_size']
    ).to(device)
    
    # チェックポイントからモデルをロード
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if 'state_dict' in checkpoint:
                decision_net.load_state_dict(checkpoint['state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
            else:
                decision_net.load_state_dict(checkpoint)
            print(f"チェックポイントをロードしました: {args.checkpoint}")
            print(f"エポック {start_epoch} から学習を再開します")
        except Exception as e:
            print(f"チェックポイントの読み込みに失敗しました: {e}")
            print("新しいモデルで学習を開始します")
    
    # 設定ファイルを保存ディレクトリにコピー
    import shutil
    config_save_dir = os.path.join(save_dir, 'configs')
    os.makedirs(config_save_dir, exist_ok=True)
    shutil.copy2(args.config, os.path.join(config_save_dir, 'training_config.yaml'))
    shutil.copy2(args.model_config, os.path.join(config_save_dir, 'model_config.yaml'))
    
    # トレーニングを実行
    print("Training started...")
    trained_model, best_state = trainer.train(decision_net, recon_nets, train_loader)
    
    # 最終モデルを保存
    final_save_path = os.path.join(save_dir, 'final_select_model.pth')
    torch.save({
        'state_dict': trained_model.state_dict(),
        'final_epoch': config.num_epochs,
        'best_expected_psnr': best_state['expected_psnr'] if best_state else None,
        'training_config': training_config,
        'model_config': model_config,
        'date_time': date_time
    }, final_save_path)
    
    # 学習結果の保存
    result_dict = {
        'final_epoch': config.num_epochs,
        'best_expected_psnr': best_state['expected_psnr'] if best_state else None,
        'num_masks': num_masks,
        'device': str(device),
        'date_time': date_time
    }
    
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        json.dump(result_dict, f, indent=4)
    
    print(f"Training completed. Results saved to {save_dir}")
    print(f"Final model saved to {final_save_path}")

if __name__ == '__main__':
    main()