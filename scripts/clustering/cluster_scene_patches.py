#!/usr/bin/env python
"""
シーンパッチクラスタリングの実行スクリプト

使用例:
    python scripts/clustering/run_patch_clustering.py --config config/clustering/patch_clustering_config.yaml
"""

import os
import sys
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.clustering import PatchClusterer


def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='シーンパッチのクラスタリングを実行')
    
    parser.add_argument('--config', type=str, required=True,
                        help='設定ファイルのパス')
    parser.add_argument('--input_dir', type=str,
                        help='入力データディレクトリ（設定ファイルを上書き）')
    parser.add_argument('--output_dir', type=str,
                        help='出力先ディレクトリ（設定ファイルを上書き）')
    parser.add_argument('--n_clusters', type=int,
                        help='クラスタ数（設定ファイルを上書き）')
    parser.add_argument('--patch_size', type=int,
                        help='パッチサイズ（設定ファイルを上書き）')
    parser.add_argument('--min_patches', type=int,
                        help='クラスタあたりの最小パッチ数（これ以下のクラスタは削除）')
    
    return parser.parse_args()


def load_config(config_path):
    """YAML設定ファイルの読み込み"""
    try:
        # UTF-8でファイルを開く
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except UnicodeDecodeError:
        # UTF-8で読み込めない場合はcp932（Shift-JIS）で試行
        try:
            with open(config_path, 'r', encoding='cp932') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"設定ファイルの読み込みに失敗しました: {e}")
            print("設定ファイルはUTF-8またはShift-JISエンコーディングである必要があります。")
            sys.exit(1)


def main():
    """メイン関数"""
    # 引数の解析
    args = parse_arguments()
    
    # 設定ファイルの読み込み
    config = load_config(args.config)
    
    # コマンドライン引数で設定を上書き
    if args.input_dir:
        config['input_dir'] = args.input_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.n_clusters:
        config['n_clusters'] = args.n_clusters
    if args.patch_size:
        config['patch_size'] = args.patch_size
    if args.min_patches:
        config['min_patches_per_cluster'] = args.min_patches
        
    # 最小パッチ数が設定されていない場合のデフォルト値
    if 'min_patches_per_cluster' not in config:
        config['min_patches_per_cluster'] = 5
    
    # 設定の表示
    print("クラスタリング設定:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 入力ディレクトリの確認
    if not os.path.exists(config['input_dir']):
        print(f"エラー: 入力ディレクトリ {config['input_dir']} が存在しません")
        sys.exit(1)
    
    # クラスタリングの実行
    clusterer = PatchClusterer(config)
    
    try:
        summary_path = clusterer.run()
        print(f"クラスタリング完了! 詳細は {summary_path} を確認してください")
    except Exception as e:
        print(f"エラー: クラスタリング中に例外が発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()