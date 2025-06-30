# マスク選択に基づく動画圧縮・再構成システム

このリポジトリは、サンプリングマスク（ビット行列）を用いた動画の圧縮および再構成システムを実装しています。本システムは、動画入力に対して最適なマスクを選択することで効率的な圧縮と高品質な再構成を実現します。

## システム概要

本システムは以下の2つの主要コンポーネントから構成されています：

1. **再構成モデル**: サンプリングマスクを用いて圧縮された動画データから元の動画を再構成します
2. **マスク選択モデル**: 入力動画に対して最適なサンプリングマスクを選択します

処理フローは以下の通りです：

1. 入力動画を[16, 8, 8]サイズのパッチに分割
2. 各パッチに対して最適なサンプリングマスクを選択
3. 選択されたマスクを用いてデータを圧縮
4. 圧縮データから元の動画を再構成

## セットアップ手順

### 必要条件

- Python 3.8+
- PyTorch 1.8+
- CUDA対応GPUを推奨

### インストール

```bash
# リポジトリのクローン
git clone https://github.com/username/video-reconstruction.git
cd video-reconstruction

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使用方法

### 再構成モデルの学習

```bash
python scripts/training/reconstruction/train_general_recon.py --config config/training/reconstruction/training_config.yaml
```

### マスクの最適化

```bash
python scripts/optimize_masks.py --input data/raw/video.mp4 --output data/masks/optimized/
```

### マスクのクラスタリング

```bash
python scripts/cluster_masks.py --input data/masks/optimized/ --output data/masks/candidates/
```

### マスク選択モデルの学習

```bash
python C:\Users\yuta-home\research\scripts\training\mask_selection\train_mask_selector.py --config C:\Users\yuta-home\research\config\training\mask_selection\training_config.yaml
```

## ディレクトリ構造

```
research/
├── README.md                      # プロジェクト概要、セットアップ手順
├── requirements.txt               # 依存パッケージ
├── config/                        # 設定ファイル
│   ├── model_config.yaml          # モデル設定
│   ├── training_config.yaml       # 学習設定
│   └── sampling_config.yaml       # サンプリング設定
├── data/                          # データ関連
│   ├── raw/                       # 元の動画データ
│   ├── processed/                 # 前処理済みデータ
│   └── masks/                     # 生成されたマスク
│       ├── optimized/             # 最適化されたマスク
│       └── candidates/            # 候補マスク
├── models/                        # モデル定義
│   ├── reconstruction/            # 再構成モデル
│   └── mask_selection/            # マスク選択モデル
├── src/                           # ソースコード
│   ├── data/                      # データ処理
│   ├── training/                  # 学習関連
│   ├── optimization/              # 最適化関連
│   ├── clustering/                # クラスタリング関連
│   └── utils/                     # ユーティリティ関数
├── scripts/                       # 実行スクリプト
├── notebooks/                     # 実験・分析用ノートブック
└── checkpoints/                   # モデルチェックポイント
```

## モデルの詳細

### 再構成モデル

再構成モデルは、サンプリングマスクによって圧縮された動画データから元の動画コンテンツを復元します。このモデルは[16, 8, 8]サイズのパッチに対して動作し、様々なタイプのマスクに対応するよう汎化されています。

### マスク選択モデル

マスク選択モデルは、入力動画の特性に基づいて、あらかじめ最適化・クラスタリングされた候補マスクの中から最適なものを選択します。これにより、動画の内容に応じた効率的な圧縮が可能になります。

## 実験結果

実験結果の詳細については `notebooks/results_analysis.ipynb` を参照してください。

## ライセンス

[ライセンス情報をここに記載]