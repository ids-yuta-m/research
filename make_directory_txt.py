import os
from pathlib import Path

def generate_filtered_tree(directory, exclude_extensions=None, exclude_patterns=None, 
                          max_files_per_dir=5, show_file_count=True):
    """
    ファイル拡張子やパターンでフィルタリングしたディレクトリ構造を生成
    
    Args:
        directory: 対象ディレクトリ
        exclude_extensions: 除外する拡張子のリスト (例: ['.mp4', '.avi', '.jpg'])
        exclude_patterns: 除外するパターンのリスト (例: ['.git', '__pycache__'])
        max_files_per_dir: 1ディレクトリあたりの最大表示ファイル数
        show_file_count: ファイル数を表示するかどうか
    """
    if exclude_extensions is None:
        exclude_extensions = []
    if exclude_patterns is None:
        exclude_patterns = ['.git', '__pycache__', '.vscode', '.idea', 'venv']
    
    def should_exclude(file_path):
        """ファイルを除外するかどうかを判定"""
        # 拡張子チェック
        if file_path.suffix.lower() in exclude_extensions:
            return True
        
        # パターンチェック
        for pattern in exclude_patterns:
            if pattern in str(file_path):
                return True
        
        return False
    
    def generate_tree_recursive(path, prefix="", is_last=True, level=0):
        """再帰的にディレクトリ構造を生成"""
        result = []
        
        # ディレクトリ名を追加
        if level > 0:
            tree_symbol = "└── " if is_last else "├── "
            result.append(f"{prefix}{tree_symbol}{path.name}/")
        else:
            result.append(f"{path.name}/")
        
        try:
            # 子要素を取得
            children = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            
            # ディレクトリとファイルを分離
            directories = [child for child in children if child.is_dir() and not should_exclude(child)]
            files = [child for child in children if child.is_file() and not should_exclude(child)]
            
            # 次のレベルのプレフィックス
            next_prefix = prefix + ("    " if is_last else "│   ")
            
            # ディレクトリを先に処理
            for i, dir_path in enumerate(directories):
                is_last_dir = i == len(directories) - 1 and len(files) == 0
                result.extend(generate_tree_recursive(dir_path, next_prefix, is_last_dir, level + 1))
            
            # ファイルを処理
            if len(files) > max_files_per_dir:
                # ファイルが多い場合は一部のみ表示
                for i, file_path in enumerate(files[:max_files_per_dir]):
                    is_last_file = i == max_files_per_dir - 1
                    tree_symbol = "└── " if is_last_file else "├── "
                    result.append(f"{next_prefix}{tree_symbol}{file_path.name}")
                
                # 省略されたファイル数を表示
                if show_file_count:
                    remaining = len(files) - max_files_per_dir
                    result.append(f"{next_prefix}└── ... ({remaining} more files)")
            else:
                # 全てのファイルを表示
                for i, file_path in enumerate(files):
                    is_last_file = i == len(files) - 1
                    tree_symbol = "└── " if is_last_file else "├── "
                    result.append(f"{next_prefix}{tree_symbol}{file_path.name}")
        
        except PermissionError:
            result.append(f"{prefix}[アクセス拒否]")
        
        return result
    
    # ツリーを生成
    tree_lines = generate_tree_recursive(Path(directory))
    return '\n'.join(tree_lines)

def main():
    # 現在のディレクトリ
    current_dir = Path.cwd()
    
    # 除外する拡張子（動画、画像、音声ファイルなど）
    exclude_extensions = [
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mat',  # 動画
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', # 画像
        '.mp3', '.wav', '.ogg', '.m4a',                   # 音声
        '.zip', '.rar', '.7z', '.tar', '.gz',             # アーカイブ
        '.pt', '.pth', '.ckpt', '.h5',                    # モデルファイル
        '.pkl', '.pickle',                                # Pickle
        '.npy', '.npz',                                   # NumPy
    ]
    
    # 除外するパターン
    exclude_patterns = [
        '.git', '__pycache__', '.vscode', '.idea', 
        'venv', 'env', '.env', '.pyc', '.DS_Store',
        'node_modules', '.pytest_cache'
    ]
    
    # ディレクトリ構造を生成
    tree_output = generate_filtered_tree(
        current_dir,
        exclude_extensions=exclude_extensions,
        exclude_patterns=exclude_patterns,
        max_files_per_dir=5,  # 各ディレクトリ最大5ファイルまで表示
        show_file_count=True
    )
    
    # ファイルに保存
    output_file = "directory_filtered.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(tree_output)
    
    print(f"フィルタリングされたディレクトリ構造を {output_file} に保存しました。")
    print("\n=== 生成されたディレクトリ構造 ===")
    print(tree_output)

if __name__ == "__main__":
    main()