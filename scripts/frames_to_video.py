import cv2
import os
import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np

def frames_to_video(frames_folder, output_video_path, fps=120, codec='mp4v'):
    """
    フレーム画像から動画を作成
    
    Args:
        frames_folder: フレーム画像が入っているフォルダパス
        output_video_path: 出力動画のパス (.mp4, .avi など)
        fps: フレームレート (default: 120)
        codec: ビデオコーデック (default: 'mp4v')
               - 'mp4v': MP4形式
               - 'XVID': AVI形式
               - 'H264': H.264エンコード（高品質）
    """
    # PNGファイルを取得して名前順にソート
    frame_files = sorted(glob.glob(os.path.join(frames_folder, "*.png")))
    
    if len(frame_files) == 0:
        raise ValueError(f"No PNG files found in {frames_folder}")
    
    print(f"Found {len(frame_files)} frames")
    print(f"Creating video at {fps} fps...")
    
    # 最初のフレームを読み込んでサイズを取得
    first_frame = cv2.imread(frame_files[0])
    height, width, channels = first_frame.shape
    
    print(f"Frame size: {width}x{height}")
    
    # VideoWriterの設定
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")
    
    # 各フレームを動画に書き込み
    for frame_file in tqdm(frame_files, desc="Writing frames"):
        frame = cv2.imread(frame_file)
        
        if frame is None:
            print(f"Warning: Failed to read {frame_file}, skipping...")
            continue
        
        video_writer.write(frame)
    
    # リソース解放
    video_writer.release()
    print(f"Video saved to: {output_video_path}")
    print(f"Duration: {len(frame_files) / fps:.2f} seconds")

def batch_convert_reds_sequences(reds_folder, output_folder, fps=120, codec='mp4v'):
    """
    REDSデータセットの複数シーケンスを一括で動画に変換
    
    Args:
        reds_folder: train_orig などのフォルダパス
        output_folder: 出力動画を保存するフォルダ
        fps: フレームレート
        codec: ビデオコーデック
    """
    # 出力フォルダ作成
    os.makedirs(output_folder, exist_ok=True)
    
    # 全シーケンスフォルダを取得
    sequence_folders = sorted([d for d in os.listdir(reds_folder) 
                              if os.path.isdir(os.path.join(reds_folder, d))])
    
    print(f"Found {len(sequence_folders)} sequences to convert")
    
    for seq_name in sequence_folders:
        seq_path = os.path.join(reds_folder, seq_name)
        output_video_path = os.path.join(output_folder, f"{seq_name}.mp4")
        
        print(f"\n{'='*60}")
        print(f"Processing sequence: {seq_name}")
        print(f"{'='*60}")
        
        try:
            frames_to_video(seq_path, output_video_path, fps=fps, codec=codec)
        except Exception as e:
            print(f"Error processing {seq_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("All sequences converted!")
    print(f"Output folder: {output_folder}")

def create_video_with_ffmpeg(frames_folder, output_video_path, fps=120, crf=18):
    """
    ffmpegを使用して高品質な動画を作成（オプション）
    
    Args:
        frames_folder: フレーム画像が入っているフォルダパス
        output_video_path: 出力動画のパス
        fps: フレームレート
        crf: 品質パラメータ (0-51, 低いほど高品質, default: 18)
    
    Note:
        ffmpegがインストールされている必要があります
        Ubuntu: sudo apt-get install ffmpeg
        macOS: brew install ffmpeg
    """
    import subprocess
    
    # フレームファイルのパターン（00000000.png, 00000001.png, ...）
    frame_pattern = os.path.join(frames_folder, "%08d.png")
    
    # ffmpegコマンド
    cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', frame_pattern,
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        '-y',  # 出力ファイルを上書き
        output_video_path
    ]
    
    print(f"Running ffmpeg command...")
    print(" ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Video created successfully: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg first.")

def verify_frame_names(frames_folder):
    """
    フレームファイル名の形式を確認
    """
    frame_files = sorted(glob.glob(os.path.join(frames_folder, "*.png")))
    
    if len(frame_files) == 0:
        print("No PNG files found!")
        return
    
    print(f"Total frames: {len(frame_files)}")
    print(f"\nFirst 5 frames:")
    for i, f in enumerate(frame_files[:5]):
        print(f"  {i}: {os.path.basename(f)}")
    
    print(f"\nLast 5 frames:")
    for i, f in enumerate(frame_files[-5:], start=len(frame_files)-5):
        print(f"  {i}: {os.path.basename(f)}")

# ====================
# 使用例
# ====================

if __name__ == "__main__":
    # 例1: 000フォルダの500フレームを動画に変換
    print("Example 1: Converting sequence 000 to video")
    
    frames_folder = "./data/raw/256x256/train_orig_part0/train/train_orig/000"  # 実際のパスに変更
    output_video_path = "./scripts/000_120fps.mp4"  # 出力パス
    
    # まずフレーム名を確認（オプション）
    # verify_frame_names(frames_folder)
    
    # OpenCVを使用して動画作成
    try:
        frames_to_video(
            frames_folder=frames_folder,
            output_video_path=output_video_path,
            fps=120,
            codec='mp4v'  # または 'H264' for better quality
        )
    except Exception as e:
        print(f"Error: {e}")
        print("Please update the paths to your actual REDS dataset location")
    
    # # 例2: 高品質動画を作成（ffmpeg使用）
    # print("\n" + "="*60)
    # print("Example 2: High-quality video with ffmpeg (optional)")
    # print("="*60)
    
    # # ffmpegがインストールされている場合はこちらを使用
    # """
    # create_video_with_ffmpeg(
    #     frames_folder=frames_folder,
    #     output_video_path="/path/to/output/000_120fps_hq.mp4",
    #     fps=120,
    #     crf=18  # 品質: 18は高品質（0=最高、23=デフォルト、51=最低）
    # )
    # """
    
    # # 例3: 全シーケンス（000-014）を一括変換
    # print("\n" + "="*60)
    # print("Example 3: Batch convert all sequences")
    # print("="*60)
    
    # reds_folder = "/path/to/REDS/train_orig"
    # output_folder = "/path/to/output/videos"
    
    # # 実際に実行する場合はコメントを外す
    # """
    # batch_convert_reds_sequences(
    #     reds_folder=reds_folder,
    #     output_folder=output_folder,
    #     fps=120,
    #     codec='mp4v'
    # )
    # """
    
    # print("\n" + "="*60)
    # print("Usage Instructions:")
    # print("="*60)
    # print("1. Update 'frames_folder' to your REDS train_orig/000 path")
    # print("2. Update 'output_video_path' to your desired output location")
    # print("3. Uncomment the function call you want to use")
    # print("4. Run the script")
    # print("\nCodec options:")
    # print("  - 'mp4v': Standard MP4 (compatible)")
    # print("  - 'H264': H.264 encoding (higher quality)")
    # print("  - 'XVID': AVI format")
    # print("\nFor best quality, use ffmpeg method with crf=18")