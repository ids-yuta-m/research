import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import glob

# matファイルを読み込み
mat_files = glob.glob('./data/mask/candidates/1024masks/*.mat')
all_tensors = []

for file_path in mat_files:
    mat_data = loadmat(file_path)
    tensor_key = [k for k in mat_data.keys() if k not in ['__header__', '__version__', '__globals__']][0]
    tensor = mat_data[tensor_key]
    flattened_tensor = tensor.flatten()
    all_tensors.append(flattened_tensor)

X = np.array(all_tensors)  # (1024サンプル, 1024次元)
print(f"Original data shape: {X.shape}")

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 各クラスタ数でのMSEを計算
max_clusters = min(1024, X.shape[0])  # 最大200クラスタ（計算コストを考慮）
mse_values = []
cluster_range = list(range(1, max_clusters + 1, 5))  # 5刻みでクラスタ数を変更

print("Computing K-means clustering for different cluster numbers...")

for n_clusters in cluster_range:
    if n_clusters == 1:
        # クラスタ数が1の場合：全データの平均で近似
        cluster_center = np.mean(X_scaled, axis=0)
        X_reconstructed = np.tile(cluster_center, (X_scaled.shape[0], 1))
    else:
        # K-meansクラスタリング
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        cluster_centers = kmeans.cluster_centers_
        
        # 各データポイントを対応するクラスタ中心で再構成
        X_reconstructed = cluster_centers[cluster_labels]
    
    # MSEを計算
    mse = np.mean((X_scaled - X_reconstructed) ** 2)
    mse_values.append(mse)
    
    if n_clusters % 20 == 0 or n_clusters == 1:
        print(f"Number of clusters: {n_clusters}, MSE: {mse:.6f}")

# グラフ作成（対数スケール版）
plt.figure(figsize=(12, 8))

# 上段：対数スケール
plt.subplot(2, 1, 1)
plt.plot(cluster_range, mse_values, marker='o', linestyle='-', markersize=4)
plt.title('Mean Squared Error vs Number of Clusters (K-means) - Log Scale')
plt.xlabel('Number of Clusters')
plt.ylabel('Mean Squared Error (MSE)')
#plt.yscale('log')
plt.grid(True, alpha=0.3)

# 特定のMSE閾値を示すライン
plt.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='MSE = 0.01')
plt.axhline(y=0.001, color='g', linestyle='--', alpha=0.5, label='MSE = 0.001')
plt.axhline(y=0.0001, color='b', linestyle='--', alpha=0.5, label='MSE = 0.0001')
plt.legend()

# 下段：線形スケール
plt.subplot(2, 1, 2)
plt.plot(cluster_range, mse_values, marker='o', linestyle='-', markersize=4)
plt.title('Mean Squared Error vs Number of Clusters (K-means) - Linear Scale')
plt.xlabel('Number of Clusters')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_mse_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# エルボー法によるクラスタ数の推定
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, mse_values, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)

# MSEの変化率を計算してエルボーポイントを特定
mse_diff = np.diff(mse_values)
mse_diff2 = np.diff(mse_diff)
if len(mse_diff2) > 0:
    elbow_idx = np.argmax(mse_diff2) + 2  # 2階微分が最大の点
    elbow_clusters = cluster_range[elbow_idx]
    plt.axvline(x=elbow_clusters, color='red', linestyle='--', 
                label=f'Elbow point: {elbow_clusters} clusters')
    plt.legend()

plt.savefig('kmeans_elbow_method.png', dpi=300)
plt.show()

# 特定のMSE閾値を達成するために必要なクラスタ数を表示
print("\n=== Analysis Results ===")
mse_thresholds = [0.01, 0.001, 0.0001]
for threshold in mse_thresholds:
    try:
        clusters_needed_idx = next(i for i, mse in enumerate(mse_values) if mse <= threshold)
        clusters_needed = cluster_range[clusters_needed_idx]
        print(f"Number of clusters needed to achieve MSE ≤ {threshold}: {clusters_needed}")
    except StopIteration:
        print(f"MSE ≤ {threshold} was not achieved with up to {max_clusters} clusters")

# 最終的なクラスタリング結果の分析
optimal_clusters = 50  # 例として50クラスタを使用
print(f"\n=== Final Analysis with {optimal_clusters} clusters ===")
kmeans_final = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
final_labels = kmeans_final.fit_predict(X_scaled)
final_centers = kmeans_final.cluster_centers_

# クラスタごとのサンプル数を確認
unique_labels, counts = np.unique(final_labels, return_counts=True)
print(f"Cluster distribution: {dict(zip(unique_labels, counts))}")

# 最終MSEを計算
X_final_reconstructed = final_centers[final_labels]
final_mse = np.mean((X_scaled - X_final_reconstructed) ** 2)
print(f"Final MSE with {optimal_clusters} clusters: {final_mse:.6f}")

# クラスタ中心を代表テンソルとして保存
representative_tensors = scaler.inverse_transform(final_centers)  # 元のスケールに戻す
print(f"Representative tensors shape: {representative_tensors.shape}")
print(f"Each representative tensor represents {X.shape[1]} dimensions")