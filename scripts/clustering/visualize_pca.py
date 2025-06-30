import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
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

# 転置：サンプル数削減のため
X_transposed = X.T  # (1024次元, 1024サンプル)
print(f"Transposed data shape: {X_transposed.shape}")

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transposed)

# 各サンプル数での再現率を計算
max_components = min(1024, X_transposed.shape[1])  # 最大1024サンプル
reconstruction_percentages = []
components_range = list(range(1, max_components + 1, 10))

for n_components in components_range:
    # PCAを適用：1024サンプルをn_components個に削減
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)  # (1024次元, n_components)
    
    # 再構成
    X_reconstructed_scaled = pca.inverse_transform(X_pca)
    
    # 再構成誤差を計算
    mse = np.mean((X_scaled - X_reconstructed_scaled) ** 2)
    
    # 再現率を計算
    total_variance = np.var(X_scaled) * X_scaled.shape[1]
    explained_variance = total_variance - mse * X_scaled.shape[1]
    reconstruction_percentage = (explained_variance / total_variance) * 100
    reconstruction_percentages.append(reconstruction_percentage)
    
    if n_components % 100 == 0:
        print(f"Using {n_components} representative tensors. Reconstruction rate: {reconstruction_percentage:.2f}%")

# グラフ作成
plt.figure(figsize=(10, 6))
plt.plot(components_range, reconstruction_percentages, marker='o', linestyle='-')
plt.title('Reconstruction Rate vs Number of Representative Tensors')
plt.xlabel('Number of Representative Tensors')
plt.ylabel('Reconstruction Rate (%)')
plt.grid(True)
plt.tight_layout()

plt.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90% Reconstruction')
plt.axhline(y=95, color='g', linestyle='--', alpha=0.5, label='95% Reconstruction')
plt.axhline(y=99, color='b', linestyle='--', alpha=0.5, label='99% Reconstruction')
plt.legend()

plt.savefig('tensor_sample_reduction_rate.png', dpi=300)
plt.show()

# 代表テンソルの取得方法
pca_final = PCA(n_components=50)  # 例：50個の代表テンソル
X_pca_final = pca_final.fit_transform(X_scaled)
representative_tensors = pca_final.components_  # (50, 1024)
print(f"Representative tensors shape: {representative_tensors.shape}")