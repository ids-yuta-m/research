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

# 各サンプル数でのMSEを計算
max_components = min(1024, X_transposed.shape[1])  # 最大1024サンプル
mse_values = []
components_range = list(range(1, max_components + 1, 10))

for n_components in components_range:
    # PCAを適用：1024サンプルをn_components個に削減
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)  # (1024次元, n_components)
    
    # 再構成
    X_reconstructed_scaled = pca.inverse_transform(X_pca)
    
    # MSEを計算
    mse = np.mean((X_scaled - X_reconstructed_scaled) ** 2)
    mse_values.append(mse)
    
    if n_components % 100 == 0:
        print(f"Using {n_components} representative tensors. MSE: {mse:.6f}")

# グラフ作成（MSE版）
plt.figure(figsize=(10, 6))
plt.plot(components_range, mse_values, marker='o', linestyle='-')
plt.title('Mean Squared Error vs Number of Representative Tensors')
plt.xlabel('Number of Representative Tensors')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)
plt.tight_layout()

# MSEの場合は対数スケールが見やすい場合があります
#plt.yscale('log')  # 対数スケールを使用

# 特定のMSE閾値を示すライン（必要に応じて調整）
plt.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='MSE = 0.01')
plt.axhline(y=0.001, color='g', linestyle='--', alpha=0.5, label='MSE = 0.001')
plt.axhline(y=0.0001, color='b', linestyle='--', alpha=0.5, label='MSE = 0.0001')
plt.legend()

plt.savefig('tensor_mse_analysis.png', dpi=300)
plt.show()

# 特定のMSE閾値を達成するために必要な代表テンソル数を表示
mse_thresholds = [0.01, 0.001, 0.0001]
for threshold in mse_thresholds:
    try:
        components_needed = next(i for i, mse in enumerate(mse_values) if mse <= threshold)
        print(f"Number of representative tensors needed to achieve MSE ≤ {threshold}: {components_range[components_needed]}")
    except StopIteration:
        print(f"MSE ≤ {threshold} was not achieved. More representative tensors are needed.")

# 線形スケール版のグラフも作成（比較用）
plt.figure(figsize=(10, 6))
plt.plot(components_range, mse_values, marker='o', linestyle='-')
plt.title('Mean Squared Error vs Number of Representative Tensors (Linear Scale)')
plt.xlabel('Number of Representative Tensors')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)
plt.tight_layout()
plt.savefig('tensor_mse_analysis_linear.png', dpi=300)
plt.show()