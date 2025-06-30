import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.clustering.mask_clustering import cluster_bit_matrices, cluster_bit_matrices_pca
from src.utils.io_utils import load_bit_matrices, save_cluster_centers

def main():
    # Configuration
    base_dir = './data/mask/candidates/1024masks'
    out_dir_name = './data/mask/candidates_pca'
    use_pca = True
    cluster_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    cluster_results = None
    
    # Step 1: Load bit matrices
    bit_matrices, file_paths = load_bit_matrices(base_dir)
    
    if not bit_matrices:
        print("No suitable bit matrices found. Exiting.")
        return
    
    if use_pca:
        cluster_results = cluster_bit_matrices_pca(bit_matrices, cluster_sizes)
    else:
        # Step 2: Perform clustering
        cluster_results = cluster_bit_matrices(bit_matrices, cluster_sizes)
    
    # Step 3: Save results
    save_cluster_centers(cluster_results, file_paths, base_dir, out_dir_name)
    
    print("\n🎉 All clustering and saving completed.")

if __name__ == "__main__":
    main()