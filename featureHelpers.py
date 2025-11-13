import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm.notebook import tqdm

def process_dataframe_optimized(df_path, output_path, classification_label):
    """
    Optimized: Build KNN tree ONCE, then query for each point's neighbors.
    Computes only the selected features (removed normals and second derivative).
    
    Args:
        df_path (str): Path to the input CSV file.
        output_path (str): Path where the output CSV will be saved.
        classification_label (str): Label for the terrain class (e.g., 'adoquin', 'grava', 'asfalto').
    
    Returns:
        pl.DataFrame: Polars DataFrame with the computed features for each point.
    """
    # Read data
    df = pl.read_csv(df_path)
    xyz = df[['X', 'Y', 'Z']].to_numpy()
    n = len(df)

    print(f"Processing {n:,} points with label '{classification_label}'...")

    # ============================================
    # BUILD KNN TREE ONLY ONCE
    # ============================================
    print("Building KNN tree...")
    knn_model = NearestNeighbors(n_neighbors=10, algorithm='kd_tree', n_jobs=-1)
    knn_model.fit(xyz)
    
    # Get ALL neighbors at once (very fast)
    print("Finding neighbors for all points...")
    distances, all_neighbor_indices = knn_model.kneighbors(xyz)
    # all_neighbor_indices shape: (n, 10)
    # all_neighbor_indices[i] = the 10 neighbors of point i
    # ============================================

    # Pre-allocate arrays
    var_zs = np.zeros(n)
    median_nirs = np.zeros(n)
    median_refls = np.zeros(n)
    median_sigs = np.zeros(n)

    # Convert columns to numpy once (avoid repeated conversions)
    z_array = df['Z'].to_numpy()
    near_ir_array = df['NEAR_IR (photons)'].to_numpy()
    reflectivity_array = df['REFLECTIVITY (%)'].to_numpy()
    signal_array = df['SIGNAL (photons)'].to_numpy()

    # Process each point
    print("Computing features for each point...")
    for i in tqdm(range(n)):
        # Get the 10 neighbor indices for point i
        neighbor_idx = all_neighbor_indices[i]  # Shape: (10,)
        
        # Compute medians of the 10 neighbors
        median_nirs[i] = np.median(near_ir_array[neighbor_idx])
        median_refls[i] = np.median(reflectivity_array[neighbor_idx])
        median_sigs[i] = np.median(signal_array[neighbor_idx])
        
        # Compute variance of Z for the 10 neighbors
        var_zs[i] = np.var(z_array[neighbor_idx], ddof=1)

    # Create final DataFrame with selected features only
    df_features = pl.DataFrame({
        'VarZ': var_zs,
        'Height Z': z_array,
        'Median Near IR': median_nirs,
        'Median Signal': median_sigs,
        'Median Reflectivity': median_refls,
        'RANGE (mm)': df['RANGE (mm)'].to_numpy(),
        'ROW': df['ROW'].to_numpy(),
        'DESTAGGERED IMAGE COLUMN': df['DESTAGGERED IMAGE COLUMN'].to_numpy(),
        'Class': [classification_label] * n
    })

    # Save results
    df_features.write_csv(output_path)
    print(f"\nProcessing complete: {df_features.shape}")
    print(f"Saved to: {output_path}")

    return df_features