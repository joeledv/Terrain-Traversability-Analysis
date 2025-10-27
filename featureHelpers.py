import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def knn(xyz, index, df):
    """
    Find k nearest neighbors of a point within an XYZ array and return
    the corresponding rows from a Polars DataFrame.

    Args:
        xyz (np.ndarray): Array of shape (N, 3) with X, Y, Z coordinates.
        index (int): Index of the target point in `xyz`.
        df (pl.DataFrame): Polars DataFrame containing the same rows as `xyz`.

    Returns:
        pl.DataFrame: Subset of `df` with the nearest neighbors to point `index`.
    """
    knn = NearestNeighbors(n_neighbors=10, algorithm='kd_tree', n_jobs=-1).fit(xyz)
    distances, indexes = knn.kneighbors([xyz[index]])

    # Polars allows indexing by row indices
    return df[indexes[0]]

def estimate_normals(df):
    """
    Estimate the normal vector of a point neighborhood using PCA.

    Args:
        df (pl.DataFrame): Polars DataFrame with 'X', 'Y', 'Z' columns defining 3D points.

    Returns:
        np.ndarray: Unit normal vector of length 3 (eigenvector associated with the smallest eigenvalue).
    """
    points = df[['X', 'Y', 'Z']].to_numpy()

    # Use PCA to find the normal
    pca = PCA(n_components=3)
    pca.fit(points)

    # The normal is the eigenvector with the smallest eigenvalue
    normal = pca.components_[-1]

    return normal

def get_median(df):
    """
    Compute the median of three relevant columns in the DataFrame.

    Args:
        df (pl.DataFrame): Polars DataFrame containing
                           'NEAR_IR (photons)', 'REFLECTIVITY (%)', and 'SIGNAL (photons)'.

    Returns:
        tuple: (median_near_ir, median_reflectivity, median_signal)
    """
    # Use Polars' median directly
    median_near_ir = df['NEAR_IR (photons)'].median()
    median_reflectivity = df['REFLECTIVITY (%)'].median()
    median_signal = df['SIGNAL (photons)'].median()
    return median_near_ir, median_reflectivity, median_signal

def get_var(df):
    """
    Compute the variance of the 'Z' column in the DataFrame.

    Args:
        df (pl.DataFrame): Polars DataFrame containing the 'Z' column.

    Returns:
        float: Variance of Z.
    """
    return df.select(pl.col('Z').var()).item()

def second_derivative(df):
    """
    Compute the median of the discrete second derivative of the
    'REFLECTIVITY (%)' column using simple convolutions.

    Args:
        df (pl.DataFrame): Polars DataFrame containing 'REFLECTIVITY (%)'.

    Returns:
        float: Median of the computed discrete second derivative.
    """
    # Convert to numpy for convolution
    reflectivity_position = df['REFLECTIVITY (%)'].to_numpy()
    first = np.convolve(reflectivity_position, [1, 0, -1], mode='valid')
    second = np.convolve(first, [1, 0, -1], mode='valid')
    return np.median(second)

def process_dataframe(df_path, output_path):
    """
    Process a complete DataFrame computing features for each point and save to CSV.

    The function reads the input CSV, computes local normals, medians and variances
    over k-NN neighborhoods for each point, and writes an output CSV with the features.

    Args:
        df_path (str): Path to the input CSV file.
        output_path (str): Path where the output CSV will be saved.

    Returns:
        pl.DataFrame: Polars DataFrame with the computed features for each point.
    """
    # Read data
    df = pl.read_csv(df_path).head(1000)
    xyz = df[['X', 'Y', 'Z']].to_numpy()
    n = len(df)

    print(f"Processing {n:,} points...")

    # Pre-allocate arrays
    xnorms = np.zeros(n)
    ynorms = np.zeros(n)
    znorms = np.zeros(n)
    var_zs = np.zeros(n)
    median_nirs = np.zeros(n)
    median_refls = np.zeros(n)
    median_sigs = np.zeros(n)
    second_ders = np.zeros(n)

    # Process each point
    for i in tqdm(range(n)):
        # Get neighbors
        neighbors_df = knn(xyz, i, df)

        # Compute features
        xnorms[i], ynorms[i], znorms[i] = estimate_normals(neighbors_df)

        median_nirs[i], median_refls[i], median_sigs[i] = get_median(neighbors_df)
        var_zs[i] = get_var(neighbors_df)
        second_ders[i] = second_derivative(neighbors_df)

    # Create final DataFrame
    df_features = pl.DataFrame({
        'Xnorm': xnorms,
        'Ynorm': ynorms,
        'Znorm': znorms,
        'VarZ': var_zs,
        'Height Z': df['Z'].to_numpy(),
        'Median Near IR': median_nirs,
        'Median Signal': median_sigs,
        'Median Reflectivity': median_refls,
        'Acceleration (REFLECTIVITY %)': second_ders,
        'RANGE (mm)': df['RANGE (mm)'].to_numpy(),
        'ROW': df['ROW'].to_numpy(),
        'DESTAGGERED IMAGE COLUMN': df['DESTAGGERED IMAGE COLUMN'].to_numpy()
    })

    # Save results
    df_features.write_csv(output_path)
    print(f"\nProcessing complete: {df_features.shape}")
    print(f"Saved to: {output_path}")

    return df_features