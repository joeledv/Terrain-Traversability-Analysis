import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors
from tqdm.notebook import tqdm
import os

def rotation_translation_y(theta=30, dx=-0.057, dy=0, dz=0.07):
    """
    Create a 4x4 homogeneous transformation matrix that rotates around the Y axis
    and applies a translation.

    Args:
        theta (float): Rotation angle in degrees about the Y axis.
        dx (float): Translation along the X axis (meters).
        dy (float): Translation along the Y axis (meters).
        dz (float): Translation along the Z axis (meters).

    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix (float).
    """
    theta_rad = np.deg2rad(theta)
    rotation_traslation = np.array([
        [np.cos(theta_rad),  0, np.sin(theta_rad), dx],
        [0,                  1, 0,                 dy],
        [-np.sin(theta_rad), 0, np.cos(theta_rad), dz],
        [0,                  0, 0,                 1]
    ])
    return rotation_traslation

def rotation_translation_z(theta=90, dx=0, dy=0, dz=0):
    """
    Create a 4x4 homogeneous transformation matrix that rotates around the Z axis
    and applies a translation.

    Args:
        theta (float): Rotation angle in degrees about the Z axis.
        dx (float): Translation along the X axis (meters).
        dy (float): Translation along the Y axis (meters).
        dz (float): Translation along the Z axis (meters).

    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix (float).
    """
    theta_rad = np.deg2rad(theta)
    rotation_traslation = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0, dx],
        [np.sin(theta_rad),  np.cos(theta_rad), 0, dy],
        [0,                  0,                 1, dz],
        [0,                  0,                 0, 1]
    ])
    return rotation_traslation

def xyz_to_rows(df, x, y, z):
    """
    Convert three named columns from a Polars DataFrame into a 4xN homogeneous
    coordinate array (rows represent x,y,z,1).

    Args:
        df (pl.DataFrame): Input Polars DataFrame.
        x (str): Column name for X coordinates.
        y (str): Column name for Y coordinates.
        z (str): Column name for Z coordinates.

    Returns:
        np.ndarray: 4xN array of homogeneous coordinates (dtype float).
    """
    points_3d = df.select([x, y, z]).to_numpy()
    xyz = points_3d.T
    xyz = np.vstack([xyz, np.ones((1, xyz.shape[1]))])
    return xyz

def new_df(df):
    """
    Normalize and transform the raw scan DataFrame:
    - Trim column names,
    - transform original X1/Y1/Z1 points to a new frame using two homogeneous transforms,
    - drop unused columns,
    - filter by spatial bounds and Z threshold,
    - apply a final Z-axis rotation.

    Args:
        df (pl.DataFrame): Raw input Polars DataFrame with columns like
                          'X1 (m)', 'Y1 (m)', 'Z1 (m)', '# TIMESTAMP (ns)', etc.

    Returns:
        pl.DataFrame: Transformed and filtered DataFrame containing columns 'X','Y','Z' and original metadata.
    """
    df = df.rename(lambda col_name: col_name.strip())

    xyz = xyz_to_rows(df, 'X1 (m)', 'Y1 (m)', 'Z1 (m)')

    transformed = rotation_translation_y() @ xyz
    df = df.with_columns([
        pl.Series('X', transformed[0]),
        pl.Series('Y', transformed[1]),
        pl.Series('Z', transformed[2])
    ])

    df = df.drop('# TIMESTAMP (ns)', 'FLAGS', 'X1 (m)', 'Y1 (m)', 'Z1 (m)', 'MEASUREMENT_ID')

    df = df.filter((pl.col('Y') >= -3) & (pl.col('Y') <= 3) & (pl.col('X') >= 0) & (pl.col('X') <= 3) & (pl.col('Z') < -0.25))

    xyz_2 = xyz_to_rows(df, 'X', 'Y', 'Z')
    transformed2 = rotation_translation_z() @ xyz_2

    df = df.with_columns([
        pl.Series('X', transformed2[0]),
        pl.Series('Y', transformed2[1]),
        pl.Series('Z', transformed2[2])
    ])

    return df

def is_good_frame(df):
    """
    Heuristic to determine whether a scan/frame is valid based on timestamp statistics.

    Args:
        df (pl.DataFrame): Polars DataFrame containing a '# TIMESTAMP (ns)' column.

    Returns:
        bool: True if the frame passes the heuristics (low timestamp std and reasonable min), False otherwise.
    """
    timestamp_col = df['# TIMESTAMP (ns)']
    return (timestamp_col.std() < 5e10 and timestamp_col.min() > 1e10)

def process_dataframe_optimized_test(df_path, output_path, n_neighbors=10):
    """
    Optimized version for test data (no classification label).
    Uses ball_tree algorithm on 2D (X,Y) coordinates and vectorized operations.
    
    Args:
        df_path (str): Path to the input CSV file.
        output_path (str): Path where the output CSV will be saved.
        n_neighbors (int): Number of nearest neighbors to use.
    
    Returns:
        pl.DataFrame: Polars DataFrame with the computed features for each point.
    """
    df = pl.read_csv(df_path)
    xy = df[['X', 'Y']].to_numpy()  
    n = len(df)

    print(f"Processing {n:,} points (test data)...")

    z_array = df['Z'].to_numpy()
    near_ir_array = df['NEAR_IR (photons)'].to_numpy()
    reflectivity_array = df['REFLECTIVITY (%)'].to_numpy()
    signal_array = df['SIGNAL (photons)'].to_numpy()
    range_array = df['RANGE (mm)'].to_numpy()
    row_array = df['ROW'].to_numpy()
    col_array = df['DESTAGGERED IMAGE COLUMN'].to_numpy()

    print(f"Building NN tree (k={n_neighbors})...")
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1)
    nn_model.fit(xy)
    
    print("Finding neighbors for all points...")
    distances, all_neighbor_indices = nn_model.kneighbors(xy)
    print("Computing features (vectorized)...")
    

    neighbor_z = z_array[all_neighbor_indices]
    neighbor_nir = near_ir_array[all_neighbor_indices]
    neighbor_refl = reflectivity_array[all_neighbor_indices]
    neighbor_sig = signal_array[all_neighbor_indices]
    
 
    var_zs = np.var(neighbor_z, axis=1, ddof=1)
    median_nirs = np.median(neighbor_nir, axis=1)
    median_refls = np.median(neighbor_refl, axis=1)
    median_sigs = np.median(neighbor_sig, axis=1)

    df_features = pl.DataFrame({
        'VarZ': var_zs,
        'Height Z': z_array,
        'Median Near IR': median_nirs,
        'Median Signal': median_sigs,
        'Median Reflectivity': median_refls,
        'RANGE (mm)': range_array,
        'ROW': row_array,
        'DESTAGGERED IMAGE COLUMN': col_array
    })

    df_features.write_csv(output_path)
    print(f"\nProcessing complete: {df_features.shape}")
    print(f"Saved to: {output_path}")

    return df_features

def process_dataframe_optimized(df_path, output_path, classification_label, n_neighbors=10):
    """
    Optimized version for training/labeled data.
    Uses ball_tree algorithm on 2D (X,Y) coordinates and vectorized operations.
    
    Args:
        df_path (str): Path to the input CSV file.
        output_path (str): Path where the output CSV will be saved.
        classification_label (str): Label for the terrain class (e.g., 'adoquin', 'grava', 'asfalto').
        n_neighbors (int): Number of nearest neighbors to use.
    
    Returns:
        pl.DataFrame: Polars DataFrame with the computed features for each point.
    """
    df = pl.read_csv(df_path)
    xy = df[['X', 'Y']].to_numpy()  
    n = len(df)

    print(f"Processing {n:,} points with label '{classification_label}'...")

    z_array = df['Z'].to_numpy()
    near_ir_array = df['NEAR_IR (photons)'].to_numpy()
    reflectivity_array = df['REFLECTIVITY (%)'].to_numpy()
    signal_array = df['SIGNAL (photons)'].to_numpy()
    range_array = df['RANGE (mm)'].to_numpy()
    row_array = df['ROW'].to_numpy()
    col_array = df['DESTAGGERED IMAGE COLUMN'].to_numpy()


    print(f"Building NN tree (k={n_neighbors}...")
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1)
    nn_model.fit(xy)
    
    print("Finding neighbors for all points...")
    distances, all_neighbor_indices = nn_model.kneighbors(xy)
    print("Computing features (vectorized)...")
    
    neighbor_z = z_array[all_neighbor_indices]
    neighbor_nir = near_ir_array[all_neighbor_indices]
    neighbor_refl = reflectivity_array[all_neighbor_indices]
    neighbor_sig = signal_array[all_neighbor_indices]
    
    var_zs = np.var(neighbor_z, axis=1, ddof=1)
    median_nirs = np.median(neighbor_nir, axis=1)
    median_refls = np.median(neighbor_refl, axis=1)
    median_sigs = np.median(neighbor_sig, axis=1)

    df_features = pl.DataFrame({
        'VarZ': var_zs,
        'Height Z': z_array,
        'Median Near IR': median_nirs,
        'Median Signal': median_sigs,
        'Median Reflectivity': median_refls,
        'RANGE (mm)': range_array,
        'ROW': row_array,
        'DESTAGGERED IMAGE COLUMN': col_array,
        'Class': [classification_label] * n
    })

    df_features.write_csv(output_path)
    print(f"\nProcessing complete: {df_features.shape}")
    print(f"Saved to: {output_path}")

    return df_features

def find_subdirectories_os(parent_dir):
    """
    Find all subdirectories within a given parent directory.
    
    Args:
        parent_dir (str): Path to the parent directory to search.
    
    Returns:
        list: List of subdirectory names (strings) found in parent_dir.
    """
    subdirectories = []
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path):
            subdirectories.append(item)
    return subdirectories