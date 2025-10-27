import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def knn(xyz, index, df):
    knn = NearestNeighbors(n_neighbors=10, algorithm='kd_tree', n_jobs=-1).fit(xyz)
    distances, indexes = knn.kneighbors([xyz[index]])

    # Polars usa .filter en lugar de .iloc
    return df[indexes[0]]

def estimate_normals(df):
    points = df[['X', 'Y', 'Z']].to_numpy()

    # PCA para encontrar la normal
    pca = PCA(n_components=3)
    pca.fit(points)

    # La normal es el eigenvector con menor eigenvalor
    normal = pca.components_[-1]

    return normal

def get_median(df):
    # Polars usa .median() directamente
    median_near_ir = df['NEAR_IR (photons)'].median()
    median_reflectivity = df['REFLECTIVITY (%)'].median()
    median_signal = df['SIGNAL (photons)'].median()
    return median_near_ir, median_reflectivity, median_signal

def get_var(df):
    return df.select(pl.col('Z').var()).item()

def second_derivative(df):
    # Polars usa .to_numpy() para convertir a numpy
    reflectivity_position = df['REFLECTIVITY (%)'].to_numpy()
    first = np.convolve(reflectivity_position, [1, 0, -1], mode='valid')
    second = np.convolve(first, [1, 0, -1], mode='valid')
    return np.median(second)

def process_dataframe(df_path, output_path):
    """
    Procesa un DataFrame completo calculando features para cada punto.

    Args:
        df_path: Ruta al archivo CSV de entrada
        output_path: Ruta donde guardar el CSV de salida

    Returns:
        df_features: DataFrame de Polars con las features calculadas
    """
    # Leer datos
    df = pl.read_csv(df_path).head(1000)
    xyz = df[['X', 'Y', 'Z']].to_numpy()
    n = len(df)

    print(f"Procesando {n:,} puntos...")

    # Pre-alocar arrays
    xnorms = np.zeros(n)
    ynorms = np.zeros(n)
    znorms = np.zeros(n)
    var_zs = np.zeros(n)
    median_nirs = np.zeros(n)
    median_refls = np.zeros(n)
    median_sigs = np.zeros(n)
    second_ders = np.zeros(n)

    # Procesar cada punto
    for i in tqdm(range(n)):
        # Obtener vecinos
        neighbors_df = knn(xyz, i, df)

        # Calcular features
        xnorms[i], ynorms[i], znorms[i] = estimate_normals(neighbors_df)

        median_nirs[i], median_refls[i], median_sigs[i] = get_median(neighbors_df)
        var_zs[i] = get_var(neighbors_df)
        second_ders[i] = second_derivative(neighbors_df)

    # Crear DataFrame final
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

    # Guardar
    df_features.write_csv(output_path)
    print(f"\n✓ Procesamiento completo: {df_features.shape}")
    print(f"✓ Guardado en: {output_path}")

    return df_features