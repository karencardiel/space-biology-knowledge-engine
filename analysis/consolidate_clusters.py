import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# === CONSTANTES ===
# Define el número de sub-clusters para cada cluster principal.
RECLUSTER_TARGETS = {
    0: 4,  # Cluster 0 -> 4 sub-clusters
    1: 4,  # Cluster 1 -> 4 sub-clusters
    2: 4,  # Cluster 2 -> 4 sub-clusters
    3: 4,  # Cluster 3 -> 4 sub-clusters
    4: 3   # Cluster 4 -> 3 sub-clusters
}

# === RUTAS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/final_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "../backend/models")
INITIAL_ASSIGNMENTS_PATH = os.path.join(BASE_DIR, "../data/cluster_assignments.csv")
FINAL_ASSIGNMENTS_PATH = os.path.join(BASE_DIR, "../data/final_cluster_assignments.csv")

# === CARGAR DATA Y MODELOS ===
print("Cargando dataset, asignaciones iniciales y modelos...")
df = pd.read_csv(DATA_PATH)
initial_assignments_df = pd.read_csv(INITIAL_ASSIGNMENTS_PATH)
df = pd.merge(df, initial_assignments_df, on='link', how='left')

tfidf_vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
with open(tfidf_vectorizer_path, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# === CONSOLIDACIÓN DE CLUSTERS ===
print("Iniciando consolidación de clusters...")
df['final_cluster'] = -1 # Inicializar la columna de cluster final
next_new_cluster_id = 100 # Empezar los nuevos IDs desde 100 para evitar colisiones

# Re-clusterizar todos los clusters originales
for target_id, num_sub_clusters in RECLUSTER_TARGETS.items():
    print(f"\nRe-clusterizando el Cluster original {target_id} en {num_sub_clusters} sub-clusters...")
    
    # Aislar los datos del cluster objetivo
    target_df_indices = df[df['cluster'] == target_id].index
    
    # Asegurarse de que hay suficientes miembros para clusterizar
    if len(target_df_indices) < num_sub_clusters:
        print(f"  ADVERTENCIA: No se puede re-clusterizar el cluster {target_id} en {num_sub_clusters} sub-clusters porque solo tiene {len(target_df_indices)} miembros. Se asignará un único ID de cluster.")
        df.loc[target_df_indices, 'final_cluster'] = next_new_cluster_id
        next_new_cluster_id += 1
        continue

    target_tfidf_matrix = tfidf_vectorizer.transform(df.loc[target_df_indices, "clean_abstract"])
    
    # Realizar el re-clustering
    sub_kmeans = KMeans(n_clusters=num_sub_clusters, random_state=42, n_init=10)
    sub_labels = sub_kmeans.fit_predict(target_tfidf_matrix)
    
    # Asignar los nuevos IDs de sub-cluster
    for i in range(num_sub_clusters):
        sub_cluster_indices = target_df_indices[sub_labels == i]
        final_id = next_new_cluster_id + i
        df.loc[sub_cluster_indices, 'final_cluster'] = final_id
        print(f"  Sub-cluster {target_id}.{i} asignado al ID final {final_id}.")
    
    next_new_cluster_id += num_sub_clusters

# === GUARDAR ASIGNACIONES FINALES ===
print(f"\nGuardando asignaciones de cluster finales en: {FINAL_ASSIGNMENTS_PATH}")
final_assignments_df = df[['link', 'final_cluster']]
final_assignments_df.to_csv(FINAL_ASSIGNMENTS_PATH, index=False)

print("\nResumen de las asignaciones finales:")
print(final_assignments_df['final_cluster'].value_counts().sort_index())