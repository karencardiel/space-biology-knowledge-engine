import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# === CONSTANTES ===
TARGET_CLUSTER_ID = 4 # Cluster a subdividir
SUB_K_RANGE = range(2, 11) # Rango de k para el re-clustering (2 a 10)

# === RUTAS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/final_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "../backend/models")
CLUSTER_ASSIGNMENTS_PATH = os.path.join(BASE_DIR, "../data/cluster_assignments.csv")
SUB_ELBOW_PLOT_PATH = os.path.join(BASE_DIR, f"elbow_method_cluster_{TARGET_CLUSTER_ID}.png")

# === CARGAR DATA Y MODELOS ===
print("Cargando dataset y modelos entrenados...")
df = pd.read_csv(DATA_PATH)
cluster_assignments_df = pd.read_csv(CLUSTER_ASSIGNMENTS_PATH)

# Fusionar el DataFrame principal con las asignaciones de cluster
df = pd.merge(df, cluster_assignments_df, on='link', how='left')

tfidf_vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
with open(tfidf_vectorizer_path, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# === AISLAR EL CLUSTER OBJETIVO ===
print(f"\nAislando artículos del Cluster {TARGET_CLUSTER_ID}...")
target_df = df[df['cluster'] == TARGET_CLUSTER_ID].copy()
print(f"Se encontraron {len(target_df)} artículos en el Cluster {TARGET_CLUSTER_ID}.")

# === PREPARAR DATOS PARA RE-CLUSTERING ===
# No re-entrenamos el vectorizador, solo transformamos los datos del subconjunto
target_tfidf_matrix = tfidf_vectorizer.transform(target_df["clean_abstract"])

# === MÉTODO DEL CODO PARA EL SUB-CLUSTER ===
print(f"\n--- Optimización de K para el Sub-Cluster {TARGET_CLUSTER_ID}: Método del Codo ---")
print("Calculando la inercia para un rango de valores de k...")
inertias = []
for k in SUB_K_RANGE:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(target_tfidf_matrix)
    inertias.append(kmeans.inertia_)
    print(f"  Inercia para sub-k={k}: {kmeans.inertia_:.2f}")

# Generar el gráfico del método del codo para el sub-cluster
plt.figure(figsize=(10, 6))
plt.plot(SUB_K_RANGE, inertias, marker='o')
plt.title(f'Método del Codo para el Sub-Cluster {TARGET_CLUSTER_ID}')
plt.xlabel('Número de Sub-Clusters (k)')
plt.ylabel('Inercia')
plt.xticks(SUB_K_RANGE)
plt.grid(True)
plt.savefig(SUB_ELBOW_PLOT_PATH)
print(f"\nGráfico del método del codo para el sub-cluster guardado en: {SUB_ELBOW_PLOT_PATH}")

# === RE-CLUSTERING Y ANÁLISIS CUALITATIVO ===
# Basado en el gráfico, elegimos un k para el re-clustering. Empecemos con k=3 como ejemplo.
OPTIMAL_SUB_K = 3 
print(f"\n--- Re-Clustering del Cluster {TARGET_CLUSTER_ID} en {OPTIMAL_SUB_K} sub-clusters ---")
sub_kmeans_model = KMeans(n_clusters=OPTIMAL_SUB_K, random_state=42, n_init=10)
target_df['sub_cluster'] = sub_kmeans_model.fit_predict(target_tfidf_matrix)

# Obtener los centroides y términos
order_centroids = sub_kmeans_model.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names_out()

for i in range(OPTIMAL_SUB_K):
    print(f"\nSub-Cluster {TARGET_CLUSTER_ID}.{i} (Artículos: {len(target_df[target_df['sub_cluster'] == i])}):")
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]
    print(f"  Palabras clave: {', '.join(top_terms)}")
