import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# === RUTAS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/final_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "../backend/models") # Ajustar la ruta al directorio de modelos
CLUSTER_ASSIGNMENTS_PATH = os.path.join(BASE_DIR, "../data/cluster_assignments.csv") # Nueva ruta
ELBOW_PLOT_PATH = os.path.join(BASE_DIR, "elbow_method.png") # Nueva ruta para el gráfico

# === CARGAR DATA Y MODELOS ===
print("Cargando dataset y modelos entrenados...")
df = pd.read_csv(DATA_PATH)
cluster_assignments_df = pd.read_csv(CLUSTER_ASSIGNMENTS_PATH)

# Fusionar el DataFrame principal con las asignaciones de cluster
df = pd.merge(df, cluster_assignments_df, on='link', how='left')

tfidf_vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
with open(tfidf_vectorizer_path, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

kmeans_model_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
with open(kmeans_model_path, "rb") as f:
    kmeans_model = pickle.load(f)

# Verificar que la columna 'cluster' exista después de la fusión
if "cluster" not in df.columns:
    raise ValueError("La columna 'cluster' no se encontró en el CSV fusionado. Asegúrate de que model_trainer.py se ejecutó correctamente y generó cluster_assignments.csv.")

# === PREPARAR DATOS PARA EVALUACIÓN ===
# Necesitamos la matriz TF-IDF original para calcular el silhouette score
# y para extraer las palabras clave de los clusters.
tfidf_matrix = tfidf_vectorizer.transform(df["clean_abstract"])

# === EVALUACIÓN CUANTITATIVA: SILHOUETTE SCORE ===
print("\n--- Evaluación Cuantitativa ---")
score = silhouette_score(tfidf_matrix, df["cluster"])
print(f"Silhouette Score: {score:.4f}")

# === EVALUACIÓN CUALITATIVA: PALABRAS CLAVE POR CLUSTER ===
print("\n--- Evaluación Cualitativa: Palabras Clave por Cluster ---")
print("Identificando las palabras clave más representativas para cada cluster...")

# Obtener los centroides de los clusters
order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names_out()

for i in range(kmeans_model.n_clusters):
    print(f"\nCluster {i} (Artículos: {len(df[df['cluster'] == i])}):")
    # Obtener las 10 palabras clave principales para este cluster
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]
    print(f"  Palabras clave: {', '.join(top_terms)}")

    # Opcional: Mostrar algunos títulos de artículos del cluster para inspección manual
    # print("  Ejemplos de títulos:")
    # for title in df[df['cluster'] == i]['title'].head(3):
    #     print(f"    - {title}")

# === OPTIMIZACIÓN DE K: MÉTODO DEL CODO ===
print("\n--- Optimización de K: Método del Codo ---")
print("Calculando la inercia para un rango de valores de k...")
inertias = []
k_range = range(2, 21) # Probar de 2 a 20 clusters

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(tfidf_matrix)
    inertias.append(kmeans.inertia_)
    print(f"  Inercia para k={k}: {kmeans.inertia_:.2f}")

# Generar el gráfico del método del codo
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, marker='o')
plt.title('Método del Codo para Encontrar el K Óptimo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.xticks(k_range)
plt.grid(True)
plt.savefig(ELBOW_PLOT_PATH)
print(f"\nGráfico del método del codo guardado en: {ELBOW_PLOT_PATH}")
