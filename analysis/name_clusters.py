import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# === CONFIGURACIÓN ===
NUM_KEYWORDS = 7
NUM_TITLES = 4

# === RUTAS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/final_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "../backend/models")
FINAL_ASSIGNMENTS_PATH = os.path.join(BASE_DIR, "../data/final_cluster_assignments.csv")

# === CARGAR DATA Y MODELOS ===
print("Cargando datos para el análisis de nombres de clusters...")
df = pd.read_csv(DATA_PATH)
final_assignments_df = pd.read_csv(FINAL_ASSIGNMENTS_PATH)
df = pd.merge(df, final_assignments_df, on='link', how='left')

tfidf_vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
with open(tfidf_vectorizer_path, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

terms = tfidf_vectorizer.get_feature_names_out()

# === ANÁLISIS Y EXTRACCIÓN DE INFO ===
print("\n--- Información para Nombrar Clusters ---\n")

for final_cluster_id in sorted(df['final_cluster'].unique()):
    cluster_articles = df[df['final_cluster'] == final_cluster_id]
    
    # Calcular palabras clave
    if not cluster_articles.empty:
        cluster_tfidf_matrix = tfidf_vectorizer.transform(cluster_articles["clean_abstract"])
        cluster_centroid = cluster_tfidf_matrix.mean(axis=0).A1
        top_indices = cluster_centroid.argsort()[::-1][:NUM_KEYWORDS]
        top_terms = [terms[ind] for ind in top_indices]
    else:
        top_terms = ["(No hay artículos)"]

    # Extraer títulos de ejemplo
    sample_titles = cluster_articles['title'].head(NUM_TITLES).tolist()

    # Imprimir información
    print(f"--- Cluster ID: {final_cluster_id} ({len(cluster_articles)} artículos) ---")
    print(f"  Palabras Clave: {', '.join(top_terms)}")
    print("  Títulos de Ejemplo:")
    for title in sample_titles:
        print(f"    - {title}")
    print("\n")

print("Análisis completado.")
