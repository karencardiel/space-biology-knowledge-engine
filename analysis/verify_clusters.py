import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans # KMeans is not directly used for centroids here, but might be useful for context

# === RUTAS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/final_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "../backend/models")
FINAL_ASSIGNMENTS_PATH = os.path.join(BASE_DIR, "../data/final_cluster_assignments.csv") # Updated path
HTML_REPORT_PATH = os.path.join(BASE_DIR, "cluster_report.html")

# === CARGAR DATA Y MODELOS ===
print("Cargando dataset y asignaciones de clusters finales para el reporte...")
df = pd.read_csv(DATA_PATH)
final_assignments_df = pd.read_csv(FINAL_ASSIGNMENTS_PATH) # Load final assignments

# Fusionar el DataFrame principal con las asignaciones de cluster finales
df = pd.merge(df, final_assignments_df, on='link', how='left')

# Cargar TF-IDF Vectorizer
tfidf_vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
with open(tfidf_vectorizer_path, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# === GENERAR REPORTE HTML ===
print(f"Generando reporte HTML en: {HTML_REPORT_PATH}")

terms = tfidf_vectorizer.get_feature_names_out()

with open(HTML_REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write('<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Análisis de Clusters Finales</title>')
    f.write('<style>body { font-family: sans-serif; line-height: 1.6; margin: 2em; } h1, h2 { border-bottom: 2px solid #ccc; padding-bottom: 5px; } .cluster-block { margin-bottom: 3em; }</style>')
    f.write('</head><body><h1>Análisis de Clusters Finales</h1>')

    # Iterar sobre cada cluster final único
    for final_cluster_id in sorted(df['final_cluster'].unique()):
        f.write(f'<div class="cluster-block">')
        cluster_articles = df[df['final_cluster'] == final_cluster_id] # Filter by final_cluster
        
        # Calcular palabras clave para este cluster final
        if not cluster_articles.empty:
            cluster_tfidf_matrix = tfidf_vectorizer.transform(cluster_articles["clean_abstract"])
            cluster_centroid = cluster_tfidf_matrix.mean(axis=0).A1 # Added .A1 to flatten to 1D array
            
            # Get top terms for this centroid
            top_n = 10
            top_indices = cluster_centroid.argsort()[::-1][:top_n] # Simplified indexing for 1D array
            top_terms = [terms[ind] for ind in top_indices]
        else:
            top_terms = ["(No hay artículos)"]

        f.write(f'<h2>Cluster Final {final_cluster_id} ({len(cluster_articles)} Artículos)</h2>')
        f.write(f"<p><strong>Palabras Clave:</strong> {', '.join(top_terms)}</p>")
        f.write('<h3>Títulos de Artículos:</h3><ol>')
        for idx, row in cluster_articles.iterrows():
            f.write(f"<li>{row['title']}</li>")
        f.write('</ol></div>')

    f.write('</body></html>')

print("Reporte HTML final generado con éxito.")