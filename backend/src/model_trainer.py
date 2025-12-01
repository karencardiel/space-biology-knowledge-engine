import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# === RUTAS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../data/final_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "../models")
CLUSTER_ASSIGNMENTS_PATH = os.path.join(BASE_DIR, "../../data/cluster_assignments.csv") # Nueva ruta

# Asegurarse de que el directorio de modelos exista
os.makedirs(MODEL_DIR, exist_ok=True)

# === CARGAR DATA ===
print("Cargando dataset preprocesado...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset cargado con {len(df)} artículos.")

if "clean_abstract" not in df.columns:
    raise ValueError("La columna 'clean_abstract' no se encontró en el CSV. Ejecuta preprocess_abstracts.py primero.")

# === VECTORIZACIÓN TF-IDF ===
print("Entrenando TF-IDF Vectorizer...")
tfidf_vectorizer = TfidfVectorizer(max_features=1000) # Limitar a 1000 características para empezar
tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_abstract"])

# Guardar el vectorizador TF-IDF
tfidf_vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
with open(tfidf_vectorizer_path, "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"TF-IDF Vectorizer guardado en: {tfidf_vectorizer_path}")

# === CLUSTERING K-MEANS ===
print("Entrenando modelo K-Means...")
# Puedes ajustar n_clusters. Para empezar, usaremos 10.
# La elección óptima de K se haría con métodos como el "método del codo" o "coeficiente de silueta".
num_clusters = 5 
kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
cluster_labels = kmeans_model.fit_predict(tfidf_matrix) # Obtener las etiquetas de cluster

# Guardar el modelo K-Means
kmeans_model_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
with open(kmeans_model_path, "wb") as f:
    pickle.dump(kmeans_model, f)
print(f"Modelo K-Means guardado en: {kmeans_model_path}")

# === GUARDAR ASIGNACIONES DE CLUSTER EN ARCHIVO SEPARADO ===
print("Guardando asignaciones de cluster en archivo separado...")
# Crear un DataFrame con los identificadores de artículo y sus clusters
# Usaremos 'link' como identificador, asumiendo que es único.
cluster_assignments_df = pd.DataFrame({
    'link': df['link'],
    'cluster': cluster_labels
})
cluster_assignments_df.to_csv(CLUSTER_ASSIGNMENTS_PATH, index=False)
print(f"Asignaciones de cluster guardadas en: {CLUSTER_ASSIGNMENTS_PATH}")
print(cluster_assignments_df.head())

# === REGLAS DE APRIORI ===
print("\nIniciando análisis de asociación con Apriori...")

# 1. Preparar los datos: Usaremos solo las 1000 palabras clave más importantes del TF-IDF para evitar errores de memoria.
top_tfidf_features = set(tfidf_vectorizer.get_feature_names_out())
print(f"Limitando el análisis de Apriori a las {len(top_tfidf_features)} características más importantes del TF-IDF.")

# Filtrar las transacciones para que solo contengan estas palabras clave
transactions = df['clean_abstract'].apply(
    lambda abstract: [word for word in abstract.split() if word in top_tfidf_features]
).tolist()

# 2. One-Hot Encode las transacciones filtradas
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_)
print(f"Datos codificados para Apriori con {df_onehot.shape[1]} ítems únicos (filtrados por TF-IDF).")

# 3. Ejecutar Apriori para encontrar conjuntos de ítems frecuentes
# min_support=0.01 significa que el conjunto de ítems debe aparecer en al menos el 1% de los documentos (~6 artículos).
# Este es un hiperparámetro clave. Un valor más bajo puede ser computacionalmente muy costoso.
frequent_itemsets = fpgrowth(df_onehot, min_support=0.04, use_colnames=True)
print(f"Apriori encontró {len(frequent_itemsets)} conjuntos de ítems frecuentes.")

# 4. Generar reglas de asociación
# min_threshold=0.5 significa que estamos buscando reglas donde tengamos al menos un 50% de confianza.
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(f"Se generaron {len(rules)} reglas de asociación.")

# 5. Formatear y guardar las reglas
if not rules.empty:
    # Convertir los frozensets a listas para que sean serializables en JSON
    rules['antecedents'] = rules['antecedents'].apply(lambda a: list(a))
    rules['consequents'] = rules['consequents'].apply(lambda a: list(a))
    
    # Seleccionar y renombrar columnas para mayor claridad
    rules_to_save = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    
    apriori_rules_path = os.path.join(MODEL_DIR, "apriori_rules.json")
    
    with open(apriori_rules_path, 'w', encoding='utf-8') as f:
        rules_to_save.to_json(f, orient='records', indent=4)
        
    print(f"Reglas de Apriori guardadas en: {apriori_rules_path}")
else:
    print("No se encontraron reglas de asociación con los umbrales definidos. Intente bajar 'min_support' o 'min_threshold'.")

# === NOTA: final_dataset.csv NO se modifica con la columna 'cluster' ===
# El script ya no sobrescribe final_dataset.csv con la columna 'cluster'.
# Ese archivo se mantiene en su estado original, sin los resultados del clustering.