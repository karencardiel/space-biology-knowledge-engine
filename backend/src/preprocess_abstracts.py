import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords

# Descargar stopwords si no están
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# === RUTAS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../data/final_merged.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "../../data/final_dataset.csv")

# === FUNCIÓN DE PREPROCESAMIENTO ===
def preprocess_text(text):
    """Limpia y normaliza texto en inglés."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # pasar a minúsculas
    text = re.sub(r"[^a-z\s]", "", text)  # eliminar caracteres no alfabéticos
    tokens = text.split()  # dividir en palabras
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]  # quitar stopwords
    return " ".join(tokens)

# === CARGAR DATA ===
print("📄 Cargando dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset cargado con {len(df)} artículos")

# Eliminar duplicados basados en el link
df.drop_duplicates(subset=['link'], keep='first', inplace=True)
print(f"Dataset después de eliminar duplicados: {len(df)} artículos")

# Eliminar columna source si existe
if "source" in df.columns:
    df = df.drop(columns=["source"])

if "abstract" not in df.columns:
    raise ValueError("No se encontró la columna 'abstract' en el CSV")

# === APLICAR PREPROCESAMIENTO ===
print("🧹 Limpiando abstracts...")
df["clean_abstract"] = df["abstract"].apply(preprocess_text)

# === GUARDAR RESULTADOS ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nArchivo preprocesado guardado en: {OUTPUT_PATH}")
print(df.head(3))
