import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os

# === CONFIG ===
# Carpeta actual (backend/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpeta data
DATA_DIR = os.path.join(BASE_DIR, "../../data")

# Archivos dentro de la carpeta data
DATA_PATH = os.path.join(DATA_DIR, "SB_publication_PMC.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "cleaned_articles.csv")
FAILED_OUTPUT_PATH = os.path.join(DATA_DIR, "failed_articles.csv")

def extract_abstract(url):
    """
    Extrae el abstract (resumen) del artículo en NCBI/PMC.
    Devuelve texto limpio o None si no lo encuentra.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Buscar el section que contiene el abstract
        abstract_section = soup.find("section", {"class": "abstract"})
        if not abstract_section:
            return None

        # Obtener el texto dentro del section (ignorando "Abstract" del h2)
        text = abstract_section.get_text(separator=" ", strip=True)

        # Quitar la palabra "Abstract" al inicio si existe
        if text.lower().startswith("abstract"):
            text = text[len("abstract"):].strip()

        return text

    except Exception as e:
        print(f"Error extrayendo {url}: {e}")
        return None


def clean_data():
    # Leer CSV original
    df = pd.read_csv(DATA_PATH)
    print(f"CSV cargado con {len(df)} artículos")

    # Normalizar nombres de columnas: minúsculas y sin espacios
    df.columns = df.columns.str.strip().str.lower()

    # Verificar que existan las columnas esperadas
    if "title" not in df.columns or "link" not in df.columns:
        print("ERROR: Tu CSV debe tener columnas 'Title' y 'Link' (o equivalente).")
        print("Columnas actuales:", df.columns.tolist())
        return

    abstracts = []

    for i, row in df.iterrows():
        title, link = row["title"], row["link"]
        print(f"🔍 ({i+1}/{len(df)}) Extrayendo abstract de: {title}")
        text = extract_abstract(link)
        abstracts.append(text)
        time.sleep(1)  # Pausa entre requests

    # Agregar la columna 'abstract'
    df["abstract"] = abstracts

    # Separar artículos exitosos y fallidos
    df_success = df.dropna(subset=["abstract"])
    df_failed = df[df["abstract"].isna()]

    # Crear carpetas si no existen
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(FAILED_OUTPUT_PATH), exist_ok=True)

    # Guardar resultados
    df_success.to_csv(OUTPUT_PATH, index=False)
    df_failed.to_csv(FAILED_OUTPUT_PATH, index=False)

    print(f"Limpieza completada. Archivo guardado en: {OUTPUT_PATH}")
    print(f"Total de artículos con abstract: {len(df_success)}")
    print(f"Artículos sin abstract guardados en: {FAILED_OUTPUT_PATH}")
    print(f"Total de artículos sin abstract: {len(df_failed)}")


if __name__ == "__main__":
    clean_data()
