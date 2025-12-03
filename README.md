# <img src="https://github.com/user-attachments/assets/53b684c2-5c1d-4108-bce7-d9eabc26f482" width="65"> BARKEDLOGY


**Barkedlogy** es una plataforma web inteligente que facilita la exploración de 572 artículos científicos de NASA sobre biología espacial. Utilizando técnicas avanzadas de minería de datos, el sistema organiza automáticamente los artículos en 19 categorías temáticas y descubre conexiones entre conceptos científicos.

## 💧 Vista Previa
 <p align="center">
  <a href="https://ramirochay.github.io/Barkedlogy_Searcher/" target="_blank">
    Entra aquí
  </a>
</p>

<img width="1897" height="967" alt="prueba2" src="https://github.com/user-attachments/assets/5aac32cb-d6f5-41c2-9245-cbccc8c880c0" />

## Índice
- [Vista Previa](#-vista-previa)
- [Características Principales](#-características-principales)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Metodología](#-metodología)
- [Resultados](#-resultados)
- [Cómo Usar](#-cómo-usar)
- [Instalación Local](#-instalación-local-desarrolladores)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Código del Proyecto](#-código-del-proyecto)
---

##  💧 Características Principales

### Búsqueda Inteligente
- Sugerencias automáticas de términos relacionados
- Basada en 10,995 reglas de asociación
- Descubre conexiones entre conceptos científicos

### Explorador de Categorías
- 19 categorías temáticas organizadas
- Imágenes generadas con IA para cada tema
- Visualización intuitiva y rápida

### Diseño Responsivo
- Funciona en computadoras, tablets y móviles
- Tiempo de carga menor a 1.5 segundos
- Tamaño optimizado (45 KB)

## 💧 Tecnologías Utilizadas

### Backend
- **FastAPI** (Python) - Servidor API
- **Scikit-learn** - Algoritmos de Machine Learning
- **Pandas** - Procesamiento de datos
- **Render** - Hosting del servidor

### Frontend
- **HTML5, CSS3, JavaScript** (Vanilla)
- **GitHub Pages** - Hosting gratuito
- Sin frameworks pesados para máxima velocidad

### Machine Learning
- **K-Means Clustering** - Organización en categorías
- **TF-IDF** - Vectorización de texto
- **FP-Growth** - Descubrimiento de asociaciones

## 💧 Metodología

### Limpieza de Datos
- Eliminación de 28 artículos duplicados (4.7%)
- Normalización de texto (lowercase, sin puntuación)
- Eliminación de palabras comunes (stop words)
- Lematización de términos

### Clustering Jerárquico
- **Nivel 1:** 5 grupos principales usando método del codo
- **Nivel 2:** División en 19 categorías específicas
- IDs asignados: 100-118

### Reglas de Asociación
- Algoritmo FP-Growth (eficiente en memoria)
- 10,995 reglas descubiertas
- Soporte mínimo: 4%
- Confianza > 70%, Lift > 1.5

## 💧 Resultados

### Rendimiento del Sistema
| Métrica | Resultado |
|---------|-----------|
| Tiempo de búsqueda | < 2 minutos |
| Respuesta del servidor | < 200 ms |
| Carga de la página | < 1.5 segundos |
| Mejora de velocidad | 15x más rápido |

### Organización del Contenido
- 572 artículos únicos organizados
- 19 categorías temáticas
- 10,995 conexiones descubiertas
- 18-47 artículos por categoría

## 💧 Cómo Usar

### Acceso Directo
Visita:  <a href="https://ramirochay.github.io/Barkedlogy_Searcher/" target="_blank"> BARKEDLOGY

### Búsqueda por Palabra Clave
1. Ingresa un término científico en la barra de búsqueda
2. Revisa las sugerencias de términos relacionados
3. Explora los artículos encontrados

### Exploración por Categorías
1. Navega a la sección de categorías
2. Selecciona una temática de interés
3. Lee los artículos agrupados

---

## 💧 Instalación Local (Desarrolladores)

### Requisitos Previos
```bash
Python 3.8+
pip
Git
```

### Backend
```bash
# Clonar repositorio
git clone https://github.com/karencardiel/space-biology-knowledge-engine.git
cd space-biology-knowledge-engine

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor
uvicorn main:app --reload
```

### Frontend
```bash
# Clonar repositorio
git clone https://github.com/ramirochay/Barkedlogy_Searcher.git
cd Barkedlogy_Searcher

# Abrir con servidor local
python -m http.server 8000
```

---

## 💧 Estructura del Proyecto


### Backend (ML y Limpieza de Datos)
```
space-biology-knowledge-engine/
│
├── analysis/
│   ├── check_duplicates.py
│   ├── consolidate_clusters.py
│   ├── evaluate_clustering.py
│   ├── name_clusters.py
│   └── elbow_method*.png
│
├── backend/
│   ├── models/
│   │   ├── apriori_rules.json
│   │   ├── kmeans_model.pkl
│   │   └── tfidf_vectorizer.pkl
│   └── src/
│       ├── api.py
│       ├── data_cleaning.py
│       ├── model_trainer.py
│       └── preprocess_abstracts.py
│
└── data/
    ├── SB_publication_PMC.csv
    ├── cleaned_articles.csv
    ├── final_dataset.csv
    └── cluster_assignments.csv
```

### Frontend
```
Barkedlogy_Searcher/
│
├── assets/
│   ├── clusters/          # Imágenes de categorías (100-118.jpg)
│   ├── docs/              # Capturas de pantalla
│   └── *.png              # Logos y recursos visuales
│
├── index.html
├── search_page.html
├── article.html
├── styles.css
└── script.js
```
---
## 💧 Código del Proyecto

- [Limpieza de datos y Machine Learning](https://github.com/karencardiel/space-biology-knowledge-engine/tree/feat/bradrobles)
- [Interfaz web (Frontend)](https://github.com/RamiroChay/Barkedlogy_Searcher)
  
---

<p align="center">
  <img width="464" height="91" alt="logo_largo" src="https://github.com/user-attachments/assets/c60408db-ee66-41fd-bd62-f8cd75c38934" />
</p>

