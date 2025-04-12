# Sistema de Clasificación de CVs

Este sistema analiza CVs y ofertas de trabajo para determinar si un candidato debe ser aceptado o rechazado para una posición específica.

## Características

- Procesamiento de documentos en múltiples formatos (PDF, DOCX, HTML, imágenes)
- Soporte multilingüe para análisis de texto
- Análisis semántico utilizando BERT
- Extracción de palabras clave relevantes
- Modelo de clasificación XGBoost
- Métricas de rendimiento y visualizaciones

## Configuración del Entorno

### 1. Crear un entorno virtual

#### Windows
```
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS
```
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias

```
pip install -r requirements.txt
```

### 3. Instalar modelos de spaCy

```
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

Para el OCR, necesitarás instalar Tesseract:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

## Estructura del Proyecto

- `main.py`: Punto de entrada principal
- `src/data_loader.py`: Carga de datos desde diferentes formatos
- `src/preprocessing.py`: Preprocesamiento de texto
- `src/feature_engineering.py`: Extracción de características con BERT
- `src/model.py`: Entrenamiento y evaluación del modelo XGBoost

## Uso

### 1. Activar el entorno virtual (si no está activado)

#### Windows
```
venv\Scripts\activate
```

#### Linux/macOS
```
source venv/bin/activate
```

### 2. Ejecutar el programa

```
python main.py
```

El programa utilizará las rutas predefinidas en la carpeta "data":
- applications.parquet: Datos de aplicaciones
- cvs/: Directorio con CVs
- jobs/: Directorio con ofertas de trabajo
- results/: Directorio donde se guardarán los resultados

## Resultados

El sistema genera:
- Matriz de confusión
- Informe de clasificación
- Distribución de afinidad
- Importancia de características
- Predicciones detalladas con porcentajes de afinidad
