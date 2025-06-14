# Sistema de Clasificación de CVs

Este sistema permite clasificar Currículum Vitae (CVs) según su afinidad con ofertas de trabajo, utilizando técnicas de procesamiento de lenguaje natural y aprendizaje automático.

## Características

- Procesamiento de CVs en múltiples formatos (PDF, DOCX, HTML, imágenes)
- Soporte multilingüe para análisis de texto
- Análisis semántico utilizando TF-IDF
- Extracción de palabras clave relevantes
- Modelo de clasificación XGBoost
- Interfaz gráfica optimizada para selección de ofertas y carga de CVs
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

### 3. Instalar modelos de spaCy y recursos de NLTK

Puedes instalar todos los recursos necesarios ejecutando:

```
python main.py init
```

O instalarlos manualmente:

```
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader wordnet
python -m nltk.downloader omw-1.4
```

Para el OCR, necesitarás instalar Tesseract:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

## Estructura del Proyecto

- `main.py`: Punto de entrada principal
- `init_nltk.py`: Inicialización de recursos NLTK en el entorno global
- `init_resources.py`: Script para inicializar todos los recursos necesarios
- `src/data_loader.py`: Carga de datos desde diferentes formatos
- `src/preprocessing.py`: Preprocesamiento de texto
- `src/feature_engineering.py`: Extracción de características
- `src/model.py`: Entrenamiento y evaluación del modelo XGBoost
- `src/gui.py`: Interfaz gráfica para selección de ofertas y carga de CVs

## Uso

### Interfaz de Usuario

Para iniciar la aplicación y seleccionar la interfaz deseada:

```
python run.py
```

Este comando abrirá un menú que permite elegir entre:
1. Interfaz Gráfica (GUI)
2. Interfaz Web
3. Salir

La interfaz permite:
- Seleccionar una oferta de trabajo de la lista disponible
- Cargar un CV desde el sistema de archivos
- Ver detalles de la oferta seleccionada
- Ejecutar el análisis de compatibilidad
- Ver el resultado con el porcentaje de afinidad

#### Características de la Interfaz:
- Carga de ofertas en hilos separados para evitar bloqueos
- Extracción inteligente de títulos de ofertas
- Procesamiento por lotes para mejor rendimiento
- Interfaz más responsiva durante operaciones intensivas

### Línea de Comandos (para entrenamiento y pruebas)

#### 1. Activar el entorno virtual (si no está activado)

#### Windows
```
venv\Scripts\activate
```

#### Linux/macOS
```
source venv/bin/activate
```

#### 2. Entrenar el modelo

```
python main.py train
```

#### 3. Evaluar el modelo

```
python main.py test
```

#### 4. Realizar inferencia con un CV específico

```
python main.py inference --cv-path ruta/al/cv.pdf --offer-id id_oferta
```

Nota: Para las interfaces gráfica y web, utilice siempre `python run.py` en lugar de acceder directamente a los módulos individuales.

## Directorios de Datos

El programa utilizará las siguientes rutas predefinidas:
- `data/applications.parquet`: Datos de aplicaciones
- `data/cvs/`: Directorio con CVs
- `data/jobs/`: Directorio con ofertas de trabajo
- `data/results/`: Directorio donde se guardarán los resultados
- `models/`: Directorio donde se guardarán los modelos entrenados

Nota: Al ejecutar `run.py`, estos directorios se crearán automáticamente si no existen.

## Resultados

El sistema genera:
- Matriz de confusión
- Informe de clasificación
- Distribución de afinidad
- Importancia de características
- Predicciones detalladas con porcentajes de afinidad

## Notas de la Última Actualización

- Optimización del rendimiento de la interfaz gráfica
- Mejora en la extracción de títulos de ofertas de trabajo
- Procesamiento asíncrono para evitar bloqueos de la interfaz
- Corrección de errores en el manejo de variables en funciones lambda
- Uso de recursos NLTK desde el entorno global de Python
