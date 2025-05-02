import os
import pandas as pd
import xgboost as xgb
import argparse
import pickle
import sys
import nltk
import numpy as np
from src.data_loader import load_applications, load_job_offers, load_cvs
from src.preprocessing import preprocess_documents
from src.feature_engineering import extract_features
from src.model import train_model, evaluate_model, predict
import datetime

# Definición de rutas hardcodeadas
DATA_DIR = "data"
MODELS_DIR = "models"
APPLICATIONS_PATH = os.path.join(DATA_DIR, "applications.parquet")
CV_DIR = os.path.join(DATA_DIR, "cvs")
JOB_OFFERS_DIR = os.path.join(DATA_DIR, "jobs")
OUTPUT_DIR = os.path.join(DATA_DIR, "results")
MODEL_PATH = os.path.join(MODELS_DIR, f"cv_classifier_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
NLTK_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')

# Asegurar que los directorios necesarios existan
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# NLTK usará automáticamente las rutas del entorno de Python

def load_and_preprocess_data():
    """Carga y preprocesa los datos necesarios para el modelo."""
    print("Cargando datos...")
    applications_df = load_applications(APPLICATIONS_PATH)
    offers_dict = load_job_offers(JOB_OFFERS_DIR)
    cv_dict = load_cvs(CV_DIR)
    
    print("Preprocesando documentos...")
    processed_offers = preprocess_documents(offers_dict)
    processed_cvs = preprocess_documents(cv_dict)
    
    print("Extrayendo características...")
    X, y, offer_ids, cv_ids = extract_features(
        applications_df, 
        processed_offers, 
        processed_cvs
    )
    
    return X, y, offer_ids, cv_ids, processed_offers, processed_cvs

def create_model():
    """Crea y configura un modelo XGBoost."""
    print("Configurando modelo XGBoost...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric=['logloss', 'auc'],
        use_label_encoder=False,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0.1,
        reg_lambda=1,
        scale_pos_weight=1,
        random_state=42
    )
    return model

def train(args):
    """Función para entrenar el modelo."""
    X, y, offer_ids, cv_ids, _, _ = load_and_preprocess_data()
    
    # Verificar que hay suficientes datos para cada clase
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"Distribución de clases: {dict(zip(unique_classes, counts))}")
    
    if len(unique_classes) < 2 or min(counts) < 2:
        print("ADVERTENCIA: No hay suficientes ejemplos para cada clase. Generando datos sintéticos...")
        # Generar datos sintéticos adicionales si es necesario
        if 0 not in unique_classes or counts[list(unique_classes).index(0)] < 2:
            # Agregar ejemplos de clase 0
            for i in range(5):
                X = np.vstack([X, X[0] if len(X) > 0 else np.random.rand(1, X.shape[1] if len(X) > 0 else 4)])
                y = np.append(y, 0)
                offer_ids.append(f"synthetic_offer_0_{i}")
                cv_ids.append(f"synthetic_cv_0_{i}")
        
        if 1 not in unique_classes or counts[list(unique_classes).index(1)] < 2:
            # Agregar ejemplos de clase 1
            for i in range(5):
                X = np.vstack([X, X[0] if len(X) > 0 else np.random.rand(1, X.shape[1] if len(X) > 0 else 4)])
                y = np.append(y, 1)
                offer_ids.append(f"synthetic_offer_1_{i}")
                cv_ids.append(f"synthetic_cv_1_{i}")
        
        print(f"Datos después de generar ejemplos sintéticos: X shape={X.shape}, y shape={y.shape}")
        print(f"Nueva distribución de clases: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    model = create_model()
    
    print("Entrenando modelo...")
    model, metrics, _ = train_model(X, y, model=model)
    
    # Guardar el modelo entrenado
    print(f"Guardando modelo en {MODEL_PATH}")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Métricas del modelo:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value}")
    
    print(f"Modelo guardado en {MODEL_PATH}")

def test(args):
    """Función para evaluar el modelo."""
    X, y, offer_ids, cv_ids, _, _ = load_and_preprocess_data()
    
    # Cargar modelo si existe
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}. Ejecute primero el entrenamiento.")
        return
    
    print(f"Cargando modelo desde {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    print("Evaluando modelo...")
    evaluate_model(model, X, y, offer_ids, cv_ids)
    
    print(f"Resultados de evaluación guardados en {OUTPUT_DIR}")

def inference(args):
    """Función para realizar inferencias con el modelo."""
    # Verificar si se proporcionó un CV y una oferta de trabajo
    if args.cv_path is None or args.offer_id is None:
        print("Error: Debe proporcionar una ruta al archivo CV y un ID de oferta de trabajo para la inferencia.")
        return
    
    # Verificar si el archivo CV existe
    if not os.path.exists(args.cv_path):
        print(f"Error: El archivo CV en la ruta {args.cv_path} no existe.")
        return
    
    # Cargar modelo si existe
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}. Ejecute primero el entrenamiento.")
        return
    
    print(f"Cargando modelo desde {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # Cargar y preprocesar solo los datos necesarios
    print("Cargando datos para inferencia...")
    offers_dict = load_job_offers(JOB_OFFERS_DIR)
    
    # Determinar el tipo de archivo y extraer texto
    _, file_extension = os.path.splitext(args.cv_path)
    file_extension = file_extension.lower()
    
    from src.data_loader import extract_text_from_pdf, extract_text_from_docx, extract_text_from_html, extract_text_from_image
    
    print(f"Procesando archivo CV: {args.cv_path}")
    cv_text = None
    if file_extension == '.pdf':
        cv_text = extract_text_from_pdf(args.cv_path)
    elif file_extension in ['.doc', '.docx']:
        cv_text = extract_text_from_docx(args.cv_path)
    elif file_extension in ['.html', '.htm']:
        cv_text = extract_text_from_html(args.cv_path)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
        cv_text = extract_text_from_image(args.cv_path)
    else:
        print(f"Error: Formato de archivo no soportado: {file_extension}")
        return
    
    if not cv_text:
        print("Error: No se pudo extraer texto del archivo CV.")
        return
    
    # Generar un ID temporal para el CV
    cv_id = f"temp_cv_{os.path.basename(args.cv_path)}"
    
    if args.offer_id not in offers_dict:
        print(f"Error: Oferta de trabajo con ID {args.offer_id} no encontrada.")
        return
    
    print("Preprocesando documentos...")
    processed_offers = preprocess_documents({args.offer_id: offers_dict[args.offer_id]})
    processed_cvs = preprocess_documents({cv_id: cv_text})
    
    # Crear un DataFrame de aplicaciones simulado para la inferencia
    applications_df = pd.DataFrame({
        'cv_id': [cv_id],
        'offer_id': [args.offer_id],
        'match': [0]  # Valor ficticio, no se usa para inferencia
    })
    
    print("Extrayendo características...")
    X, _, offer_ids, cv_ids = extract_features(
        applications_df, 
        processed_offers, 
        processed_cvs
    )
    
    print("Realizando predicción...")
    predictions = predict(model, X, offer_ids, cv_ids)
    
    # Mostrar resultado
    prediction = predictions.iloc[0]
    print("\nResultado de la predicción:")
    print(f"CV ID: {prediction['cv_id']}")
    print(f"Oferta ID: {prediction['offer_id']}")
    print(f"Probabilidad de coincidencia: {prediction['probability']:.2f}")
    print(f"Predicción: {'Coincide' if prediction['prediction'] == 1 else 'No coincide'}")

def main():
    parser = argparse.ArgumentParser(description='Sistema de clasificación de CVs')
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Subparser para entrenamiento
    train_parser = subparsers.add_parser('train', help='Entrenar el modelo')
    
    # Subparser para pruebas
    test_parser = subparsers.add_parser('test', help='Evaluar el modelo')
    
    # Subparser para inferencia
    inference_parser = subparsers.add_parser('inference', help='Realizar inferencia con el modelo')
    inference_parser.add_argument('--cv-path', type=str, help='Ruta al archivo CV para inferencia')
    inference_parser.add_argument('--offer-id', type=str, help='ID de la oferta de trabajo para inferencia')
    
    # Subparser para inicializar NLTK
    init_parser = subparsers.add_parser('init', help='Inicializar recursos NLTK')
    
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    print(f"Comando ejecutado: {args.command}")
    
    # Si el comando es init, ejecutar la inicialización de NLTK
    if args.command == 'init':
        try:
            from init_nltk import download_nltk_resources
            download_nltk_resources()
            return
        except ImportError:
            print("Error: No se pudo importar el módulo init_nltk.")
            return
    
    # Verificar si los recursos NLTK están disponibles antes de continuar
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Los recursos de NLTK no están disponibles en el entorno global. Ejecutando inicialización automática...")
        try:
            from init_nltk import download_nltk_resources
            download_nltk_resources()
        except ImportError:
            print("Error: No se pudo importar el módulo init_nltk.")
            return
    
    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
    elif args.command == 'inference':
        inference(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
