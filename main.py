import os
import pandas as pd
import xgboost as xgb
import argparse
import pickle
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
MODEL_PATH = os.path.join(MODELS_DIR, f"cv_classifier_model_{datetime.datetime.now()}_.pkl")

# Asegurar que el directorio de resultados exista
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    if args.cv_id is None or args.offer_id is None:
        print("Error: Debe proporcionar un ID de CV y un ID de oferta de trabajo para la inferencia.")
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
    cv_dict = load_cvs(CV_DIR)
    
    if args.cv_id not in cv_dict:
        print(f"Error: CV con ID {args.cv_id} no encontrado.")
        return
    
    if args.offer_id not in offers_dict:
        print(f"Error: Oferta de trabajo con ID {args.offer_id} no encontrada.")
        return
    
    print("Preprocesando documentos...")
    processed_offers = preprocess_documents({args.offer_id: offers_dict[args.offer_id]})
    processed_cvs = preprocess_documents({args.cv_id: cv_dict[args.cv_id]})
    
    # Crear un DataFrame de aplicaciones simulado para la inferencia
    applications_df = pd.DataFrame({
        'cv_id': [args.cv_id],
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
    inference_parser.add_argument('--cv-id', type=str, help='ID del CV para inferencia')
    inference_parser.add_argument('--offer-id', type=str, help='ID de la oferta de trabajo para inferencia')
    
    args = parser.parse_args()
    
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
