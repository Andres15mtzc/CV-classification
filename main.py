import os
import pandas as pd
from src.data_loader import load_applications, load_job_offers, load_cvs
from src.preprocessing import preprocess_documents
from src.feature_engineering import extract_features
from src.model import train_model, evaluate_model, predict

# Definición de rutas hardcodeadas
DATA_DIR = "data"
APPLICATIONS_PATH = os.path.join(DATA_DIR, "applications.parquet")
CV_DIR = os.path.join(DATA_DIR, "cvs")
JOB_OFFERS_DIR = os.path.join(DATA_DIR, "jobs")
OUTPUT_DIR = os.path.join(DATA_DIR, "results")
MODEL_PATH = os.path.join(OUTPUT_DIR, "cv_classifier_model.pkl")

# Asegurar que los directorios existan
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CV_DIR, exist_ok=True)
os.makedirs(JOB_OFFERS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    
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
    
    print("Entrenando modelo...")
    model, metrics, data_splits = train_model(X, y)
    
    print("Evaluando modelo...")
    evaluate_model(model, X, y, offer_ids, cv_ids, OUTPUT_DIR, data_splits)
    
    print(f"Métricas del modelo:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value}")
    
    print(f"Resultados guardados en {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
