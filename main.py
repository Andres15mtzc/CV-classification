import os
import argparse
import pandas as pd
from src.data_loader import load_applications, load_job_offers, load_cvs
from src.preprocessing import preprocess_documents
from src.feature_engineering import extract_features
from src.model import train_model, evaluate_model, predict

def main():
    parser = argparse.ArgumentParser(description='CV Classification System')
    parser.add_argument('--applications', type=str, required=True, help='Path to applications parquet file')
    parser.add_argument('--offers_dir', type=str, required=True, help='Directory containing job offers PDFs')
    parser.add_argument('--cv_dir', type=str, required=True, help='Directory containing applicant CVs')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--model_path', type=str, default=None, help='Path to saved model (for inference)')
    
    args = parser.parse_args()
    
    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Cargando datos...")
    applications_df = load_applications(args.applications)
    offers_dict = load_job_offers(args.offers_dir)
    cv_dict = load_cvs(args.cv_dir)
    
    print("Preprocesando documentos...")
    processed_offers = preprocess_documents(offers_dict)
    processed_cvs = preprocess_documents(cv_dict)
    
    print("Extrayendo características...")
    X, y, offer_ids, cv_ids = extract_features(
        applications_df, 
        processed_offers, 
        processed_cvs
    )
    
    if args.model_path is None:
        print("Entrenando modelo...")
        model, metrics = train_model(X, y)
        
        print("Evaluando modelo...")
        evaluate_model(model, X, y, offer_ids, cv_ids, args.output_dir)
        
        print(f"Métricas del modelo:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value}")
    else:
        print(f"Cargando modelo desde {args.model_path}...")
        # Implementar carga de modelo y predicción
        pass
    
    print(f"Resultados guardados en {args.output_dir}")

if __name__ == "__main__":
    main()
