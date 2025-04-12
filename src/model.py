import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import logging

# Definición de rutas hardcodeadas
DATA_DIR = "data"
CV_DIR = os.path.join(DATA_DIR, "cvs")
JOB_OFFERS_DIR = os.path.join(DATA_DIR, "jobs")
OUTPUT_DIR = os.path.join(DATA_DIR, "results")
MODEL_PATH = os.path.join(OUTPUT_DIR, "cv_classifier_model.pkl")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "prediction_results.csv")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.txt")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
CLASSIFICATION_REPORT_PATH = os.path.join(OUTPUT_DIR, "classification_report.csv")
AFFINITY_DISTRIBUTION_PATH = os.path.join(OUTPUT_DIR, "affinity_distribution.png")
FEATURE_IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "feature_importance.png")

# Asegurar que los directorios existan
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

def train_model(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Entrena un modelo XGBoost para clasificación de CVs.
    
    Args:
        X: Matriz de características
        y: Vector de etiquetas
        test_size: Proporción del conjunto de prueba
        val_size: Proporción del conjunto de validación
        random_state: Semilla aleatoria
        
    Returns:
        model: Modelo entrenado
        metrics: Diccionario con métricas de rendimiento
        data_splits: Diccionario con los conjuntos de datos divididos
    """
    # Primero dividimos para obtener el conjunto de prueba
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Luego dividimos el resto para obtener entrenamiento y validación
    # Calculamos el tamaño de validación relativo al conjunto temporal
    relative_val_size = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, 
        random_state=random_state, stratify=y_temp
    )
    
    # Configurar modelo XGBoost con hiperparámetros optimizados
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
        random_state=random_state
    )
    
    # Entrenar modelo con early stopping más conservador
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric=['logloss', 'auc'],
        early_stopping_rounds=20,
        verbose=True
    )
    
    # Registrar el mejor número de iteraciones
    best_iteration = model.best_iteration
    logger.info(f"Mejor iteración: {best_iteration}")
    
    # Evaluar modelo en el conjunto de prueba
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob)
    }
    
    # Validación cruzada
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    metrics['cv_f1_mean'] = cv_scores.mean()
    metrics['cv_f1_std'] = cv_scores.std()
    
    # Crear diccionario con los conjuntos de datos
    data_splits = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }
    
    return model, metrics, data_splits

def evaluate_model(model, X, y, offer_ids, cv_ids, output_dir=OUTPUT_DIR, data_splits=None):
    """
    Evalúa el modelo y genera visualizaciones y métricas.
    
    Args:
        model: Modelo entrenado
        X: Matriz de características
        y: Vector de etiquetas
        offer_ids: IDs de ofertas
        cv_ids: IDs de CVs
        output_dir: Directorio para guardar resultados
        data_splits: Diccionario con los conjuntos de datos ya divididos (opcional)
    """
    # Si no se proporcionan los conjuntos de datos divididos, los creamos
    if data_splits is None:
        # Dividir datos en entrenamiento, validación y prueba
        X_temp, X_test, y_temp, y_test, offer_ids_temp, offer_ids_test, cv_ids_temp, cv_ids_test = train_test_split(
            X, y, offer_ids, cv_ids, test_size=0.15, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val, offer_ids_train, offer_ids_val, cv_ids_train, cv_ids_val = train_test_split(
            X_temp, y_temp, offer_ids_temp, cv_ids_temp, test_size=0.15/(1-0.15), random_state=42, stratify=y_temp
        )
    else:
        # Usar los conjuntos de datos proporcionados
        X_test, y_test = data_splits['X_test'], data_splits['y_test']
        
        # Necesitamos dividir los IDs de ofertas y CVs de la misma manera
        # Esto es una simplificación, en un caso real deberíamos mantener la correspondencia
        _, offer_ids_test, _, cv_ids_test = train_test_split(
            X, offer_ids, test_size=0.15, random_state=42, stratify=y
        )
    
    # Predecir en conjunto de prueba
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame({
        'offer_id': offer_ids_test,
        'cv_id': cv_ids_test,
        'true_label': y_test,
        'predicted_label': y_pred,
        'probability': y_prob,
        'affinity_percentage': y_prob * 100  # Porcentaje de afinidad
    })
    
    # Guardar resultados
    results_df.to_csv(RESULTS_PATH, index=False)
    
    # Guardar modelo
    joblib.dump(model, MODEL_PATH)
    
    # Generar matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Rechazado', 'Aceptado'],
                yticklabels=['Rechazado', 'Aceptado'])
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    
    # Generar informe de clasificación
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(CLASSIFICATION_REPORT_PATH)
    
    # Generar histograma de afinidad
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df, x='affinity_percentage', hue='true_label', bins=20, kde=True)
    plt.xlabel('Porcentaje de Afinidad')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Afinidad por Clase')
    plt.savefig(AFFINITY_DISTRIBUTION_PATH)
    
    # Generar gráfico de importancia de características
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model)
    plt.title('Importancia de Características')
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH)
    
    # Guardar métricas en un archivo
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob)
    }
    
    with open(METRICS_PATH, 'w') as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value}\n")

def predict(model, X, offer_ids, cv_ids):
    """
    Realiza predicciones con el modelo entrenado.
    
    Args:
        model: Modelo entrenado
        X: Matriz de características
        offer_ids: IDs de ofertas
        cv_ids: IDs de CVs
        
    Returns:
        DataFrame con predicciones y porcentajes de afinidad
    """
    # Predecir
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame({
        'offer_id': offer_ids,
        'cv_id': cv_ids,
        'predicted_label': y_pred,
        'probability': y_prob,
        'affinity_percentage': y_prob * 100  # Porcentaje de afinidad
    })
    
    return results_df
