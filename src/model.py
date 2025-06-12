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

def handle_class_imbalance(X, y, method='smote', random_state=42):
    """
    Handles class imbalance in the dataset.
    
    Args:
        X: Feature matrix
        y: Target vector
        method: Resampling method ('smote', 'adasyn', 'random_over', 'random_under')
        random_state: Random seed
        
    Returns:
        X_resampled, y_resampled: Resampled dataset
    """
    # Check if we need to handle imbalance
    class_counts = np.bincount(y)
    if len(class_counts) < 2 or min(class_counts) / max(class_counts) > 0.4:
        logger.info("Class distribution is relatively balanced. No resampling needed.")
        return X, y
        
    logger.info(f"Original class distribution: {class_counts}")
    
    try:
        if method == 'smote':
            from imblearn.over_sampling import SMOTE
            resampler = SMOTE(random_state=random_state)
        elif method == 'adasyn':
            from imblearn.over_sampling import ADASYN
            resampler = ADASYN(random_state=random_state)
        elif method == 'random_over':
            from imblearn.over_sampling import RandomOverSampler
            resampler = RandomOverSampler(random_state=random_state)
        elif method == 'random_under':
            from imblearn.under_sampling import RandomUnderSampler
            resampler = RandomUnderSampler(random_state=random_state)
        else:
            logger.warning(f"Unknown resampling method: {method}. Using SMOTE.")
            from imblearn.over_sampling import SMOTE
            resampler = SMOTE(random_state=random_state)
            
        X_resampled, y_resampled = resampler.fit_resample(X, y)
        
        # Log new class distribution
        new_class_counts = np.bincount(y_resampled)
        logger.info(f"Resampled class distribution: {new_class_counts}")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        logger.error(f"Error in resampling: {e}")
        return X, y

def adjust_feature_importance(model, feature_names):
    """
    Adjusts feature importance to reduce the weight of soft skills features.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        
    Returns:
        model: Model with adjusted feature importance
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model doesn't have feature_importances_ attribute. Skipping adjustment.")
        return model
    
    # Import soft skills list from feature_engineering
    try:
        from feature_engineering import SOFT_SKILLS_WORDS
        
        # Identify features related to soft skills
        soft_skills_indices = []
        for i, feature in enumerate(feature_names):
            if any(skill in feature.lower() for skill in SOFT_SKILLS_WORDS):
                soft_skills_indices.append(i)
        
        # Log the number of soft skills features found
        logger.info(f"Found {len(soft_skills_indices)} soft skills related features")
        
        # If we found any soft skills features, adjust their importance
        if soft_skills_indices:
            # Reduce importance of soft skills features by 70%
            for idx in soft_skills_indices:
                if idx < len(model.feature_importances_):
                    model.feature_importances_[idx] *= 0.3
            
            # Normalize feature importances to sum to 1
            total_importance = sum(model.feature_importances_)
            if total_importance > 0:
                model.feature_importances_ = model.feature_importances_ / total_importance
                
            logger.info("Adjusted feature importances to reduce soft skills weight")
    except Exception as e:
        logger.error(f"Error adjusting feature importance: {e}")
    
    return model

def train_model(X, y, model=None, test_size=0.2, random_state=42, feature_selection=True, handle_imbalance=True):
    """
    Entrena un modelo XGBoost para clasificación de CVs.
    
    Args:
        X: Matriz de características
        y: Vector de etiquetas
        model: Modelo XGBoost preconfigurado (opcional)
        test_size: Proporción del conjunto de prueba
        random_state: Semilla aleatoria
        
    Returns:
        model: Modelo entrenado
        metrics: Diccionario con métricas de rendimiento
        data_splits: Diccionario con los conjuntos de datos divididos
    """
    # Verificar que hay suficientes datos para entrenar
    if len(X) < 10:
        logger.warning("Conjunto de datos muy pequeño para entrenar. Se necesitan al menos 10 muestras.")
        raise ValueError("Conjunto de datos muy pequeño para entrenar")
    
    # Verificar que hay muestras de ambas clases
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        logger.warning(f"Solo hay muestras de la clase {unique_classes[0]}. Se necesitan muestras de ambas clases.")
        raise ValueError("Se necesitan muestras de ambas clases para entrenar")
    
    # Calcular el peso para balancear las clases
    class_counts = np.bincount(y)
    if len(class_counts) > 1:
        scale_pos_weight = class_counts[0] / class_counts[1]
    else:
        scale_pos_weight = 1.0
    
    logger.info(f"Distribución de clases: {class_counts}")
    logger.info(f"scale_pos_weight: {scale_pos_weight}")
    
    # Handle class imbalance if enabled
    if handle_imbalance:
        X, y = handle_class_imbalance(X, y, method='smote', random_state=random_state)
    
    # Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Crear un conjunto de validación separado
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=random_state, stratify=y_train
    )
    
    # Feature selection if enabled and we have enough features
    if feature_selection and X.shape[1] > 3:
        from sklearn.feature_selection import SelectFromModel
        
        # Train a preliminary model to get feature importances
        selector_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            random_state=random_state
        )
        selector_model.fit(X_train, y_train)
        
        # Select important features
        selector = SelectFromModel(selector_model, threshold='mean', prefit=True)
        X_train = selector.transform(X_train)
        X_val = selector.transform(X_val)
        X_test = selector.transform(X_test)
        
        logger.info(f"Feature selection reduced features from {X.shape[1]} to {X_train.shape[1]}")
    
    # Configurar modelo XGBoost si no se proporciona uno
    if model is None:
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric=['logloss', 'auc'],  # Track multiple metrics
            use_label_encoder=False,
            n_estimators=200,  # More trees for better performance
            max_depth=5,  # Slightly deeper trees
            learning_rate=0.05,  # Lower learning rate for better generalization
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,  # Add column sampling by level
            min_child_weight=2,  # Increase to prevent overfitting
            gamma=0.1,  # Minimum loss reduction for further partition
            reg_alpha=0.2,  # Increase L1 regularization
            reg_lambda=1.2,  # Increase L2 regularization
            scale_pos_weight=scale_pos_weight,  # Usar el valor calculado
            random_state=random_state,
            early_stopping_rounds=20  # More rounds before stopping
        )
    
    # Entrenar modelo con early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  # Solo usar conjunto de validación
        verbose=False
    )
    
    # Adjust feature importance to reduce weight of soft skills
    try:
        # Get feature names if available
        if hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
            feature_names = model.get_booster().feature_names
        else:
            # Create generic feature names
            feature_names = [f'f{i}' for i in range(X_train.shape[1])]
        
        # Adjust feature importance
        model = adjust_feature_importance(model, feature_names)
    except Exception as e:
        logger.error(f"Error in feature importance adjustment: {e}")
    
    # Registrar el mejor número de iteraciones si está disponible
    if hasattr(model, 'best_iteration'):
        best_iteration = model.best_iteration
        logger.info(f"Mejor iteración: {best_iteration}")
    else:
        logger.info("Early stopping no activado o no se encontró la mejor iteración")
    
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
    
    # Validación cruzada con múltiples métricas
    from sklearn.model_selection import cross_validate
    
    cv_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(
        model, X, y, 
        cv=5, 
        scoring=cv_metrics,
        return_train_score=True
    )
    
    # Add cross-validation results to metrics
    for metric in cv_metrics:
        test_key = f'test_{metric}'
        if test_key in cv_results:
            metrics[f'cv_{metric}_mean'] = cv_results[test_key].mean()
            metrics[f'cv_{metric}_std'] = cv_results[test_key].std()
            
    # Check for overfitting by comparing train and test scores
    for metric in cv_metrics:
        train_key = f'train_{metric}'
        test_key = f'test_{metric}'
        if train_key in cv_results and test_key in cv_results:
            train_score = cv_results[train_key].mean()
            test_score = cv_results[test_key].mean()
            metrics[f'cv_{metric}_gap'] = train_score - test_score
    
    # Crear diccionario con los conjuntos de datos
    data_splits = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }
    
    return model, metrics, data_splits

def tune_hyperparameters(X, y, cv=5, random_state=42):
    """
    Performs hyperparameter tuning for the XGBoost model.
    
    Args:
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        best_params: Dictionary with best hyperparameters
    """
    from sklearn.model_selection import RandomizedSearchCV
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 2, 3],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 1.5]
    }
    
    # Create base model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        random_state=random_state
    )
    
    # Set up RandomizedSearchCV
    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter settings sampled
        scoring='f1',
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Perform search
    logger.info("Starting hyperparameter tuning...")
    search.fit(X, y)
    
    # Log results
    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best score: {search.best_score_:.4f}")
    
    return search.best_params_

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
        # Verificar que hay suficientes datos para dividir
        if len(X) < 10:
            logger.warning("Conjunto de datos muy pequeño para evaluar. Se usará todo el conjunto.")
            X_test, y_test = X, y
            offer_ids_test, cv_ids_test = offer_ids, cv_ids
        else:
            # Dividir datos en entrenamiento, validación y prueba
            try:
                X_temp, X_test, y_temp, y_test, offer_ids_temp, offer_ids_test, cv_ids_temp, cv_ids_test = train_test_split(
                    X, y, offer_ids, cv_ids, test_size=0.15, random_state=42, stratify=y
                )
                
                X_train, X_val, y_train, y_val, offer_ids_train, offer_ids_val, cv_ids_train, cv_ids_val = train_test_split(
                    X_temp, y_temp, offer_ids_temp, cv_ids_temp, test_size=0.15/(1-0.15), random_state=42, stratify=y_temp
                )
            except ValueError as e:
                logger.warning(f"Error al dividir datos: {e}. Se usará todo el conjunto.")
                X_test, y_test = X, y
                offer_ids_test, cv_ids_test = offer_ids, cv_ids
    else:
        # Usar los conjuntos de datos proporcionados
        X_test, y_test = data_splits['X_test'], data_splits['y_test']
        
        # Necesitamos dividir los IDs de ofertas y CVs de la misma manera
        # Esto es una simplificación, en un caso real deberíamos mantener la correspondencia
        try:
            _, offer_ids_test, _, cv_ids_test = train_test_split(
                X, offer_ids, test_size=0.15, random_state=42, stratify=y
            )
        except ValueError as e:
            logger.warning(f"Error al dividir IDs: {e}. Se usarán todos los IDs.")
            offer_ids_test, cv_ids_test = offer_ids, cv_ids
    
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
    try:
        # Verificar que hay muestras de ambas clases para calcular métricas
        if len(np.unique(y_test)) < 2:
            logger.warning("No hay muestras de ambas clases en el conjunto de prueba. Algunas métricas no se calcularán.")
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'samples_count': len(y_test),
                'class_distribution': np.bincount(y_test).tolist()
            }
        else:
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_prob),
                'samples_count': len(y_test),
                'class_distribution': np.bincount(y_test).tolist()
            }
    except Exception as e:
        logger.error(f"Error al calcular métricas: {e}")
        metrics = {
            'error': str(e),
            'samples_count': len(y_test)
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
