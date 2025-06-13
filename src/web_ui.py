import os
import sys
import logging
import threading
import pickle
import pandas as pd
import re
import uuid
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
# Eliminamos la dependencia de flask_bootstrap que podría no estar instalada
from werkzeug.utils import secure_filename

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Asegurar que podemos importar desde el directorio raíz
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Definir rutas
DATA_DIR = "data"
JOB_OFFERS_DIR = os.path.join(DATA_DIR, "jobs")
MODELS_DIR = "models"
UPLOAD_FOLDER = os.path.join(DATA_DIR, "uploads")
ALLOWED_EXTENSIONS = {'html', 'htm'}  # Solo permitimos archivos HTML

# Crear directorios si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Crear la aplicación Flask con la configuración correcta de plantillas
app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates'),
           static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static'))
app.config['SECRET_KEY'] = 'clave-secreta-para-cvq-qualification'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max
# Intentamos inicializar Bootstrap solo si está disponible
try:
    from flask_bootstrap import Bootstrap
    Bootstrap(app)
    logger.info("Flask-Bootstrap inicializado correctamente")
except ImportError:
    logger.warning("Flask-Bootstrap no está instalado. La UI podría verse afectada.")

# Función para verificar extensiones permitidas
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Función para cargar ofertas de trabajo
def load_job_offers():
    try:
        from src.data_loader import load_job_offers as load_offers
        offers_dict = load_offers(JOB_OFFERS_DIR)
        
        # Procesar ofertas para mostrar en la interfaz
        processed_offers = []
        for offer_id, offer_text in offers_dict.items():
            title = extract_title(offer_text, offer_id)
            processed_offers.append({
                'id': offer_id,
                'title': title,
                'text': offer_text[:500] + "..." if len(offer_text) > 500 else offer_text
            })
        
        # Ordenar por título (que ahora es el número de oferta)
        processed_offers.sort(key=lambda x: x['title'])
        return processed_offers
    except Exception as e:
        logger.error(f"Error al cargar ofertas: {str(e)}")
        return []

# Función para extraer número de oferta del nombre del archivo o generar uno nuevo
def extract_offer_number(filename):
    """Extrae el número de oferta del nombre del archivo o genera uno nuevo"""
    # Intentar extraer un número de oferta del nombre del archivo
    match = re.search(r'oferta[_-]?(\d+)', filename.lower())
    if match:
        return f"Oferta #{match.group(1)}"
    
    # Si no se encuentra, generar un número único
    return f"Oferta #{uuid.uuid4().hex[:6].upper()}"

# Función para extraer título de la oferta (ahora devuelve el número de oferta)
def extract_title(text, offer_id):
    """Devuelve el número de oferta como título"""
    # Intentar extraer un número de oferta del ID
    match = re.search(r'oferta[_-]?(\d+)', offer_id.lower())
    if match:
        return f"Oferta #{match.group(1)}"
    
    # Si el ID no contiene un número de oferta, usar el ID como número
    return f"Oferta #{offer_id}"

# Función para ejecutar la inferencia
def run_inference(cv_path, offer_id):
    try:
        # Buscar el modelo más reciente
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        if not model_files:
            return {"error": "No se encontró ningún modelo entrenado"}
        
        # Ordenar por fecha de modificación (más reciente primero)
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)), reverse=True)
        model_path = os.path.join(MODELS_DIR, model_files[0])
        
        # Importar funciones necesarias
        from src.data_loader import extract_text_from_pdf, extract_text_from_docx, extract_text_from_html, extract_text_from_image
        from src.preprocessing import preprocess_documents
        from src.feature_engineering import extract_features
        from src.model import predict
        
        # Determinar el tipo de archivo y extraer texto
        _, file_extension = os.path.splitext(cv_path)
        file_extension = file_extension.lower()
        
        cv_text = None
        if file_extension == '.pdf':
            cv_text = extract_text_from_pdf(cv_path)
        elif file_extension in ['.doc', '.docx']:
            cv_text = extract_text_from_docx(cv_path)
        elif file_extension in ['.html', '.htm']:
            cv_text = extract_text_from_html(cv_path)
        elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            cv_text = extract_text_from_image(cv_path)
        elif file_extension == '.txt':
            with open(cv_path, 'r', encoding='utf-8', errors='ignore') as f:
                cv_text = f.read()
        else:
            return {"error": f"Formato de archivo no soportado: {file_extension}"}
        
        if not cv_text:
            return {"error": "No se pudo extraer texto del archivo CV"}
        
        # Cargar ofertas
        from src.data_loader import load_job_offers as load_offers
        offers_dict = load_offers(JOB_OFFERS_DIR)
        
        if offer_id not in offers_dict:
            return {"error": f"Oferta de trabajo con ID {offer_id} no encontrada"}
        
        # Generar un ID temporal para el CV
        cv_id = f"temp_cv_{os.path.basename(cv_path)}"
        
        # Preprocesar documentos
        processed_offers = preprocess_documents({offer_id: offers_dict[offer_id]})
        processed_cvs = preprocess_documents({cv_id: cv_text})
        
        # Crear un DataFrame de aplicaciones simulado para la inferencia
        applications_df = pd.DataFrame({
            'cv_id': [cv_id],
            'offer_id': [offer_id],
            'match': [0]  # Valor ficticio, no se usa para inferencia
        })
        
        # Extraer características
        X, _, offer_ids, cv_ids = extract_features(
            applications_df, 
            processed_offers, 
            processed_cvs
        )
        
        # Cargar modelo
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Realizar predicción
        predictions = predict(model, X, offer_ids, cv_ids)
        
        # Obtener resultado
        prediction = predictions.iloc[0]
        probability = float(prediction['probability'])
        is_match = bool(prediction['predicted_label'] == 1)
        
        return {
            "success": True,
            "probability": probability,
            "is_match": is_match,
            "percentage": f"{probability * 100:.2f}%",
            "result": "COINCIDE" if is_match else "NO COINCIDE"
        }
        
    except Exception as e:
        logger.error(f"Error en la inferencia: {str(e)}")
        return {"error": f"Error en la inferencia: {str(e)}"}

# Rutas de la aplicación
@app.route('/')
def index():
    offers = load_job_offers()
    return render_template('index.html', offers=offers)

@app.route('/upload_offer', methods=['GET', 'POST'])
def upload_offer():
    if request.method == 'POST':
        # Verificar si se subió un archivo
        if 'offer_file' not in request.files:
            flash('No se seleccionó ningún archivo', 'warning')
            return redirect(request.url)
        
        file = request.files['offer_file']
        
        if file.filename == '':
            flash('No se seleccionó ningún archivo', 'warning')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generar un número de oferta basado en el nombre del archivo o crear uno nuevo
            offer_number = extract_offer_number(file.filename)
            # Generar un ID único para la oferta
            offer_id = f"oferta_{offer_number.replace('#', '')}"
            
            # Guardar el archivo HTML directamente
            filename = secure_filename(f"{offer_id}.html")
            file_path = os.path.join(JOB_OFFERS_DIR, filename)
            
            # Asegurar que el directorio existe
            os.makedirs(JOB_OFFERS_DIR, exist_ok=True)
            
            # Guardar el archivo
            file.save(file_path)
            
            try:
                # Extraer texto del archivo HTML
                from src.data_loader import extract_text_from_html
                offer_text = extract_text_from_html(file_path)
                
                # Si se pudo extraer texto, guardar en un archivo de texto plano
                if offer_text:
                    text_file_path = os.path.join(JOB_OFFERS_DIR, f"{offer_id}.txt")
                    with open(text_file_path, 'w', encoding='utf-8') as f:
                        f.write(offer_text)
                    
                    flash('Oferta de trabajo subida correctamente', 'success')
                    return redirect(url_for('index'))
                else:
                    flash('No se pudo extraer texto del archivo HTML', 'danger')
                    return redirect(request.url)
                
            except Exception as e:
                logger.error(f"Error al procesar archivo de oferta: {str(e)}")
                flash(f'Error al procesar el archivo: {str(e)}', 'danger')
                return redirect(request.url)
        else:
            flash('Solo se permiten archivos HTML', 'danger')
            return redirect(request.url)
    
    return render_template('upload_offer.html')

@app.route('/offer/<offer_id>')
def offer_detail(offer_id):
    try:
        from src.data_loader import load_job_offers as load_offers
        offers_dict = load_offers(JOB_OFFERS_DIR)
        
        if offer_id not in offers_dict:
            flash('Oferta no encontrada', 'danger')
            return redirect(url_for('index'))
        
        offer_text = offers_dict[offer_id]
        title = extract_title(offer_text, offer_id)
        
        return render_template('offer_detail.html', 
                              offer_id=offer_id, 
                              title=title, 
                              text=offer_text)
    except Exception as e:
        flash(f'Error al cargar la oferta: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        # Verificar si se seleccionó una oferta
        offer_id = request.form.get('offer_id')
        if not offer_id:
            flash('Por favor seleccione una oferta de trabajo', 'warning')
            return redirect(url_for('analyze'))
        
        # Verificar si se subió un archivo
        if 'cv_file' not in request.files:
            flash('No se seleccionó ningún archivo', 'warning')
            return redirect(request.url)
        
        file = request.files['cv_file']
        
        if file.filename == '':
            flash('No se seleccionó ningún archivo', 'warning')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Ejecutar inferencia en un hilo separado
            result = run_inference(file_path, offer_id)
            
            if "error" in result:
                flash(result["error"], 'danger')
                return redirect(url_for('analyze'))
            
            # Guardar resultado en sesión
            session['result'] = result
            session['offer_id'] = offer_id
            
            return redirect(url_for('result'))
        else:
            flash('Tipo de archivo no permitido', 'danger')
            return redirect(request.url)
    
    # GET request
    offers = load_job_offers()
    return render_template('analyze.html', offers=offers)

@app.route('/result')
def result():
    if 'result' not in session:
        flash('No hay resultados disponibles', 'warning')
        return redirect(url_for('analyze'))
    
    result = session['result']
    offer_id = session['offer_id']
    
    # Obtener detalles de la oferta
    try:
        from src.data_loader import load_job_offers as load_offers
        offers_dict = load_offers(JOB_OFFERS_DIR)
        
        if offer_id in offers_dict:
            offer_text = offers_dict[offer_id]
            offer_title = extract_title(offer_text)
        else:
            offer_title = "Oferta no encontrada"
            offer_text = ""
    except Exception as e:
        logger.error(f"Error al cargar oferta para resultados: {str(e)}")
        offer_title = "Error al cargar oferta"
        offer_text = ""
    
    return render_template('result.html', 
                          result=result, 
                          offer_id=offer_id,
                          offer_title=offer_title,
                          offer_text=offer_text)

@app.route('/init_nltk')
def init_nltk():
    def run_init():
        try:
            from init_nltk import download_nltk_resources
            download_nltk_resources()
            logger.info("Recursos NLTK inicializados correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar NLTK: {str(e)}")
    
    # Ejecutar en un hilo separado
    threading.Thread(target=run_init).start()
    flash('Inicialización de recursos NLTK en progreso...', 'info')
    return redirect(url_for('index'))

# Crear directorios de plantillas y estáticos
def create_template_directories():
    # Obtener la ruta absoluta del directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Obtener el directorio raíz del proyecto (un nivel arriba)
    root_dir = os.path.dirname(current_dir)
    
    # Definir rutas para plantillas y archivos estáticos
    templates_dir = os.path.join(root_dir, 'templates')
    static_dir = os.path.join(root_dir, 'static')
    
    # Crear directorios si no existen
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(os.path.join(static_dir, 'css'), exist_ok=True)
    os.makedirs(os.path.join(static_dir, 'js'), exist_ok=True)
    
    logger.info(f"Directorios de plantillas creados en: {templates_dir}")
    logger.info(f"Directorios estáticos creados en: {static_dir}")
    
    return templates_dir, static_dir

def main():
    # Crear directorios necesarios
    templates_dir, static_dir = create_template_directories()
    
    # Verificar si las plantillas existen, si no, crearlas
    create_templates()
    
    # Imprimir información de depuración
    logger.info(f"Directorio de plantillas: {app.template_folder}")
    logger.info(f"Directorio estático: {app.static_folder}")
    
    try:
        logger.info(f"Plantillas existentes: {os.listdir(app.template_folder)}")
    except Exception as e:
        logger.error(f"Error al listar plantillas: {str(e)}")
    
    # Iniciar la aplicación
    logger.info("Iniciando servidor Flask en http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

def create_templates():
    # Obtener la ruta absoluta del directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Obtener el directorio raíz del proyecto (un nivel arriba)
    root_dir = os.path.dirname(current_dir)
    # Definir ruta para plantillas
    templates_dir = os.path.join(root_dir, 'templates')
    
    # Plantilla base
    base_template = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}CV Matcher{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary-color: #4a6fdc;
            --secondary-color: #6c757d;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --info-color: #17a2b8;
            --light-color: #f8f9fa;
            --dark-color: #343a40;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .navbar {
            background-color: var(--primary-color);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .navbar-brand {
            font-weight: 700;
            color: white !important;
        }
        
        .nav-link {
            color: rgba(255,255,255,0.85) !important;
            font-weight: 500;
        }
        
        .nav-link:hover {
            color: white !important;
        }
        
        .main-container {
            flex: 1;
            padding: 2rem 0;
        }
        
        .card {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 1.5rem;
            border: none;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        
        .card-header {
            background-color: rgba(74, 111, 220, 0.1);
            border-bottom: none;
            font-weight: 600;
            border-radius: 10px 10px 0 0 !important;
        }
        
        .btn-primary {
            background-color: var(--primary-color);
            border-color: var(--primary-color);
        }
        
        .btn-primary:hover {
            background-color: #3a5fc8;
            border-color: #3a5fc8;
        }
        
        .footer {
            background-color: var(--dark-color);
            color: white;
            padding: 1.5rem 0;
            margin-top: auto;
        }
        
        .progress {
            height: 25px;
            border-radius: 15px;
        }
        
        .result-card {
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .result-match {
            background-color: rgba(40, 167, 69, 0.1);
            border: 2px solid var(--success-color);
        }
        
        .result-no-match {
            background-color: rgba(220, 53, 69, 0.1);
            border: 2px solid var(--danger-color);
        }
        
        .percentage-display {
            font-size: 3rem;
            font-weight: 700;
        }
        
        .result-label {
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 1rem;
        }
        
        .match-label {
            color: var(--success-color);
        }
        
        .no-match-label {
            color: var(--danger-color);
        }
        
        .offer-list {
            max-height: 500px;
            overflow-y: auto;
        }
        
        .offer-item {
            cursor: pointer;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 5px;
            transition: background-color 0.2s ease;
        }
        
        .offer-item:hover {
            background-color: rgba(74, 111, 220, 0.1);
        }
        
        .offer-item.active {
            background-color: rgba(74, 111, 220, 0.2);
            border-left: 4px solid var(--primary-color);
        }
        
        .file-upload-container {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 1.5rem;
            transition: border-color 0.3s ease;
        }
        
        .file-upload-container:hover {
            border-color: var(--primary-color);
        }
        
        .upload-icon {
            font-size: 3rem;
            color: var(--secondary-color);
            margin-bottom: 1rem;
        }
        
        @media (max-width: 768px) {
            .card {
                margin-bottom: 1rem;
            }
            
            .percentage-display {
                font-size: 2.5rem;
            }
            
            .result-label {
                font-size: 1.2rem;
            }
        }
    </style>
    {% block extra_css %}{% endblock %}
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">
                <i class="fas fa-file-contract me-2"></i>CV Matcher
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('index') }}">
                            <i class="fas fa-home me-1"></i>Inicio
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('analyze') }}">
                            <i class="fas fa-search me-1"></i>Analizar CV
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('init_nltk') }}">
                            <i class="fas fa-sync me-1"></i>Inicializar NLTK
                        </a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container main-container">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        {% block content %}{% endblock %}
    </div>

    <footer class="footer">
        <div class="container text-center">
            <p class="mb-0">&copy; 2025 CV Matcher - Análisis inteligente de currículums</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>
    """
    
    # Plantilla de inicio
    index_template = """
{% extends "base.html" %}

{% block title %}CVQ - CV's Qualification - Inicio{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-12 text-center mb-4">
        <h1 class="display-4">Bienvenido a CVQ - CV's Qualification</h1>
        <p class="lead">Sistema inteligente para analizar la compatibilidad entre currículums y ofertas de trabajo</p>
    </div>
</div>

<div class="row">
    <div class="col-md-3">
        <div class="card h-100">
            <div class="card-header">
                <i class="fas fa-search me-2"></i>Analizar CV
            </div>
            <div class="card-body">
                <p>Sube tu currículum y compáralo con nuestras ofertas de trabajo para encontrar la mejor coincidencia.</p>
                <a href="{{ url_for('analyze') }}" class="btn btn-primary">Comenzar análisis</a>
            </div>
        </div>
    </div>
    
    <div class="col-md-3">
        <div class="card h-100">
            <div class="card-header">
                <i class="fas fa-upload me-2"></i>Subir Oferta
            </div>
            <div class="card-body">
                <p>Añade nuevas ofertas de trabajo al sistema para ampliar la base de datos de análisis.</p>
                <a href="{{ url_for('upload_offer') }}" class="btn btn-primary">Subir oferta</a>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card h-100">
            <div class="card-header">
                <i class="fas fa-briefcase me-2"></i>Ofertas disponibles
            </div>
            <div class="card-body">
                <p>Explora nuestra base de datos de ofertas de trabajo disponibles para análisis.</p>
                <p><strong>{{ offers|length }}</strong> ofertas disponibles</p>
                <button type="button" class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#offersModal">
                    Ver ofertas
                </button>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card h-100">
            <div class="card-header">
                <i class="fas fa-cogs me-2"></i>Recursos
            </div>
            <div class="card-body">
                <p>Inicializa los recursos necesarios para el análisis de texto y procesamiento de lenguaje natural.</p>
                <a href="{{ url_for('init_nltk') }}" class="btn btn-primary">Inicializar NLTK</a>
            </div>
        </div>
    </div>
</div>

<!-- Modal de ofertas -->
<div class="modal fade" id="offersModal" tabindex="-1" aria-labelledby="offersModalLabel" aria-hidden="true">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="offersModalLabel">Ofertas de trabajo disponibles</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
                <div class="input-group mb-3">
                    <span class="input-group-text"><i class="fas fa-search"></i></span>
                    <input type="text" class="form-control" id="offerSearch" placeholder="Buscar ofertas...">
                </div>
                
                <div class="list-group offer-list">
                    {% if offers %}
                        {% for offer in offers %}
                            <a href="{{ url_for('offer_detail', offer_id=offer.id) }}" class="list-group-item list-group-item-action">
                                <div class="d-flex w-100 justify-content-between">
                                    <h5 class="mb-1">{{ offer.title }}</h5>
                                    <small>ID: {{ offer.id }}</small>
                                </div>
                            </a>
                        {% endfor %}
                    {% else %}
                        <div class="text-center py-3">
                            <p>No hay ofertas disponibles</p>
                        </div>
                    {% endif %}
                </div>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cerrar</button>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script>
    $(document).ready(function() {
        // Filtro de búsqueda para ofertas
        $("#offerSearch").on("keyup", function() {
            var value = $(this).val().toLowerCase();
            $(".offer-list .list-group-item").filter(function() {
                $(this).toggle($(this).text().toLowerCase().indexOf(value) > -1)
            });
        });
    });
</script>
{% endblock %}
"""
    
    # Plantilla de análisis
    analyze_template = """
{% extends "base.html" %}

{% block title %}CVQ - CV's Qualification - Analizar CV{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-12 mb-4">
        <h1>Analizar Currículum</h1>
        <p class="lead">Sube tu CV y selecciona una oferta de trabajo para analizar la compatibilidad</p>
    </div>
</div>

<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <i class="fas fa-upload me-2"></i>Subir CV
            </div>
            <div class="card-body">
                <form method="POST" enctype="multipart/form-data" id="analyzeForm">
                    <div class="file-upload-container" id="dropZone">
                        <i class="fas fa-file-upload upload-icon"></i>
                        <h4>Arrastra y suelta tu CV aquí</h4>
                        <p>o</p>
                        <input type="file" name="cv_file" id="cv_file" class="d-none" accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png">
                        <button type="button" class="btn btn-primary" id="browseButton">
                            <i class="fas fa-folder-open me-2"></i>Examinar archivos
                        </button>
                        <p class="mt-2 text-muted">Formatos aceptados: HTML</p>
                        <div id="fileInfo" class="mt-3 d-none">
                            <div class="alert alert-info">
                                <i class="fas fa-file-alt me-2"></i><span id="fileName"></span>
                                <button type="button" class="btn-close float-end" id="removeFile"></button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <label for="offer_id" class="form-label">Selecciona una oferta de trabajo</label>
                        <select class="form-select" name="offer_id" id="offer_id" required>
                            <option value="">-- Selecciona una oferta --</option>
                            {% for offer in offers %}
                                <option value="{{ offer.id }}">{{ offer.title }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="d-grid gap-2">
                        <button type="submit" class="btn btn-primary" id="analyzeButton" disabled>
                            <i class="fas fa-search me-2"></i>Analizar compatibilidad
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <i class="fas fa-info-circle me-2"></i>Información
            </div>
            <div class="card-body">
                <h5>¿Cómo funciona?</h5>
                <ol>
                    <li>Sube tu currículum en formato PDF, DOC, DOCX o TXT</li>
                    <li>Selecciona una oferta de trabajo de la lista</li>
                    <li>Haz clic en "Analizar compatibilidad"</li>
                    <li>Nuestro algoritmo analizará la compatibilidad entre tu CV y la oferta seleccionada</li>
                    <li>Recibirás un resultado con el porcentaje de afinidad</li>
                </ol>
                
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <strong>Importante:</strong> Asegúrate de que tu CV esté actualizado y contenga información relevante para obtener mejores resultados.
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script>
    $(document).ready(function() {
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('cv_file');
        const browseButton = document.getElementById('browseButton');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const removeFile = document.getElementById('removeFile');
        const offerSelect = document.getElementById('offer_id');
        const analyzeButton = document.getElementById('analyzeButton');
        
        // Evento para el botón de examinar
        browseButton.addEventListener('click', function() {
            fileInput.click();
        });
        
        // Evento para cuando se selecciona un archivo
        fileInput.addEventListener('change', function() {
            updateFileInfo();
        });
        
        // Eventos para arrastrar y soltar
        dropZone.addEventListener('dragover', function(e) {
            e.preventDefault();
            dropZone.classList.add('border-primary');
        });
        
        dropZone.addEventListener('dragleave', function() {
            dropZone.classList.remove('border-primary');
        });
        
        dropZone.addEventListener('drop', function(e) {
            e.preventDefault();
            dropZone.classList.remove('border-primary');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                updateFileInfo();
            }
        });
        
        // Evento para remover archivo
        removeFile.addEventListener('click', function() {
            fileInput.value = '';
            fileInfo.classList.add('d-none');
            updateAnalyzeButtonState();
        });
        
        // Evento para cambio en select de ofertas
        offerSelect.addEventListener('change', function() {
            updateAnalyzeButtonState();
        });
        
        // Función para actualizar información del archivo
        function updateFileInfo() {
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                fileName.textContent = file.name;
                fileInfo.classList.remove('d-none');
            } else {
                fileInfo.classList.add('d-none');
            }
            updateAnalyzeButtonState();
        }
        
        // Función para actualizar estado del botón de análisis
        function updateAnalyzeButtonState() {
            if (fileInput.files.length > 0 && offerSelect.value) {
                analyzeButton.disabled = false;
            } else {
                analyzeButton.disabled = true;
            }
        }
    });
</script>
{% endblock %}
"""
    
    # Plantilla de detalle de oferta
    offer_detail_template = """
{% extends "base.html" %}

{% block title %}CVQ - CV's Qualification - Detalle de Oferta{% endblock %}

{% block content %}
<div class="row mb-4">
    <div class="col-md-8">
        <h1>{{ title }}</h1>
        <p class="text-muted">ID: {{ offer_id }}</p>
    </div>
    <div class="col-md-4 text-end">
        <a href="{{ url_for('analyze') }}?offer_id={{ offer_id }}" class="btn btn-primary">
            <i class="fas fa-search me-2"></i>Analizar CV con esta oferta
        </a>
    </div>
</div>

<div class="card">
    <div class="card-header">
        <i class="fas fa-file-alt me-2"></i>Contenido de la oferta
    </div>
    <div class="card-body">
        <div class="offer-content">
            {{ text|replace('\\n', '<br>')|safe }}
        </div>
    </div>
</div>

<div class="mt-4">
    <a href="{{ url_for('index') }}" class="btn btn-secondary">
        <i class="fas fa-arrow-left me-2"></i>Volver
    </a>
</div>
{% endblock %}
"""
    
    # Plantilla de resultados
    result_template = """
{% extends "base.html" %}

{% block title %}CVQ - CV's Qualification - Resultados{% endblock %}

{% block content %}
<div class="row mb-4">
    <div class="col-md-12">
        <h1>Resultados del Análisis</h1>
        <p class="lead">Compatibilidad entre tu CV y la oferta seleccionada</p>
    </div>
</div>

<div class="row">
    <div class="col-md-8">
        <div class="result-card {% if result.is_match %}result-match{% else %}result-no-match{% endif %}">
            <h2>Resultado:</h2>
            <div class="percentage-display">{{ result.percentage }}</div>
            <div class="progress mt-3 mb-3">
                <div class="progress-bar {% if result.is_match %}bg-success{% else %}bg-danger{% endif %}" 
                     role="progressbar" 
                     style="width: {{ result.probability * 100 }}%"
                     aria-valuenow="{{ result.probability * 100 }}" 
                     aria-valuemin="0" 
                     aria-valuemax="100">
                </div>
            </div>
            <div class="result-label {% if result.is_match %}match-label{% else %}no-match-label{% endif %}">
                {% if result.is_match %}
                    <i class="fas fa-check-circle me-2"></i>COINCIDE
                {% else %}
                    <i class="fas fa-times-circle me-2"></i>NO COINCIDE
                {% endif %}
            </div>
        </div>
        
        <div class="card mb-4">
            <div class="card-header">
                <i class="fas fa-briefcase me-2"></i>Oferta analizada
            </div>
            <div class="card-body">
                <h5>{{ offer_title }}</h5>
                <p class="text-muted">ID: {{ offer_id }}</p>
                <div class="offer-preview">
                    {{ offer_text[:300] + '...' if offer_text|length > 300 else offer_text }}
                </div>
                <a href="{{ url_for('offer_detail', offer_id=offer_id) }}" class="btn btn-outline-primary mt-3">
                    <i class="fas fa-eye me-2"></i>Ver oferta completa
                </a>
            </div>
        </div>
        
        <div class="card mb-4">
            <div class="card-header">
                <i class="fas fa-lightbulb me-2"></i>Recomendaciones
            </div>
            <div class="card-body">
                {% if result.is_match %}
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle me-2"></i>
                        <strong>¡Excelente coincidencia!</strong> Tu perfil es compatible con esta oferta de trabajo.
                    </div>
                    <p>Recomendaciones para mejorar aún más tu compatibilidad:</p>
                    <ul>
                        <li>Destaca tus logros relacionados con las habilidades requeridas</li>
                        <li>Personaliza tu carta de presentación para esta oferta específica</li>
                        <li>Prepárate para demostrar tus habilidades durante la entrevista</li>
                    </ul>
                {% else %}
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        <strong>Compatibilidad baja.</strong> Tu perfil no coincide completamente con esta oferta.
                    </div>
                    <p>Recomendaciones para mejorar tu compatibilidad:</p>
                    <ul>
                        <li>Actualiza tu CV para destacar habilidades relevantes para esta posición</li>
                        <li>Considera adquirir formación adicional en las áreas requeridas</li>
                        <li>Busca ofertas más alineadas con tu perfil actual</li>
                    </ul>
                {% endif %}
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <i class="fas fa-chart-pie me-2"></i>Estadísticas
            </div>
            <div class="card-body">
                <h5>Desglose de compatibilidad</h5>
                <div class="mb-3">
                    <label class="form-label">Compatibilidad general</label>
                    <div class="progress">
                        <div class="progress-bar bg-primary" role="progressbar" style="width: {{ result.probability * 100 }}%"></div>
                    </div>
                    <small class="text-muted">{{ result.percentage }}</small>
                </div>
                
                <div class="mb-3">
                    <label class="form-label">Umbral de coincidencia</label>
                    <div class="progress">
                        <div class="progress-bar bg-info" role="progressbar" style="width: 50%"></div>
                    </div>
                    <small class="text-muted">50%</small>
                </div>
                
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>
                    <small>El umbral de coincidencia es el valor mínimo de compatibilidad para considerar que un CV coincide con una oferta.</small>
                </div>
            </div>
        </div>
        
        <div class="card mt-4">
            <div class="card-header">
                <i class="fas fa-cogs me-2"></i>Acciones
            </div>
            <div class="card-body">
                <div class="d-grid gap-2">
                    <a href="{{ url_for('analyze') }}" class="btn btn-primary">
                        <i class="fas fa-search me-2"></i>Nuevo análisis
                    </a>
                    <a href="{{ url_for('index') }}" class="btn btn-outline-secondary">
                        <i class="fas fa-home me-2"></i>Volver al inicio
                    </a>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
"""
    
    # Escribir plantillas en archivos
    with open(os.path.join(templates_dir, 'base.html'), 'w', encoding='utf-8') as f:
        f.write(base_template)
    
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(index_template)
    
    with open(os.path.join(templates_dir, 'analyze.html'), 'w', encoding='utf-8') as f:
        f.write(analyze_template)
    
    with open(os.path.join(templates_dir, 'offer_detail.html'), 'w', encoding='utf-8') as f:
        f.write(offer_detail_template)
    
    with open(os.path.join(templates_dir, 'result.html'), 'w', encoding='utf-8') as f:
        f.write(result_template)

if __name__ == "__main__":
    main()
