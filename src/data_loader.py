import os
import pandas as pd
import pypdf
from docx import Document
import html2text
from PIL import Image
import pytesseract
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_applications(file_path):
    """
    Carga el archivo parquet que contiene la relación entre ofertas y aplicantes.
    
    Args:
        file_path: Ruta al archivo parquet
        
    Returns:
        DataFrame con las relaciones entre ofertas y aplicantes
    """
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Error al cargar el archivo de aplicaciones: {e}")
        raise

def extract_text_from_pdf(file_path):
    """Extrae texto de archivos PDF."""
    try:
        text = ""
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.warning(f"Error al procesar PDF {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extrae texto de archivos DOCX."""
    try:
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        logger.warning(f"Error al procesar DOCX {file_path}: {e}")
        return ""

def extract_text_from_html(file_path):
    """Extrae texto de archivos HTML."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            h = html2text.HTML2Text()
            h.ignore_links = True
            return h.handle(file.read())
    except Exception as e:
        logger.warning(f"Error al procesar HTML {file_path}: {e}")
        return ""

def extract_text_from_image(file_path):
    """Extrae texto de imágenes usando OCR."""
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        logger.warning(f"Error al procesar imagen {file_path}: {e}")
        return ""

def load_job_offers(directory):
    """
    Carga todas las ofertas de trabajo desde archivos HTML.
    
    Args:
        directory: Directorio que contiene los HTMLs de ofertas
        
    Returns:
        Diccionario con ID de oferta como clave y texto como valor
    """
    offers = {}
    
    if not os.path.exists(directory):
        logger.error(f"El directorio {directory} no existe")
        return offers
    
    for filename in os.listdir(directory):
        if filename.endswith(('.html', '.htm')):
            file_path = os.path.join(directory, filename)
            offer_id = os.path.splitext(filename)[0]  # Usar nombre de archivo como ID
            text = extract_text_from_html(file_path)
            offers[offer_id] = text
            
    logger.info(f"Cargadas {len(offers)} ofertas de trabajo")
    return offers

def load_cvs(directory):
    """
    Carga todos los CVs desde varios formatos (PDF, DOCX, HTML, imágenes).
    
    Args:
        directory: Directorio que contiene los CVs
        
    Returns:
        Diccionario con ID de CV como clave y texto como valor
    """
    cvs = {}
    
    if not os.path.exists(directory):
        logger.error(f"El directorio {directory} no existe")
        return cvs
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        cv_id = os.path.splitext(filename)[0]  # Usar nombre de archivo como ID
        
        if filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif filename.lower().endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif filename.lower().endswith(('.html', '.htm')):
            text = extract_text_from_html(file_path)
        elif filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            text = extract_text_from_image(file_path)
        else:
            logger.warning(f"Formato no soportado: {filename}")
            continue
            
        cvs[cv_id] = text
    
    logger.info(f"Cargados {len(cvs)} CVs")
    return cvs
