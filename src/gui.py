import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
import pickle
import threading
import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CVMatcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CV Matcher")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # Definir rutas
        self.data_dir = "data"
        self.job_offers_dir = os.path.join(self.data_dir, "jobs")
        self.models_dir = "models"
        self.cv_path = None
        self.selected_offer_id = None
        
        # Crear frame principal
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Crear pestañas
        self.tab_control = ttk.Notebook(self.main_frame)
        
        # Pestaña de inferencia
        self.inference_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.inference_tab, text="Inferencia")
        
        # Pestaña de configuración
        self.settings_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.settings_tab, text="Configuración")
        
        self.tab_control.pack(fill=tk.BOTH, expand=True)
        
        # Configurar pestaña de inferencia
        self.setup_inference_tab()
        
        # Configurar pestaña de configuración
        self.setup_settings_tab()
        
        # Cargar ofertas de trabajo
        self.load_job_offers()
    
    def setup_inference_tab(self):
        # Frame para selección de oferta
        offer_frame = ttk.LabelFrame(self.inference_tab, text="Seleccionar Oferta de Trabajo", padding="10")
        offer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Crear Treeview para mostrar ofertas (sin columna Empresa)
        self.offer_tree = ttk.Treeview(offer_frame, columns=("ID", "Título"), show="headings")
        self.offer_tree.heading("ID", text="ID")
        self.offer_tree.heading("Título", text="Título")
        self.offer_tree.column("ID", width=100)
        self.offer_tree.column("Título", width=500)  # Más ancho para el título
        self.offer_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Añadir scrollbar
        scrollbar = ttk.Scrollbar(offer_frame, orient=tk.VERTICAL, command=self.offer_tree.yview)
        self.offer_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        
        # Vincular evento de selección
        self.offer_tree.bind("<<TreeviewSelect>>", self.on_offer_selected)
        
        # Frame para selección de CV
        cv_frame = ttk.LabelFrame(self.inference_tab, text="Seleccionar CV", padding="10")
        cv_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Campo para mostrar la ruta del CV
        self.cv_path_var = tk.StringVar()
        cv_path_entry = ttk.Entry(cv_frame, textvariable=self.cv_path_var, width=50, state="readonly")
        cv_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Botón para seleccionar CV
        select_cv_button = ttk.Button(cv_frame, text="Examinar...", command=self.select_cv)
        select_cv_button.pack(side=tk.RIGHT)
        
        # Frame para mostrar detalles y resultados
        details_frame = ttk.LabelFrame(self.inference_tab, text="Detalles y Resultados", padding="10")
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Área de texto para mostrar detalles de la oferta
        self.offer_details_text = tk.Text(details_frame, height=5, width=50, wrap=tk.WORD)
        self.offer_details_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.offer_details_text.config(state=tk.DISABLED)
        
        # Frame para resultados
        results_frame = ttk.Frame(details_frame)
        results_frame.pack(fill=tk.X, pady=5)
        
        # Etiqueta para mostrar resultado
        ttk.Label(results_frame, text="Resultado:").pack(side=tk.LEFT, padx=(0, 5))
        self.result_var = tk.StringVar(value="No hay resultados")
        result_label = ttk.Label(results_frame, textvariable=self.result_var, font=("", 10, "bold"))
        result_label.pack(side=tk.LEFT)
        
        # Barra de progreso para mostrar porcentaje de afinidad
        ttk.Label(details_frame, text="Porcentaje de Afinidad:").pack(anchor=tk.W, pady=(5, 0))
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(details_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Etiqueta para mostrar porcentaje
        self.percentage_var = tk.StringVar(value="0%")
        percentage_label = ttk.Label(details_frame, textvariable=self.percentage_var, font=("", 12, "bold"))
        percentage_label.pack(anchor=tk.CENTER, pady=5)
        
        # Botón para realizar inferencia
        self.run_button = ttk.Button(self.inference_tab, text="Ejecutar Análisis", command=self.run_inference)
        self.run_button.pack(pady=10)
        self.run_button.config(state=tk.DISABLED)
    
    def setup_settings_tab(self):
        # Frame para configuración de rutas
        paths_frame = ttk.LabelFrame(self.settings_tab, text="Rutas", padding="10")
        paths_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Ruta de datos
        ttk.Label(paths_frame, text="Directorio de datos:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.data_dir_var = tk.StringVar(value=self.data_dir)
        data_dir_entry = ttk.Entry(paths_frame, textvariable=self.data_dir_var, width=50)
        data_dir_entry.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        ttk.Button(paths_frame, text="Examinar...", command=lambda: self.select_directory(self.data_dir_var)).grid(column=2, row=0, padx=5, pady=5)
        
        # Ruta de ofertas
        ttk.Label(paths_frame, text="Directorio de ofertas:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        self.job_offers_dir_var = tk.StringVar(value=self.job_offers_dir)
        job_offers_dir_entry = ttk.Entry(paths_frame, textvariable=self.job_offers_dir_var, width=50)
        job_offers_dir_entry.grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(paths_frame, text="Examinar...", command=lambda: self.select_directory(self.job_offers_dir_var)).grid(column=2, row=1, padx=5, pady=5)
        
        # Ruta de modelos
        ttk.Label(paths_frame, text="Directorio de modelos:").grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
        self.models_dir_var = tk.StringVar(value=self.models_dir)
        models_dir_entry = ttk.Entry(paths_frame, textvariable=self.models_dir_var, width=50)
        models_dir_entry.grid(column=1, row=2, sticky=tk.W, padx=5, pady=5)
        ttk.Button(paths_frame, text="Examinar...", command=lambda: self.select_directory(self.models_dir_var)).grid(column=2, row=2, padx=5, pady=5)
        
        # Botón para guardar configuración
        ttk.Button(self.settings_tab, text="Guardar Configuración", command=self.save_settings).pack(pady=10)
        
        # Botón para inicializar NLTK
        ttk.Button(self.settings_tab, text="Inicializar Recursos NLTK", command=self.init_nltk).pack(pady=10)
    
    def select_directory(self, var):
        directory = filedialog.askdirectory()
        if directory:
            var.set(directory)
    
    def save_settings(self):
        self.data_dir = self.data_dir_var.get()
        self.job_offers_dir = self.job_offers_dir_var.get()
        self.models_dir = self.models_dir_var.get()
        messagebox.showinfo("Configuración", "Configuración guardada correctamente")
        
        # Recargar ofertas de trabajo
        self.load_job_offers()
    
    def init_nltk(self):
        def run_init():
            try:
                from init_nltk import download_nltk_resources
                download_nltk_resources()
                messagebox.showinfo("NLTK", "Recursos de NLTK inicializados correctamente")
            except Exception as e:
                messagebox.showerror("Error", f"Error al inicializar NLTK: {str(e)}")
        
        # Ejecutar en un hilo separado para no bloquear la interfaz
        threading.Thread(target=run_init).start()
    
    def load_job_offers(self):
        """Carga las ofertas de trabajo en un hilo separado para evitar bloqueos"""
        # Mostrar mensaje de carga
        for item in self.offer_tree.get_children():
            self.offer_tree.delete(item)
        
        self.offer_tree.insert("", tk.END, values=("", "Cargando ofertas, por favor espere..."))
        
        # Iniciar carga en un hilo separado
        threading.Thread(target=self._load_job_offers_thread).start()
    
    def _load_job_offers_thread(self):
        """Proceso de carga de ofertas en un hilo separado"""
        # Verificar si el directorio existe
        if not os.path.exists(self.job_offers_dir):
            self.root.after(0, lambda: messagebox.showwarning("Advertencia", 
                                                             f"El directorio de ofertas {self.job_offers_dir} no existe"))
            return
        
        # Cargar ofertas
        try:
            from src.data_loader import load_job_offers
            offers_dict = load_job_offers(self.job_offers_dir)
            
            # Limpiar el treeview en el hilo principal
            self.root.after(0, lambda: self._clear_treeview())
            
            # Procesar ofertas en lotes para mejorar rendimiento
            batch_size = 100
            offer_items = list(offers_dict.items())
            
            for i in range(0, len(offer_items), batch_size):
                batch = offer_items[i:i+batch_size]
                processed_batch = [(offer_id, self.extract_title(offer_text)) 
                                  for offer_id, offer_text in batch]
                
                # Actualizar treeview en el hilo principal
                self.root.after(0, lambda b=processed_batch: self._add_offers_to_treeview(b))
            
            logger.info(f"Cargadas {len(offers_dict)} ofertas de trabajo")
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error al cargar ofertas: {str(e)}"))
            logger.error(f"Error al cargar ofertas: {str(e)}")
    
    def _clear_treeview(self):
        """Limpia el treeview en el hilo principal"""
        for item in self.offer_tree.get_children():
            self.offer_tree.delete(item)
    
    def _add_offers_to_treeview(self, offers):
        """Añade un lote de ofertas al treeview en el hilo principal"""
        for offer_id, title in offers:
            self.offer_tree.insert("", tk.END, values=(offer_id, title))
        self.root.update_idletasks()  # Actualizar la interfaz después de cada lote
    
    def extract_title(self, text):
        """Extrae un título más significativo de la oferta de trabajo"""
        if not text:
            return "Sin título"
        
        # Buscar patrones comunes de títulos de trabajo
        title_patterns = [
            r'(?i)puesto:?\s*([^\n\.]{5,50})',
            r'(?i)posición:?\s*([^\n\.]{5,50})',
            r'(?i)vacante:?\s*([^\n\.]{5,50})',
            r'(?i)título:?\s*([^\n\.]{5,50})',
            r'(?i)oferta:?\s*([^\n\.]{5,50})',
            r'(?i)empleo:?\s*([^\n\.]{5,50})',
            r'(?i)trabajo:?\s*([^\n\.]{5,50})'
        ]
        
        import re
        for pattern in title_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Si no se encuentra un patrón, usar las primeras líneas no vacías
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            # Usar la primera línea que tenga entre 5 y 50 caracteres
            for line in lines:
                if 5 <= len(line) <= 50:
                    return line
            
            # Si no hay líneas adecuadas, usar la primera línea
            title = lines[0]
            return title[:50] + "..." if len(title) > 50 else title
        
        # Si todo falla, usar las primeras palabras
        words = text.split()
        title = " ".join(words[:min(7, len(words))])
        return title[:50] + "..." if len(title) > 50 else title
    
    def on_offer_selected(self, event):
        selected_items = self.offer_tree.selection()
        if selected_items:
            item = selected_items[0]
            self.selected_offer_id = self.offer_tree.item(item, "values")[0]
            
            # Usar un hilo para cargar los detalles de la oferta
            threading.Thread(target=self._load_offer_details, args=(self.selected_offer_id,)).start()
            
            # Habilitar botón de inferencia si hay un CV seleccionado
            self.update_run_button_state()
    
    def _load_offer_details(self, offer_id):
        """Carga los detalles de la oferta en un hilo separado para evitar bloqueos"""
        try:
            from src.data_loader import load_job_offers
            offers_dict = load_job_offers(self.job_offers_dir)
            offer_text = offers_dict.get(offer_id, "No se encontró la oferta")
            
            # Actualizar la interfaz en el hilo principal
            self.root.after(0, lambda: self._update_offer_details(offer_text))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error al cargar detalles de la oferta: {str(e)}"))
    
    def _update_offer_details(self, offer_text):
        """Actualiza el texto de detalles de la oferta en el hilo principal"""
        self.offer_details_text.config(state=tk.NORMAL)
        self.offer_details_text.delete(1.0, tk.END)
        self.offer_details_text.insert(tk.END, offer_text[:500] + "..." if len(offer_text) > 500 else offer_text)
        self.offer_details_text.config(state=tk.DISABLED)
    
    def select_cv(self):
        filetypes = [
            ("Documentos", "*.pdf;*.docx;*.doc;*.txt"),
            ("PDF", "*.pdf"),
            ("Word", "*.docx;*.doc"),
            ("Texto", "*.txt"),
            ("Todos los archivos", "*.*")
        ]
        
        cv_path = filedialog.askopenfilename(
            title="Seleccionar CV",
            filetypes=filetypes
        )
        
        if cv_path:
            self.cv_path = cv_path
            self.cv_path_var.set(cv_path)
            
            # Habilitar botón de inferencia si hay una oferta seleccionada
            self.update_run_button_state()
    
    def update_run_button_state(self):
        if self.cv_path and self.selected_offer_id:
            self.run_button.config(state=tk.NORMAL)
        else:
            self.run_button.config(state=tk.DISABLED)
    
    def run_inference(self):
        if not self.cv_path or not self.selected_offer_id:
            messagebox.showwarning("Advertencia", "Debe seleccionar una oferta y un CV")
            return
        
        # Buscar el modelo más reciente
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        if not model_files:
            messagebox.showerror("Error", "No se encontró ningún modelo entrenado")
            return
        
        # Ordenar por fecha de modificación (más reciente primero)
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.models_dir, x)), reverse=True)
        model_path = os.path.join(self.models_dir, model_files[0])
        
        def run_inference_thread():
            try:
                # Importar funciones necesarias
                import sys
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from src.data_loader import extract_text_from_pdf, extract_text_from_docx, extract_text_from_html, extract_text_from_image
                from src.preprocessing import preprocess_documents
                from src.feature_engineering import extract_features
                from src.model import predict
                import pickle
                import pandas as pd
                
                # Determinar el tipo de archivo y extraer texto
                _, file_extension = os.path.splitext(self.cv_path)
                file_extension = file_extension.lower()
                
                cv_text = None
                if file_extension == '.pdf':
                    cv_text = extract_text_from_pdf(self.cv_path)
                elif file_extension in ['.doc', '.docx']:
                    cv_text = extract_text_from_docx(self.cv_path)
                elif file_extension in ['.html', '.htm']:
                    cv_text = extract_text_from_html(self.cv_path)
                elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                    cv_text = extract_text_from_image(self.cv_path)
                elif file_extension == '.txt':
                    with open(self.cv_path, 'r', encoding='utf-8', errors='ignore') as f:
                        cv_text = f.read()
                else:
                    raise ValueError(f"Formato de archivo no soportado: {file_extension}")
                
                if not cv_text:
                    raise ValueError("No se pudo extraer texto del archivo CV")
                
                # Cargar ofertas
                from src.data_loader import load_job_offers
                offers_dict = load_job_offers(self.job_offers_dir)
                
                if self.selected_offer_id not in offers_dict:
                    raise ValueError(f"Oferta de trabajo con ID {self.selected_offer_id} no encontrada")
                
                # Generar un ID temporal para el CV
                cv_id = f"temp_cv_{os.path.basename(self.cv_path)}"
                
                # Preprocesar documentos
                processed_offers = preprocess_documents({self.selected_offer_id: offers_dict[self.selected_offer_id]})
                processed_cvs = preprocess_documents({cv_id: cv_text})
                
                # Crear un DataFrame de aplicaciones simulado para la inferencia
                applications_df = pd.DataFrame({
                    'cv_id': [cv_id],
                    'offer_id': [self.selected_offer_id],
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
                probability = prediction['probability']
                is_match = prediction['predicted_label'] == 1
                
                # Actualizar interfaz
                self.root.after(0, lambda: self.update_results(probability, is_match))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error en la inferencia: {str(e)}"))
                logger.error(f"Error en la inferencia: {str(e)}")
        
        # Mostrar mensaje de procesamiento
        self.result_var.set("Procesando...")
        self.percentage_var.set("Calculando...")
        self.progress_var.set(0)
        
        # Ejecutar en un hilo separado para no bloquear la interfaz
        threading.Thread(target=run_inference_thread).start()
    
    def update_results(self, probability, is_match):
        # Actualizar barra de progreso
        self.progress_var.set(probability * 100)
        
        # Actualizar etiqueta de porcentaje
        self.percentage_var.set(f"{probability * 100:.2f}%")
        
        # Actualizar resultado
        if is_match:
            self.result_var.set("COINCIDE")
        else:
            self.result_var.set("NO COINCIDE")

def main():
    root = tk.Tk()
    app = CVMatcherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
