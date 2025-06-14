# Instrucciones para ejecutar CVQ - CV's Qualification

Este proyecto ofrece dos interfaces de usuario diferentes:

## 1. Interfaz Gráfica (GUI)

Para ejecutar la interfaz gráfica basada en Tkinter:

```
python run_gui.py
```

Esta interfaz abrirá una ventana de aplicación donde podrás:
- Seleccionar ofertas de trabajo
- Cargar tu CV
- Analizar la compatibilidad entre ambos

## 2. Interfaz Web

Para ejecutar la interfaz web basada en Flask:

```
python run_web_ui.py
```

Una vez iniciado el servidor, abre tu navegador y visita:
- http://127.0.0.1:5000

La interfaz web te permitirá:
- Ver todas las ofertas disponibles
- Subir tu CV
- Analizar la compatibilidad
- Ver resultados detallados

## Ejecutar con selector de interfaz

También puedes usar el script principal que te permite elegir qué interfaz usar:

```
python run.py
```

## Requisitos

Asegúrate de tener instaladas todas las dependencias:

```
pip install -r requirements.txt
```

Si encuentras problemas con la interfaz web, puedes instalar Flask-Bootstrap:

```
pip install flask-bootstrap
```

## Estructura de directorios

El sistema creará automáticamente los siguientes directorios:
- `data/jobs`: Para almacenar ofertas de trabajo
- `data/cvs`: Para almacenar CVs
- `data/results`: Para almacenar resultados
- `models`: Para almacenar modelos entrenados
- `templates`: Para las plantillas web
- `static`: Para archivos estáticos web
