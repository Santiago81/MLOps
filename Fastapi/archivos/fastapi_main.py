from fastapi import FastAPI
from transformers import pipeline

# Inicializar la app
app = FastAPI()

# Inicializar los pipelines de Hugging Face
sentiment_pipeline = pipeline("sentiment-analysis")
translation_pipeline = pipeline("translation_en_to_fr")

# Endpoint raíz
@app.get("/")
def root():
    return {"message": "API de prueba con FastAPI y Hugging Face"}

# Endpoint 1: Saludo
@app.get("/saludo")
def saludo(nombre: str = "Santiago"):
    return {"mensaje": f"Hola {nombre}, bienvenido a FastAPI"}

# Endpoint 2: Análisis de sentimiento
@app.get("/sentimiento")
def sentimiento(texto: str):
    resultado = sentiment_pipeline(texto)
    return {"resultado": resultado}

# Endpoint 3: Traducción EN → FR
@app.get("/traduccion")
def traduccion(texto: str):
    resultado = translation_pipeline(texto)
    return {"traduccion": resultado[0]['translation_text']}

# Endpoint 4: Longitud del texto
@app.get("/longitud")
def longitud_texto(texto: str):
    return {"longitud": len(texto)}

# Endpoint 5: Contador de palabras
@app.get("/palabras")
def contar_palabras(texto: str):
    return {"palabras": len(texto.split())}