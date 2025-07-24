from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Dict, List, Optional
import tensorflow as tf # Se mantiene por si el modelo de joblib contiene objetos de TF
from tensorflow.keras.preprocessing import image # Se mantiene para el preprocesamiento de imagen
import numpy as np
import base64
import io
import joblib

# Importa tus funciones de preprocesamiento e interpretación
# Asegúrate de que estas rutas sean correctas para tu estructura de proyecto
from src.nombre_paquete.preprocessing.preprocessing_prediction import preprocess_image_for_prediction
from src.nombre_paquete.evaluation.interpret_prediction import interpret_prediction # Asume que esta función toma la salida del modelo y la convierte en algo interpretable

# --- Configuración ---
# **Importante: Revisa esta ruta y el tipo de modelo que realmente tienes.**
# Si es un modelo de TensorFlow/Keras guardado con .save(), usa tf.keras.models.load_model()
# Si es un modelo de sklearn, lightgbm, etc., usa joblib.load()
MODEL_PATH = "./src/nombre_paquete/models/best_models/model.joblib"

app = FastAPI(
    title="API DE CLASIFICACIÓN DE ENFERMEDADES DE PLANTAS",
    description="API para clasificar imágenes saludables o afectadas por 2 enfermedades.",
    version="1.0.0"
)
# Variable global para almacenar el modelo
model = None
model = joblib.load(MODEL_PATH)

# --- Clases de Pydantic para el Input/Output (AJUSTADAS PARA IMÁGENES BASE64) ---
class ImageInput(BaseModel):
    # 'image_base64' contendrá la imagen codificada en Base64
    # Esto es común cuando se envían imágenes en JSON a una API.
    image_base64: str

class PredictionOutput(BaseModel):
    predicted_class: str
    confidence: float
    #all_probabilities: Dict[str, float] # Probabilidades para todas las clases

# --- Endpoint Raíz ---
@app.get("/")
async def read_root():
    return {"message": "¡API funcionando! Visita /docs para ver la documentación."} 

@app.get("/predict")
async def read_root(data: ImageInput) -> PredictionOutput:
  img_array, __ = preprocess_image_for_prediction(img_path = data.image_base64)
  prediction = model.predict(img_array).flatten()
  results = interpret_prediction(prediccion)
  preds = PredictionOutput('predicted_class': results['predicted_class'],
                           'confidence': results['confidence'])
  return preds

