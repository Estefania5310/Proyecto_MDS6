from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import tensorflow as tf # Se mantiene por si el modelo de joblib contiene objetos de TF
from tensorflow.keras.preprocessing import image # Se mantiene para el preprocesamiento de imagen
import numpy as np
import base64
import io
import joblib

from src.nombre_paquete.preprocessing.preprocessing_prediction import preprocess_image_for_prediction
from src.nombre_paquete.evaluation.interpret_prediction import interpret_prediction


MODEL_PATH = "./src/nombre_paquete/models/best_models/model.joblib"


class ApiInput(BaseModel):
    """
    Define la estructura de la entrada del API para una imagen.
    La imagen se espera como una cadena Base64.
    """
    image_base64: str

class ApiOutput(BaseModel):
    """
    Define la estructura de la salida del API para la predicción de imagen.
    Incluye la clase predicha, la confianza y las probabilidades por clase.
    """
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]

app = FastAPI()
model: Optional[object] = None


@app.on_event("startup")
async def load_model():
    """
    Carga el modelo guardado con joblib cuando la aplicación se inicia.
    """
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Modelo '{MODEL_PATH}' cargado exitosamente con joblib.")
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo: {e}. Asegúrate de que la ruta es correcta y el archivo existe.")

@app.get("/")
async def read_root():
    return {"message": "¡API de clasificación de imágenes funcionando!"}

@app.post("/predict_image")
async def predict_image_class(data: ApiInput) -> ApiOutput:
    """
    Endpoint para predecir la clase de una imagen enviada como Base64.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo aún no ha sido cargado.")

    try:
        # CORRECCIÓN: Pasamos la cadena Base64 y las dimensiones a la función de preprocesamiento.
        # IMPORTANTE: Tu función 'preprocess_image_for_prediction'
        # (en src/nombre_paquete/preprocessing/preprocessing_prediction.py)
        # DEBE aceptar la cadena Base64 como primer argumento y las dimensiones (altura, ancho)
        # como segundo y tercer argumento, y devolver el array NumPy preprocesado.
        # Ejemplo de firma esperada:
        # def preprocess_image_for_prediction(base64_string: str, target_height: int, target_width: int) -> np.ndarray:
        #   ...
        img_array, _ = preprocess_image_for_prediction(
            data.image_base64, # Se pasa la cadena Base64 de la entrada del API
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al preprocesar la imagen: {e}")

    # 2. Generar la predicción
    prediccion = model.predict(img_array) # Variable corregida a 'prediccion'

    # 3. Interpretar y estructurar la salida
    interpretation_results = interpret_prediction(prediccion)

    # Convertir el diccionario de resultados a la estructura ApiOutput
    output = ApiOutput(
        predicted_class=interpretation_results['predicted_class'],
        confidence=interpretation_results['confidence'],
        probabilities=interpretation_results['probabilities']
    )

    return output
