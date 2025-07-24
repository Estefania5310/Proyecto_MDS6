<<<<<<< HEAD
from fastapi import FastAPI, HTTPException, status, UploadFile, File # <-- ¡IMPORTANTE! Agrega UploadFile y File
from pydantic import BaseModel
from typing import Dict, List, Optional
import tensorflow as tf
import numpy as np
import joblib
import os # <-- ¡IMPORTANTE! Necesitas os para manejar archivos temporales
import shutil # <-- ¡OPCIONAL pero ÚTIL! Para guardar archivos de manera más robusta
=======
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Dict, List, Optional
import tensorflow as tf # Se mantiene por si el modelo de joblib contiene objetos de TF
from tensorflow.keras.preprocessing import image # Se mantiene para el preprocesamiento de imagen
import numpy as np
import base64
import io
import joblib
>>>>>>> origin/main

# Importa tus funciones de preprocesamiento e interpretación
# Asegúrate de que estas rutas sean correctas para tu estructura de proyecto
from src.nombre_paquete.preprocessing.preprocessing_prediction import preprocess_image_for_prediction
<<<<<<< HEAD
from src.nombre_paquete.evaluation.interpret_prediction import interpret_prediction

# --- Configuración ---
=======
from src.nombre_paquete.evaluation.interpret_prediction import interpret_prediction # Asume que esta función toma la salida del modelo y la convierte en algo interpretable

# --- Configuración ---
# **Importante: Revisa esta ruta y el tipo de modelo que realmente tienes.**
# Si es un modelo de TensorFlow/Keras guardado con .save(), usa tf.keras.models.load_model()
# Si es un modelo de sklearn, lightgbm, etc., usa joblib.load()
>>>>>>> origin/main
MODEL_PATH = "./src/nombre_paquete/models/best_models/model.joblib"

app = FastAPI(
    title="API DE CLASIFICACIÓN DE ENFERMEDADES DE PLANTAS",
    description="API para clasificar imágenes saludables o afectadas por 2 enfermedades.",
    version="1.0.0"
)
<<<<<<< HEAD

# ¡Carga del modelo al inicio de la aplicación!
# Mueve esta línea dentro de @app.on_event("startup")
model = None # Inicializa como None

class PredictionOutput(BaseModel):
    saludable: float
    powdery: float
    rush: float
    predicted_class: str
    confidence: float
=======
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
>>>>>>> origin/main

# --- Endpoint Raíz ---
@app.get("/")
async def read_root():
<<<<<<< HEAD
    return {"message": "¡API funcionando! Visita /docs para ver la documentación."}

# --- Health Check Endpoint (Recomendado) ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Verifica el estado de la API y si el modelo ha sido cargado exitosamente.
    """
    if model is not None:
        return {"status": "ok", "model_loaded": True, "message": "Modelo cargado y API lista."}
    else:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Modelo no cargado. El servicio no está disponible.")

# --- Carga del modelo al inicio de la aplicación ---
@app.on_event("startup")
async def load_model_on_startup():
    """
    Carga el modelo de Joblib al iniciar la aplicación FastAPI.
    """
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Modelo cargado exitosamente desde: {MODEL_PATH}")
    except Exception as e:
        print(f"Error al cargar el modelo desde '{MODEL_PATH}': {e}")
        # Considera no levantar una excepción fatal aquí si quieres que la API inicie
        # pero sin el modelo cargado, y manejar el caso en el endpoint /predict
        # raise RuntimeError(f"No se pudo cargar el modelo al iniciar la aplicación: {e}")
    print("Startup event finished.") # Debug print

# --- Endpoint de Predicción ---
@app.post("/predict", response_model=PredictionOutput)
# ¡IMPORTANTE! Cambia 'data: ImageInput' a 'file: UploadFile = File(...)'
# Esto le dice a FastAPI que esperas un archivo subido.
async def predict_plant_disease(file: UploadFile = File(...)):
    """
    Recibe un archivo de imagen, lo guarda temporalmente, lo preprocesa
    y devuelve la predicción de la enfermedad de la planta.
    """
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="El modelo no ha sido cargado todavía. Intente de nuevo en unos momentos.")

    # Definir una ruta temporal para guardar el archivo
    # Usamos os.path.join para construir rutas de forma segura en diferentes OS
    # Y un nombre único para evitar conflictos si muchas peticiones llegan a la vez
    temp_dir = "/tmp" # O "./temp_uploads" si quieres una carpeta local
    # Asegúrate de que el directorio exista
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{os.urandom(16).hex()}_{file.filename}")

    try:
        # 1. Guardar el archivo subido en la ruta temporal
        # shutil.copyfileobj es eficiente para archivos grandes
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Archivo temporal guardado en: {temp_file_path}")

        # 2. Preprocesar la imagen usando la ruta del archivo temporal
        # Tu función preprocess_image_for_prediction espera esta ruta
        img_array, _ = preprocess_image_for_prediction(img_path=temp_file_path)
        if img_array is None:
            # Si preprocess_image_for_prediction devuelve None, hubo un error (ej. archivo no es imagen)
            # La función ya imprime el error específico
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error al preprocesar la imagen. Asegúrate de que es una imagen válida.")

        # 3. Realizar la predicción
        # model.predict(img_array) devuelve un array de arrays (ej: [[prob_s, prob_p, prob_r]])
        # Usamos [0] para obtener el array plano de probabilidades para la única imagen procesada.
        probabilities = model.predict(img_array)[0]

        # 4. Interpretar la predicción
        # Tu función interpret_prediction debe tomar este array plano de probabilidades
        # y la lista de nombres de clases, y devolver el diccionario esperado.
        # Asegúrate de que CLASS_NAMES está definido y en el orden correcto.
        # Aquí asumo que interpret_prediction ya tiene los nombres de clase, si no, pásalos como argumento.
        # Define CLASS_NAMES somewhere if needed, or update interpret_prediction
        # Example: CLASS_NAMES = ['Healthy', 'Powdery', 'Rust']
        results = interpret_prediction(probabilities) # Assuming interpret_prediction doesn't need class_names here

        # 5. Retornar los resultados
        # FastAPI se encargará de mapear las claves del diccionario 'results'
        # a los campos de PredictionOutput y validar los tipos.
        return PredictionOutput(**results)

    except Exception as e:
        # Captura cualquier error durante el proceso y devuélvelo como un error HTTP
        print(f"Error en la predicción: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Ocurrió un error en el servidor durante la predicción: {e}")
    finally:
        # ¡IMPORTANTE! Asegúrate de borrar el archivo temporal, siempre.
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Archivo temporal eliminado: {temp_file_path}")
=======
    return {"message": "¡API funcionando! Visita /docs para ver la documentación."} 

@app.get("/predict")
async def read_root(data: ImageInput) -> PredictionOutput:
  img_array, __ = preprocess_image_for_prediction(img_path = data.image_base64)
  prediction = model.predict(img_array).flatten()
  results = interpret_prediction(prediccion)
  preds = PredictionOutput('predicted_class': results['predicted_class'],
                           'confidence': results['confidence'])
  return preds

>>>>>>> origin/main
