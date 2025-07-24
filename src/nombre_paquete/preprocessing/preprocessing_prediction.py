from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image_for_prediction(img_path, target_height=180, target_width=180):
    """
    Carga, redimensiona y preprocesa una imagen para ser usada en un modelo de Keras.

    Args:
        img_path (str): La ruta completa a la imagen.
        target_height (int): La altura a la que se debe redimensionar la imagen.
        target_width (int): El ancho a la que se debe redimensionar la imagen.

    Returns:
        tuple: Una tupla que contiene:
            - img_array (np.ndarray): La imagen preprocesada lista para el modelo.
                                      None si ocurre un error.
            - original_dimensions (tuple): Las dimensiones originales (ancho, alto) de la imagen.
                                           None si ocurre un error.
    """
    try:
        original_img = image.load_img(img_path)
        original_width, original_height = original_img.size
        original_dimensions = (original_width, original_height)
        img_for_prediction = image.load_img(img_path, target_size=(target_width, target_height))
        print("\n--- Preprocesamiento de imagen ---")
        print(f"Imagen cargada desde: '{img_path}'.")
        print(f"Dimensiones originales de la imagen: {original_dimensions}")
        print(f"Dimensiones después de escalar (para el modelo): {img_for_prediction.size}")


        img_array = image.img_to_array(img_for_prediction)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        print("Preprocesamiento: valores de píxeles normalizados a [0, 1].")
        print(f"Forma final del array de imagen: {img_array.shape}")

        return img_array, original_dimensions

    except FileNotFoundError:
        print(f"Error: La imagen no se encontró en la ruta '{img_path}'. Por favor, verifica la ruta.")
        return None, None
    except Exception as e:
        print(f"Ocurrió un error al procesar la imagen: {e}")
        return None, None
