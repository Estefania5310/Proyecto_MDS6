
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from skimage import util, transform
import gc # Para gestión de memoria

def augment_image_simple(image, shape=[224, 224], flip=True, rotate=[-180, 180]):
    """
    Realiza aumentación de imagen simple (volteo horizontal, rotación y redimensionamiento).
    Esta función opera sobre arrays de NumPy (normalizados a [0, 1]).
    """
    current_image = image.copy()

    # Volteo horizontal
    flip_flag = np.random.choice([True, False]) if flip else False
    if flip_flag:
        current_image = np.fliplr(current_image)

    # Rotación
    rotate_angle = np.random.randint(rotate[0], rotate[1] + 1)
    current_image = transform.rotate(current_image, angle=rotate_angle, resize=False, preserve_range=True)

    # Redimensionamiento final
    augmented_image = transform.resize(current_image, shape, anti_aliasing=True)

    return augmented_image


# --- Función de Envoltura para TensorFlow (tf.py_function) ---
def tf_augment_wrapper(image, label, target_shape=[224, 224]):
    """
    Función de envoltura para llamar a `augment_image_simple` (NumPy/Skimage)
    dentro de un pipeline de tf.data.Dataset.
    """
    processed_image = tf.py_function(
        func=lambda img: augment_image_simple(img / 255.0, shape=target_shape, flip=True, rotate=[-180, 180]),
        inp=[image],
        Tout=tf.float32
    )

    processed_image.set_shape(target_shape + [3])

    return processed_image, label


# --- Función Principal para Aplicar Preprocesamiento al Dataset ---
def apply_preprocessing_to_dataset(dataset: tf.data.Dataset, target_shape: list = [224, 224], augment: bool = False) -> tf.data.Dataset:
    """
    Aplica el preprocesamiento y/o la aumentación a un tf.data.Dataset.

    Args:
        dataset (tf.data.Dataset): El dataset de TensorFlow a procesar.
        target_shape (list): Dimensiones [alto, ancho] de las imágenes.
        augment (bool): Si se debe aplicar aumentación a las imágenes.

    Returns:
        tf.data.Dataset: El dataset con imágenes preprocesadas/aumentadas y etiquetas.
    """
    def normalize_img_only(image, label):
        return tf.cast(image, tf.float32) / 255.0, label # Normalizamos a [0, 1]

    # Aplica esta normalización a todo el dataset
    dataset = dataset.map(normalize_img_only, num_parallel_calls=tf.data.AUTOTUNE)


    # Aplicar Aumentación (si es necesario)
    if augment:
        # Aquí usamos la función de envoltura que llama a tu augment_image_simple
        dataset = dataset.map(lambda x, y: tf_augment_wrapper(x, y, target_shape),
                              num_parallel_calls=tf.data.AUTOTUNE)

    # 3. Optimización del dataset para rendimiento
    # .cache() guarda el dataset en memoria o disco después de la primera iteración,
    # útil si el dataset cabe en memoria o si las operaciones son costosas.
    dataset = dataset.cache()
    # .prefetch() superpone el preprocesamiento de datos y la ejecución del modelo,
    # lo que mejora el rendimiento de la pipeline.
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
