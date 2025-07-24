from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple, Dict, Any, Literal, Union

def create_augmented_train_generator(
    data_directory: str,
    image_height: int,
    image_width: int,
    batch_size: int,
    class_mode: Literal['categorical', 'binary', 'sparse', 'input', 'multi_output', None] = 'categorical',
    augmentation_params: Dict[str, Any] | None = None
) -> tf.keras.preprocessing.image.DirectoryIterator:
    """
    Crea un generador de datos de imágenes para el conjunto de entrenamiento con aumentación.

    Utiliza `tf.keras.preprocessing.image.ImageDataGenerator` para cargar imágenes
    desde un directorio y aplicar transformaciones de aumentación de datos.

    Args:
        data_directory (str): La ruta al directorio raíz que contiene las imágenes de entrenamiento,
                              organizadas en subdirectorios por clase.
        image_height (int): La altura deseada de las imágenes de salida en píxeles.
        image_width (int): El ancho deseado de las imágenes de salida en píxeles.
        batch_size (int): El número de imágenes por lote (batch) que generará el iterador.
        class_mode (Literal): El modo de clase para las etiquetas. Puede ser 'categorical',
                              'binary', 'sparse', 'input', 'multi_output', o None.
                              Por defecto es 'categorical'.
        augmentation_params (Dict[str, Any] | None, opcional): Un diccionario que contiene
                                    los parámetros para las transformaciones de aumentación de datos
                                    que se aplicarán (ej. 'rotation_range', 'zoom_range', etc.).
                                    Si es `None`, se utiliza un conjunto de parámetros predeterminado
                                    (similar al de tu ejemplo original). Por defecto es `None`.

    Returns:
        tf.keras.preprocessing.image.DirectoryIterator: Un objeto iterador de Keras
                                                        que genera lotes de imágenes
                                                        y etiquetas aumentadas.

    Raises:
        FileNotFoundError: Si el directorio de datos especificado no existe.
    """
    # Verifica si el directorio existe
    if not Path(data_directory).is_dir():
        raise FileNotFoundError(f"El directorio de datos de entrenamiento no existe: {data_directory}")

    # Parámetros de aumentación predeterminados si no se proporcionan
    if augmentation_params is None:
        augmentation_params = {
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.2,
            'zoom_range': 0.2,
            'horizontal_flip': True,
            'fill_mode': 'nearest'
        }

    # Crear el ImageDataGenerator con reescalado y parámetros de aumentación
    data_generator = ImageDataGenerator(rescale=1./255, **augmentation_params)

    print(f"Creando generador de entrenamiento desde: {data_directory}")
    print(f"Tamaño de imagen: ({image_height}, {image_width}), Tamaño de lote: {batch_size}")
    print(f"Modo de clase: '{class_mode}'")
    print("Parámetros de aumentación aplicados:", augmentation_params)

    # Crear el generador a partir del directorio
    generator = data_generator.flow_from_directory(
        directory=data_directory,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode=class_mode
    )

    print(f"Generador de entrenamiento creado. Encontradas {generator.samples} imágenes pertenecientes a {generator.num_classes} clases.")

    return generator



