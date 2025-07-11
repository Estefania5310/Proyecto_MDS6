
from tensorflow.keras.utils import load_img, img_to_array
from skimage import util, transform # Necesario para transform.rotate y transform.resize
import gc # Para gestión de memoria en Colab
import numpy as np


def augment_image_simple(image, shape=[224, 224], flip=True, rotate=[-180, 180]):
    current_image = image.copy() # Trabajar en una copia para no modificar el original

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
