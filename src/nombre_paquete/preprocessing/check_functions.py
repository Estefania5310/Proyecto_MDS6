
import os
import warnings
from PIL import Image, UnidentifiedImageError
import numpy as np

def check_image_files(folder_path):
    errors = []
    dirs = list(os.walk(folder_path))

    for i, regsiter in enumerate(dirs):
        root, _, files = regsiter
        print(f'\r Analizados {i+1} de {len(dirs)} registros',end='')
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    img = Image.open(file_path)
                    img_array = np.array(img)

                if img_array.size == 0:
                  errors.append((file_path, "Imagen vacía"))

                elif np.all(img_array == img_array.flat[0]):
                    errors.append((file_path, "Imagen constante"))

            except UnidentifiedImageError:
                errors.append((file_path, "No se puede leer la imagen (formato no reconocido)"))

            except Exception as e:
                errors.append((file_path, f"Error: {e}"))
    return errors
