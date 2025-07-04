
## Origen de los datos

Los datos utilizados en este proyecto provienen del repositorio público [Plant Disease Recognition Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset), disponible en la plataforma Kaggle. La descarga se realizó a través de la API oficial de Kaggle, y los archivos fueron almacenados en Google Drive para facilitar su procesamiento.

---

## Especificación de los scripts para la carga de datos

La carga de los datos se realiza mediante el script ubicado en:  
**Proyecto_MDS6/src/data/load_data.py**

Este script define la función `cargar_datasets()`, que permite cargar imágenes organizadas en carpetas según su clase para construir los conjuntos de entrenamiento, validación y prueba. A partir de las rutas proporcionadas (train_dir, val_dir, test_dir), la función utiliza `tf.keras.utils.image_dataset_from_directory` para leer automáticamente las imágenes, redimensionarlas al tamaño especificado, agruparlas en lotes (batch_size) y asignar etiquetas basadas en el nombre de las subcarpetas. El resultado son tres objetos `tf.data.Dataset` que contienen tensores con las imágenes y sus respectivas etiquetas

---

## Referencias a rutas o bases de datos origen y destino

- **Ruta de origen externa (Kaggle)**:  
  https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset

- **Ruta local de almacenamiento (Google Drive)**:  
  https://drive.google.com/drive/folders/1v1ZclyXgblKxRnChCdwB_TC_xFC09ygq

En esta carpeta se almacenan las imágenes organizadas por clases y por conjunto (entrenamiento, validación, prueba).

---

### Estructura y Procedimientos de transformación

- **Formato**: Imágenes en `.jpg`  
- **Etiquetas**: Definidas por el nombre de la carpeta que contiene la imagen  
- **Transformaciones aplicadas**:
  - Redimensionamiento de imágenes a (180×180 px)
  - Aumentación de datos (rotaciones, brillo, zoom)
  - Normalización de píxeles

