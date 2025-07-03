## Origen de los datos

Los datos utilizados en este proyecto provienen del repositorio público [Plant Disease Recognition Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset), disponible en la plataforma Kaggle. El conjunto incluye 1.530 imágenes clasificadas en tres categorías: Healthy, Powdery Mildew y Rust.  
La descarga se realizó a través de la API oficial de Kaggle, y los archivos fueron almacenados en Google Drive para facilitar su procesamiento en Google Colab.

---

## Especificación de los scripts para la carga de datos

La carga de los datos se realiza mediante el script ubicado en:  
**code/data_ingestion/load_data.py**

Este script define la función `cargar_datasets()` que usa `tf.keras.utils.image_dataset_from_directory()` para convertir carpetas de imágenes en objetos `tf.data.Dataset`, listos para ser usados en el entrenamiento del modelo.

---

## Referencias a rutas o bases de datos origen y destino

- **Ruta de origen externa (Kaggle)**:  
  https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset

- **Ruta local de almacenamiento (Google Drive)**:  
  `/content/drive/MyDrive/Proyecto_MDS6/data/raw/`

En esta carpeta se almacenan las imágenes organizadas por clases y por conjunto (entrenamiento, validación, prueba).

---

## Rutas de origen de datos

- **Formato**: Imágenes en `.jpg`  
- **Etiquetas**: Definidas por el nombre de la carpeta que contiene la imagen  
- **Transformaciones aplicadas**:
  - Redimensionamiento de imágenes a (180×180 px)
  - Aumentación de datos (rotaciones, brillo, zoom)
  - Normalización de píxeles

---

## Base de datos de destino

Este proyecto no utiliza una base de datos tradicional (SQL/NoSQL).  
Los datos se manipulan como tensores en memoria a través de objetos `tf.data.Dataset` dentro del entorno de Google Colab.

- **Destino en memoria**:  
  Tensores cargados en batches para entrenamiento y validación

- **Estructura destino**:
  - Tensores de imágenes: `(batch_size, height, width, channels)`
  - Tensores de etiquetas: `(batch_size, 1)`

- **Procesos aplicados**:
  - Conversión de imágenes en tensores
  - Batching y shuffling
  - Asignación automática de etiquetas según carpeta
