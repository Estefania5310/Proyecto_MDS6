import tensorflow as tf

# Define la función de preprocesamiento
def preprocess_image(image, label):
    # Usa la ruta completa desde el módulo 'tf'
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label
