import tensorflow as tf
from tensorflow import keras

def train_model(model, train_generator, val_generator, epochs, callbacks_list):
    """
    Entrena un modelo de Keras utilizando generadores de datos.

    Args:
        model (keras.Model): El modelo compilado a entrenar.
        train_generator (tf.keras.utils.Sequence): Generador de datos para entrenamiento.
        val_generator (tf.keras.utils.Sequence): Generador de datos para validación.
        epochs (int): Número de épocas para entrenar.
        callbacks_list (list): Lista de callbacks a usar durante el entrenamiento.

    Returns:
        tf.keras.callbacks.History: Objeto History que contiene los registros de entrenamiento.
    """
    print("\n--- Iniciando el entrenamiento del modelo ---")
    history = model.fit(
        train_generator,
        steps_per_epoch= train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data= val_generator,
        validation_steps= val_generator.samples // val_generator.batch_size,
        callbacks=callbacks_list
    )
    print("--- Entrenamiento finalizado ---")
    return history
