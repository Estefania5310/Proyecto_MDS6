import tensorflow as tf
from tensorflow import keras

def get_callbacks(model_save_path, patience=10, monitor='val_loss'):
    """
    Genera una lista de callbacks para el entrenamiento del modelo.

    Args:
        model_save_path (str): Ruta donde se guardará el mejor modelo.
        patience (int): Número de épocas sin mejora para EarlyStopping.
        monitor (str): Métrica a monitorear para EarlyStopping y ModelCheckpoint.

    Returns:
        list: Una lista de objetos tf.keras.callbacks.
    """
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True
    )

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        model_save_path,
        save_best_only=True,
        monitor=monitor # Aseguramos que el checkpoint también use el mismo monitor
    )
    return [checkpoint_cb, early_stopping_cb]
