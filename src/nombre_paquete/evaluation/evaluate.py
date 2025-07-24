import tensorflow as tf

def evaluate_model(model, test_generator):
    """
    Evalúa un modelo de Keras en un conjunto de prueba.

    Args:
        model (keras.Model): El modelo a evaluar.
        test_generator (tf.keras.utils.Sequence): Generador de datos para el conjunto de prueba.

    Returns:
        tuple: Una tupla que contiene la pérdida y la precisión del modelo en el conjunto de prueba.
    """
    print("\n--- Evaluando el modelo en el conjunto de prueba ---")
    loss, accuracy = model.evaluate(test_generator)
    print(f"Pérdida en el conjunto de prueba: {loss:.4f}")
    print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")
    return loss, accuracy
