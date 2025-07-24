import numpy as np

def interpret_prediction(predictions):
    """
    Interpreta el resultado de model.predict() y muestra las probabilidades por clase.

    Args:
        predictions (np.ndarray): El array de predicciones devuelto por model.predict().
                                  Debe ser un array 2D (ej: [[prob_clase1, prob_clase2, ...]])
        class_names (list): Una lista de strings con los nombres de las clases
                            en el mismo orden en que el modelo predice las probabilidades.

    Returns:
        dict: Un diccionario con las probabilidades de cada clase,
              la clase predicha y su confianza.
              Retorna None si las dimensiones no coinciden.
    """
    class_names = ['saludable', 'powdery', 'rush']
    if predictions.shape[0] == 0:
        print("Error: No se proporcionaron predicciones (array vacío).")
        return None

    # Si se predice una sola imagen, predictions[0] contiene las probabilidades
    probabilities = predictions[0]

    if len(probabilities) != len(class_names):
        print(f"Error: El número de probabilidades ({len(probabilities)}) no coincide "
              f"con el número de nombres de clases ({len(class_names)}). "
              "Verifica tu lista 'class_names'.")
        return None

    print("\n--- Resultados Detallados de la Predicción ---")

    # Almacenar probabilidades en un diccionario para fácil acceso
    results = {}
    for i, prob in enumerate(probabilities):
        class_name = class_names[i]
        results[class_name] = float(prob) # Convertir a float estándar para el dict
        print(f"  {class_name.capitalize()}: {prob:.8f} ({prob*100:.4f}%)") # Más decimales para valores pequeños

    # Encontrar la clase predicha (la de mayor probabilidad)
    predicted_class_index = np.argmax(probabilities)
    predicted_class_name = class_names[predicted_class_index]
    confidence = probabilities[predicted_class_index] * 100
    print("\n--- Clasificación ---")
    print(f"La imagen se clasifica como: {predicted_class_name.capitalize()} "
          f"con una confianza del {confidence:.2f}%.")

    results['predicted_class'] = predicted_class_name
    results['confidence'] = confidence

    return results
