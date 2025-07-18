
def create_model_0(input_shape, num_classes):
    """
    Crea un modelo de red neuronal convolucional (CNN).

    Args:
        input_shape (tuple): La forma de las imágenes de entrada (alto, ancho, canales).
                             Ej: (180, 180, 3) para imágenes a color.
        num_classes (int): El número de clases para la clasificación.

    Returns:
        tf.keras.Model: El modelo CNN compilado.
    """
    model = keras.models.Sequential([
        # Primera capa convolucional y de pooling
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),

        # Segunda capa convolucional y de pooling
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        # Tercera capa convolucional y de pooling
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        # Aplanar la salida para conectarla a las capas densas
        keras.layers.Flatten(),

        # Capa densa
        keras.layers.Dense(units=num_classes, activation='softmax') # Capa de salida con activación softmax para clasificación multiclase
    ])

    # Compilar el modelo
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', # Usar para clasificación multiclase con one-hot encoding
                  metrics=['accuracy'])

    return model
