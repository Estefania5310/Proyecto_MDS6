
def build_cnn_model(
    input_shape,
    num_classes,
    num_conv_layers=3,         # Número de bloques Conv2D-MaxPooling
    initial_filters=32,        # Número de filtros en la primera capa convolucional
    kernel_size=(3, 3),        # Tamaño del kernel para todas las capas convolucionales
    conv_activation='relu',    # Función de activación para capas convolucionales
    dense_units=None,          # Número de unidades en la capa densa antes de la salida
    optimizer_name='adam',     # Nombre del optimizador
    learning_rate=1e-3,        # Tasa de aprendizaje
    loss_function='categorical_crossentropy', # Función de pérdida
    metrics=['accuracy']       # Métricas a monitorear
):
    """
    Crea un modelo de red neuronal convolucional (CNN) configurable.

    Args:
        input_shape (tuple): La forma de las imágenes de entrada (alto, ancho, canales).
        num_classes (int): El número de clases para la clasificación.
        num_conv_layers (int): Número de bloques Conv2D + MaxPooling.
        initial_filters (int): Número de filtros en la primera capa convolucional.
                                Este número se duplica en cada capa subsiguiente.
        kernel_size (tuple): Tamaño del kernel para las capas Conv2D.
        conv_activation (str): Función de activación para las capas convolucionales.
        dense_units (int, optional): Número de unidades en la capa densa antes de la salida.
                                     Si es None, no se añade una capa densa intermedia.
        optimizer_name (str): Nombre del optimizador ('adam', 'sgd', 'rmsprop', etc.).
        learning_rate (float): Tasa de aprendizaje para el optimizador.
        loss_function (str): Función de pérdida a usar.
        metrics (list): Lista de métricas para la compilación del modelo.

    Returns:
        tf.keras.Model: El modelo CNN compilado.
    """
    model = keras.models.Sequential()

    # Añadir capas convolucionales y de pooling
    filters = initial_filters
    for i in range(num_conv_layers):
        if i == 0:
            # Primera capa: especificar input_shape
            model.add(keras.layers.Conv2D(filters, kernel_size, activation=conv_activation, input_shape=input_shape))
        else:
            # Capas subsiguientes: no necesitan input_shape
            model.add(keras.layers.Conv2D(filters, kernel_size, activation=conv_activation))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        filters *= 2 # Duplicar los filtros para la siguiente capa

    # Aplanar la salida para conectarla a las capas densas
    model.add(keras.layers.Flatten())

    # Opcional: Capa densa intermedia
    if dense_units is not None:
        model.add(keras.layers.Dense(units=dense_units, activation='relu'))

    # Capa de salida
    model.add(keras.layers.Dense(units=num_classes, activation='softmax'))

    # Configurar el optimizador
    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported.")

    # Compilar el modelo
    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=metrics)

    return model
