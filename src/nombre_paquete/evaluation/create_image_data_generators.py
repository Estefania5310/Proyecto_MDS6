
def create_image_data_generators(
    train_dir,
    val_dir,
    test_dir,
    img_height=180,
    img_width=180,
    batch_size=32,
    class_mode="categorical",
    seed= 123,
):
    """
    Crea generadores de imágenes usando ImageDataGenerator con y sin aumento de datos.

    Args:
        train_dir (str): Ruta al directorio de entrenamiento.
        val_dir (str): Ruta al directorio de validación.
        test_dir (str): Ruta al directorio de prueba.
        img_height (int): Altura deseada de las imágenes.
        img_width (int): Ancho deseado de las imágenes.
        batch_size (int): Tamaño del lote.
        class_mode (str): Modo de clase para el generador ('categorical', 'binary', etc.).

    Returns:
        tuple: train_generator, val_generator, test_generator
    """

    # Data Augmentation y preprocesamiento para el conjunto de entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,  # Normalizar los valores de píxeles a [0, 1]
        rotation_range=20,  # Rotar imágenes aleatoriamente hasta 20 grados
        width_shift_range=0.2,  # Desplazar imágenes horizontalmente aleatoriamente
        height_shift_range=0.2,  # Desplazar imágenes verticalmente aleatoriamente
        shear_range=0.2,  # Aplicar transformaciones de cizallamiento
        zoom_range=0.2,  # Aplicar zoom aleatorio
        horizontal_flip=True,  # Voltear imágenes horizontalmente aleatoriamente
        fill_mode="nearest",  # Rellenar píxeles nuevos después de transformaciones
    )

    # Preprocesamiento para los conjuntos de validación y prueba (solo reescalado)
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    print("Creando generador de entrenamiento...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=class_mode,
        seed=seed,

    )

    print("Creando generador de validación...")
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=class_mode,
        seed=seed,
    )

    print("Creando generador de prueba...")
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False,  # Importante no barajar el conjunto de prueba
        seed=seed,
    )

    return train_generator, val_generator, test_generator
