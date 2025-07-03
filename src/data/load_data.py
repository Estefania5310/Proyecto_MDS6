import tensorflow as tf

def cargar_datasets(train_dir, val_dir, test_dir, img_height=180, img_width=180, batch_size=32):
    """
    Carga datasets desde carpetas de imágenes organizadas por clase.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(img_height, img_width), batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=(img_height, img_width), batch_size=batch_size)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=(img_height, img_width), batch_size=batch_size)

    return train_ds, val_ds, test_ds
