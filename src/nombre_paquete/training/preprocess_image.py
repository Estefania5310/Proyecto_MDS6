from tensorflow.keras.applications.efficientnet import preprocess_input
def preprocess_image(image, label):
    image = preprocess_input(image)
    return image, label
