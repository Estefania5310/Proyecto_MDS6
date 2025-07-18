from tensorflow import keras
def preprocess_image(image, label):
    image = keras.applications.efficientnet.preprocess_input(image)
    return image, label
