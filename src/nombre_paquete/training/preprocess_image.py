from keras.applications.efficientnet import preprocess_input
def preprocess_image(image, label):
    image = keras.applications.efficientnet.preprocess_input(image)
    return image, label
