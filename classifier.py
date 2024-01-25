import tensorflow as tf
import tensorflow_hub as hub
import PIL.Image as Image
import numpy as np
from skimage import io
from skimage.transform import resize
import sys

np.set_printoptions(threshold=sys.maxsize)
IMAGE_SHAPE = (224, 224)
CHANNEL = 3
mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

classifier_model = mobilenet_v2

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])


def image_classifier(image_array):

    # image_array must be upsampled to IMAGE_SHAPE
    image_array = np.resize(image_array, (IMAGE_SHAPE[0], IMAGE_SHAPE[1], CHANNEL))

    # expand dimension
    image_expended_dim = np.expand_dims(image_array, axis=0)

    # perform model classification
    result = classifier.predict(image_expended_dim)

    # get predicted class label
    predicted_class = tf.math.argmax(result[0], axis=-1)

    # label file path
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

    # get the imagenet labels
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    # predicted class
    predicted_class_name = imagenet_labels[predicted_class]

    return (image_expended_dim, result, predicted_class, predicted_class_name)




