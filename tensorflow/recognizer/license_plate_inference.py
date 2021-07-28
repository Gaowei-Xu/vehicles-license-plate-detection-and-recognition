import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import numpy as np


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class CNLicensePlateRecognizer(object):
    def __init__(self, model_root_path):
        self._model_root_path = model_root_path
        self._recognizer = tf.saved_model.load(self._model_root_path, tags=[tag_constants.SERVING])
        self._recognizer.signatures['serving_default']

    def recognize(self):
        """
        perform yolo-v4 detection inference

        :param image: cv2 image data with channel order BGR
        :return:
        """
        image = np.zeros(shape=[1, 40, 116, 3])
        images_data = np.asarray(image).astype(np.float32)
        predictions = self._recognizer(images_data)
        print(predictions)


if __name__ == '__main__':
    recognizer = CNLicensePlateRecognizer(model_root_path='../models/recognizer')
    recognizer.recognize()
