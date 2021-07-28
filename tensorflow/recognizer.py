import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class CNLicensePlateRecognizer(object):
    def __init__(self, model_root_path):
        self._model_input_image_width = 116
        self._model_input_image_height = 40

        # DO NOT MODIFY
        self._provinces = [
            "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽",
            "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘",
            "青", "宁", "新", "警", "学", "O"]
        self._alphabets = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
        self._ads = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7',
            '8', '9', 'O']

        # the model should be placed before tensorflow session
        self._model_root_path = model_root_path
        self._recognizer = tf.saved_model.load(self._model_root_path, tags=[tag_constants.SERVING])
        self._recognizer.signatures['serving_default']

    def pre_process(self, roi):
        """
        resize roi to recognition model's input

        :param roi: cv2 image data with channel order BGR
        :return:
        """
        image_data = cv2.resize(roi, (self._model_input_image_width, self._model_input_image_height))
        image_data = image_data[:, :, ::-1]     # convert BGR to RGB
        image_data_float = image_data.astype('float32')
        image_data_float /= 255.0
        image_data_float = np.asarray([image_data_float]).astype(np.float32)
        image_batch_data = tf.constant(image_data_float)
        return image_batch_data

    def get_text(self, predictions):
        """
        get license plate text

        :param predictions: tensorflow output tensor
        :return:
        """
        p1 = predictions[0]
        p2 = predictions[1]
        p3 = predictions[2]
        p4 = predictions[3]
        p5 = predictions[4]
        p6 = predictions[5]
        p7 = predictions[6]

        p1_index = np.argmax(p1)
        p2_index = np.argmax(p2)
        p3_index = np.argmax(p3)
        p4_index = np.argmax(p4)
        p5_index = np.argmax(p5)
        p6_index = np.argmax(p6)
        p7_index = np.argmax(p7)

        text = str()
        text += self._provinces[p1_index]
        text += self._alphabets[p2_index]
        text += self._ads[p3_index]
        text += self._ads[p4_index]
        text += self._ads[p5_index]
        text += self._ads[p6_index]
        text += self._ads[p7_index]

        return text

    def recognize(self, image, boxes, confidences, conf_thresh=0.85):
        """
        recognize the ROIs

        :param image: original image
        :param boxes: bounding boxes with shape [N, 4]
        :param confidences: confidences with shape [N, 1]
        :param conf_thresh: threshold of bounding boxes
        :return: a dictionary with text and its corresponding probabilities
        """
        amount = boxes.shape[0]
        rec_boxes = list()
        rec_scores = list()
        rec_texts = list()

        for index in range(amount):
            score = confidences[index][0]
            if score < conf_thresh:
                continue

            bounding_box = boxes[index]
            [x_min, y_min, x_max, y_max] = bounding_box
            roi = image[y_min:y_max, x_min:x_max, :]
            roi_batch_data = self.pre_process(roi=roi)
            predictions = self._recognizer(roi_batch_data)
            license_text = self.get_text(predictions=predictions)

            rec_boxes.append(bounding_box)
            rec_scores.append(score)
            rec_texts.append(license_text)

        return rec_boxes, rec_scores, rec_texts

