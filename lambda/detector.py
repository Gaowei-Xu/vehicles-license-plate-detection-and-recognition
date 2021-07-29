import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class CNLicensePlateDetector(object):
    def __init__(self, model_root_path):
        self._model_root_path = model_root_path
        self._detector = tf.saved_model.load(self._model_root_path, tags=[tag_constants.SERVING])
        self._detector.signatures['serving_default']

        self._height, self._width, self._channels = None, None, None
        self._scale_height_ratio, self._scale_width_ratio = 1.0, 1.0

    def pre_process(self, image):
        """
        resize image to yolo-v4 input size

        :param image: cv2 image data with channel order RGB
        :return:
        """
        self._height, self._width, self._channels = image.shape

        self._scale_height_ratio = 512.0 / self._height
        self._scale_width_ratio = 512.0 / self._width
        resized_image = cv2.resize(image, (512, 512))
        resized_normalized_image = resized_image / 255.0
        images_data = np.asarray([resized_normalized_image]).astype(np.float32)
        image_batch_data = tf.constant(images_data)
        return image_batch_data

    def post_process(self, detections, iou_threshold=0.45, score_threshold=0.05):
        """
        scale the detections back to its original image size

        :param detections: yolo-v4 detections
        :param iou_threshold: threshold for IoU
        :param score_threshold: threshold for confidence
        :return:
        """
        boxes = detections[:, :, 0:4]
        pred_conf = detections[:, :, 4:]

        bbox_coords, bbox_scores, class_ids, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold
        )

        ret_bbox_coords = list()    # shape = (N, 4)
        ret_bbox_scores = list()    # shape = (N, 1)
        ret_class_ids = list()      # shape = (N, 1)

        for index in range(valid_detections[0]):
            [scale_y_min, scale_x_min, scale_y_max, scale_x_max] = bbox_coords[0][index]
            confidence = bbox_scores[0][index]
            cls_id = class_ids[0][index]

            ret_bbox_coords.append([
                int(scale_x_min * 512.0 / self._scale_width_ratio),
                int(scale_y_min * 512.0 / self._scale_height_ratio),
                int(scale_x_max * 512.0 / self._scale_width_ratio),
                int(scale_y_max * 512.0 / self._scale_height_ratio),
            ])
            ret_bbox_scores.append([float(confidence)])
            ret_class_ids.append([int(cls_id)])

        return ret_bbox_coords, ret_bbox_scores, ret_class_ids

    def detect(self, image):
        """
        perform yolo-v4 detection inference

        :param image: cv2 image data with channel order BGR
        :return:
        """
        image_batch_data = self.pre_process(image)
        detections = self._detector(image_batch_data)
        boxes, scores, cls_ids = self.post_process(detections=detections)

        return boxes, scores


