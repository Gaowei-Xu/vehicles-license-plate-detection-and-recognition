import os
import cv2
from detector import CNLicensePlateDetector
from recognizer import CNLicensePlateRecognizer


if __name__ == '__main__':
    license_plate_detector = CNLicensePlateDetector(model_root_path='./models/detector', enable_vis=False)
    license_plate_recognizer = CNLicensePlateRecognizer(model_root_path='./models/recognizer')

    test_images_root_dir = './images'
    image_names = [name for name in os.listdir(test_images_root_dir) if name.endswith('.jpeg')]
    image_names = sorted(image_names)

    for index, image_name in enumerate(image_names):
        print('Processing image {} / {}...'.format(index + 1, len(image_names)))
        image_full_path = os.path.join(test_images_root_dir, image_name)
        image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)

        # step 1: detect all bounding boxes with their confidence score
        boxes, confidences = license_plate_detector.detect(image)

        # step 2: recognize these bounding boxes
        boxes, confidences, license_texts = license_plate_recognizer.recognize(
            image=image,
            boxes=boxes,
            confidences=confidences,
            conf_thresh=0.85)

        print('boxes = {}'.format(boxes))
        print('confidences = {}'.format(boxes))
        print('license_texts = {}'.format(boxes))

        break








