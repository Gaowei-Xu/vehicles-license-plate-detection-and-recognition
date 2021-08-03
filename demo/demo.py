import cv2
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# configuration
response_json_full_path = '../videos/test_2.ts'
raw_input_video_clip_full_path = '../videos/test_2.ts_response.json'


def plot_detection_recognition(image, bounding_boxes, scores, recognize_texts, textColor=(255, 0, 255), textSize=24):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for index, bbox in enumerate(bounding_boxes):
        text = recognize_texts[index][0]
        [x_min, y_min, x_max, y_max] = bbox
        draw = ImageDraw.Draw(image)
        fontStyle = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", textSize, encoding="utf-8")
        draw.text((x_min-35, y_min-35), text, textColor, font=fontStyle)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=textColor, fill=None, width=2)

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def run(vis_root_dir):
    if not os.path.exists(vis_root_dir):
        os.makedirs(vis_root_dir)

    inference_result = json.load(open(raw_input_video_clip_full_path, 'r'))
    duration = inference_result['frames_amount']
    frames_interval = inference_result['frames_interval']
    response = inference_result['response']
    lut = dict()
    for item in response:
        lut[item['frame_index']] = {
            'detect_boxes': item['detect_boxes'],
            'detect_scores': item['detect_scores'],
            'recognize_boxes': item['recognize_boxes'],
            'recognize_scores': item['recognize_scores'],
            'recognize_texts': item['recognize_texts']
        }

    cap = cv2.VideoCapture(response_json_full_path)
    success = cap.read()

    index = 0
    while success:
        success, frame = cap.read()
        if frame is None:
            break

        if index not in lut.keys():
            index += 1
            continue

        bbox_coords = lut[index]['recognize_boxes']
        bbox_scores = lut[index]['recognize_scores']
        recognize_texts = lut[index]['recognize_texts']

        # convert to numpy array
        bbox_coords = np.array(bbox_coords).astype(np.float64)
        bbox_scores = np.array(bbox_scores)

        # visualization
        image = plot_detection_recognition(frame, bbox_coords, bbox_scores, recognize_texts)
        cv2.imwrite(os.path.join(vis_root_dir, 'frame_{}.jpg'.format(index)), image)

        index += 1


if __name__ == '__main__':
    run(vis_root_dir='./frames')

