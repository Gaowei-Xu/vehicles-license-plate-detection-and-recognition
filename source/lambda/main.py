import os
import json
import boto3
import uuid
import numpy as np
from decord import VideoReader
from detector import CNLicensePlateDetector
from recognizer import CNLicensePlateRecognizer


s3 = boto3.resource('s3')


# Load the license plate detection & recognition model
license_plate_detector = CNLicensePlateDetector(model_root_path='/opt/ml/models/detector')
license_plate_recognizer = CNLicensePlateRecognizer(model_root_path='/opt/ml/models/recognizer')


def handler(event, context):
    """
    perform license plate detection and recognition

    :param event: passing event
    :param context: computing context
    :return: None
    """
    video_assets_bucket_name = os.environ['VideoAssetsS3BucketName']
    inference_results_bucket_name = os.environ['InferenceResultsS3BucketName']
    frame_interval = int(os.environ['FramesInterval'])

    video_clip_name = event['Records'][0]['s3']['object']['key']
    local_temp_path = '/tmp/' + video_clip_name

    # download mp4 video clip from s3 bucket
    s3.meta.client.download_file(video_assets_bucket_name, video_clip_name, local_temp_path)
    print('video_assets_bucket_name = {}'.format(video_assets_bucket_name))
    print('video_clip_name = {}'.format(video_clip_name))

    # load video clip
    vr = VideoReader(local_temp_path)
    duration = len(vr)

    # construct the event data, which will be dumped into S3 bucket later
    event_data = {
        'event_id': str(uuid.uuid4()),
        'video_source': os.path.join(video_assets_bucket_name, video_clip_name),
        'frames_amount': frame_interval,
        'frames_interval': frame_interval
    }
    frames_response_list = list()

    for frame_index in np.arange(0, duration, frame_interval):
        image = vr[frame_index].asnumpy()       # RGB order

        # step 1: detect all bounding boxes with their confidence score
        detect_boxes, detect_scores = license_plate_detector.detect(image)

        # step 2: recognize these bounding boxes
        recognize_boxes, recognize_scores, recognize_texts = license_plate_recognizer.recognize(
            image=image,
            boxes=detect_boxes,
            confidences=detect_scores,
            conf_thresh=0.25)

        print('Frame {}/{}: image shape = {}, detect_boxes = {}, detect_scores = {}, recognize_boxes = {}, '
              'recognize_scores = {}, recognize_texts = {}'.format(frame_index + 1, duration, image.shape,
                                                                   detect_boxes, detect_scores,
                                                                   recognize_boxes, recognize_scores, recognize_texts))

        frames_response_list.append(
            {
                'frame_index': int(frame_index),
                'detect_boxes': detect_boxes,           # shape = (N, 4)
                'detect_scores': detect_scores,         # shape = (N, 1)
                'recognize_boxes': recognize_boxes,     # shape = (N, 4)
                'recognize_scores': recognize_scores,   # shape = (N, 1)
                'recognize_texts': recognize_texts      # shape = (N, 1)
            }
        )

    # dump detection and recognition results into S3 bucket
    event_data['response'] = frames_response_list
    serialized_data = json.dumps(event_data, ensure_ascii=False, indent=4)
    print('event_data = {}'.format(event_data))

    # upload inference data into S3 bucket
    key_name =
    s3_dump_response = s3.Object(
        inference_results_bucket_name,
        video_clip_name + '_response.json').put(Body=serialized_data)
    print('s3_dump_response = {}'.format(s3_dump_response))
    print('Lambda Task Completed.')

    return None
