import os
import json
import boto3
import uuid
import numpy as np
from decord import VideoReader
from detector import CNLicensePlateDetector
from recognizer import CNLicensePlateRecognizer


s3 = boto3.resource('s3')
dynamodb = boto3.client('dynamodb')


# Load the license plate detection & recognition model
license_plate_detector = CNLicensePlateDetector(model_root_path='/opt/ml/models/detector')
license_plate_recognizer = CNLicensePlateRecognizer(model_root_path='/opt/ml/models/recognizer')

# process a frame every frame_interval
frame_interval = 10


def handler(event, context):
    """
    perform license plate detection and recognition

    :param event: passing event
    :param context: computing context
    :return: None
    """
    bucket_name = os.environ['S3BucketName']
    ddb_table_name = os.environ['DynamoDBTableName']
    ddb_primary_key = os.environ['DynamoDBPrimaryKey']

    video_clip_name = event['Records'][0]['s3']['object']['key']
    local_temp_path = '/tmp/' + video_clip_name

    # download mp4 video clip from s3 bucket
    s3.meta.client.download_file(bucket_name, video_clip_name, local_temp_path)
    print('bucket_name = {}'.format(bucket_name))
    print('video_clip_name = {}'.format(video_clip_name))

    # load video clip
    vr = VideoReader(local_temp_path)
    duration = len(vr)

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

        if len(detect_boxes) == 0:
            print('Frame {}/{}: image shape = {}, detect_boxes = {}, detect_scores = {}'.format(
                frame_index + 1, duration, image.shape, detect_boxes, detect_scores))
            continue

        # step 3: save response into dynamodb
        detection_response = json.dumps({
            'boxes': detect_boxes,              # shape = (N, 4)
            'confidences': detect_scores        # shape = (N, 1)
        })
        recognition_response = json.dumps({
            'boxes': recognize_boxes,           # shape = (N, 4)
            'confidences': recognize_scores,    # shape = (N, 1)
            'texts': recognize_texts            # shape = (N, 1)
        })

        print('Frame {}/{}: image shape = {}, detection = {}, recognition = {}'.format(
            frame_index + 1, duration, image.shape, detection_response, recognition_response))

        event_id = str(uuid.uuid4())
        insert_item = {
            ddb_primary_key: {'S': event_id},
            'video_source': {'S': os.path.join(bucket_name, video_clip_name)},
            'frames_amount': {'S': str(duration)},
            'frames_interval': {'S': str(frame_interval)},
            'frame_index': {'S': str(frame_index)},
            'detect_response': {'S': detection_response},
            'recognize_response': {'S': recognition_response}
        }

        response = dynamodb.put_item(
            TableName=ddb_table_name,
            Item=insert_item
        )

    print('Lambda Task Completed.')

    return None
