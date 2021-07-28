import os
import json
import base64
import requests
import random
import time
import numpy as np
import boto3
from decord import VideoReader


# configuration
AI_INFERENCE_ENDPOINT = "https://your_inference_endpoint_url"
s3 = boto3.resource('s3')


def get_base64_encoding(full_path):
    with open(full_path, "rb") as f:
        data = f.read()
        image_base64_enc = base64.b64encode(data)
        image_base64_enc = str(image_base64_enc, 'utf-8')

    return image_base64_enc


def handler(event, context):
    bucket_name = os.environ['S3BucketName']
    mp4_video_clip_name = event['Records'][0]['s3']['object']['key']
    local_temp_path = '/tmp/' + mp4_video_clip_name

    # download mp4 video clip from s3 bucket
    s3.meta.client.download_file(bucket_name, mp4_video_clip_name, local_temp_path)

    # load video clip and pick up key frame
    vr = VideoReader(local_temp_path)
    duration = len(vr)
    index = random.randint(0, duration)
    frame_selected = vr[index].asnumpy()
    print('The video {} contains {} frames'.format(local_temp_path, duration))
    print('Frame shape = {}'.format(frame_selected.shape))
    print('type(frame_selected) = {}'.format(type(frame_selected)))

    # execute base64 encoding for key frame
    image_base64_enc = base64.b64encode(frame_selected)
    image_base64_enc = str(image_base64_enc, 'utf-8')
    print('image_base64_enc = {}'.format(image_base64_enc))

    # send HTTPS request to AI Service
    request_body = {
        "timestamp": str(time.time()),
        "request_id": 1242322,
        "image_base64_enc": image_base64_enc
    }

    response = requests.post(AI_INFERENCE_ENDPOINT, data=json.dumps(request_body))
    print('AI Inference Response = {}'.format(response))

    # Save response into DynamoDB/S3/RDS, etc.
    pass

    return response
