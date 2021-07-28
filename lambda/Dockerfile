FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get -y update && \
    apt-get install -y wget \
    python3 \
    python3-opencv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN wget -c https://ip-camera-ai-saas.s3.amazonaws.com/models/pets_detection/yolov4-pets.trt -O /opt/ml/model/yolov4-pets.trt
RUN wget -c https://ip-camera-ai-saas.s3.amazonaws.com/models/pets_detection/yolov4-pets.trt -O /opt/ml/model/yolov4-pets.trt
RUN wget -c https://ip-camera-ai-saas.s3.amazonaws.com/models/pets_detection/yolov4-pets.trt -O /opt/ml/model/yolov4-pets.trt
RUN wget -c https://ip-camera-ai-saas.s3.amazonaws.com/models/pets_detection/yolov4-pets.trt -O /opt/ml/model/yolov4-pets.trt

RUN python3 -m pip install --upgrade pip==21.2.1
RUN pip3 install --no-cache decord==0.6.0 \
    requests==2.25.1 \
    numpy==1.21.0

COPY main.py ./
CMD [ "main.handler" ]
