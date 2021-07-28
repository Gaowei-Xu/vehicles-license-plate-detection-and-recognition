FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get -y update && \
    apt-get install -y wget \
    python3 \
    zip \
    python3-opencv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/ml/models/
RUN wget -c https://ip-camera-ai-saas.s3.amazonaws.com/models/license_plates_detection/detector.zip -O /opt/ml/models/detector.zip
RUN wget -c https://ip-camera-ai-saas.s3.amazonaws.com/models/license_plates_recognition/recognizer.zip -O /opt/ml/models/recognizer.zip
RUN cd /opt/ml/models/ && unzip detector.zip && rm detector.zip
RUN cd /opt/ml/models/ && unzip recognizer.zip && rm recognizer.zip


RUN python3 -m pip install --upgrade pip==21.2.1
RUN pip3 install --no-cache decord==0.6.0 \
    tensorflow==2.5.0 \
    requests==2.25.1

COPY main.py ./
COPY detector.py ./
COPY recognizer.py ./

CMD [ "main.handler" ]
