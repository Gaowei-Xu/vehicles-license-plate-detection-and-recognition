FROM public.ecr.aws/lambda/python:3.8

# Install dependencies
RUN yum -y update && \
    yum install -y wget \
    unzip \
    opencv-python \
    opencv-devel

RUN mkdir -p /opt/ml/models/
RUN wget -c https://ip-camera-ai-saas.s3.amazonaws.com/models/license_plates_detection/detector.zip -O /opt/ml/models/detector.zip
RUN wget -c https://ip-camera-ai-saas.s3.amazonaws.com/models/license_plates_recognition/recognizer.zip -O /opt/ml/models/recognizer.zip
RUN cd /opt/ml/models/ && unzip detector.zip && rm detector.zip
RUN cd /opt/ml/models/ && unzip recognizer.zip && rm recognizer.zip

RUN pip install --no-cache decord==0.6.0 \
    tensorflow==2.5.0 \
    requests==2.25.1

COPY main.py ./
COPY detector.py ./
COPY recognizer.py ./

CMD [ "main.handler" ]
