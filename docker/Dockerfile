FROM ubuntu:18.04


RUN apt-get update \
    && apt-get install -y python3.8 \
    python3-pip \
    poppler-utils \
    tesseract-ocr \
    htop \
    libtesseract-dev \
    tesseract-ocr-fra \
    ffmpeg libsm6 libxext6  -y \
    locales

RUN pip3 install --upgrade pip
RUN pip3 install opencv-python


ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8



WORKDIR /usr/src/webapp 

COPY app ./app
COPY static ./static
COPY templates ./templates
COPY requirements.txt .

 


RUN pip3 install -r requirements.txt




CMD uvicorn app.app:app --host 0.0.0.0 --port 80  --forwarded-allow-ips '*' --reload --log-config ./app/log.ini
