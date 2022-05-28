# this dockerfile creates the following new docker image: jchoi531/text-classification:1
FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY /model ./model
