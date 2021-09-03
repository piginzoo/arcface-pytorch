# From pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
From tensorflow/tensorflow:2.1.0-gpu-py3

MAINTAINER piginzoo

RUN cp /etc/apt/sources.list /etc/apt/sources.list.backup
ADD deploy/sources.list /etc/apt/sources.list
RUN apt-get update

# support cv2
RUN apt install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0

RUN mkdir /root/.pip
ADD deploy/pip.conf /root/.pip
RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
ADD requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

# time zone
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true
RUN apt install tzdata
RUN rm /etc/localtime
RUN ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo "Asia/Shanghai">/etc/timezone