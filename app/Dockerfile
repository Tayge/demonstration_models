FROM tiangolo/uwsgi-nginx-flask:python3.8

COPY ./ /app

RUN pip install -r requirements.txt
RUN pip install --no-cache-dir tensorflow
# opencv dependencies
RUN  apt-get update \
  && apt-get install ffmpeg libsm6 libxext6  -y