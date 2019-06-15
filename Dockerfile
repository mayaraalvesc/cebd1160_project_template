FROM ubuntu:latest

RUN apt-get update \
  && apt-get -y install python3-pip \
  && pip3 install  numpy \
  && pip3 install  pandas matplotlib \
  && pip3 install  seaborn \
  && pip3 install  sklearn \


COPY dataset-processor.py .

COPY wine.data .

CMD ["python3","-u","project.py"]

