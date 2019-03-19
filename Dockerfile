FROM ubuntu:16.04

RUN apt-get update -qqq
RUN apt-get install python3-pip

RUN mkdir /opt/
COPY requirements.txt /opt/

RUN pip3 install -r /opt/requirements.txt
COPY network_analysis.py /opt/

ENTRYPOINT ["python3", "/opt/network_analysis.py"]
