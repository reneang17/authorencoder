FROM nvidia/cuda:10.1-base-ubuntu18.04

RUN apt-get update; \

    apt-get install -y software-properties-common; \

    add-apt-repository ppa:deadsnakes/ppa; \

    apt-get install python3.7 curl -y ; \

    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py ; \

    python3 get-pip.py ; \

    pip3 install requirements.txt ; \

    pip3 install -U spacy ; \

    python3 -m spacy download en_core_web_sm

RUN apt-get install python3-tk -y

COPY resources /Users/rene/work/docker_authorencoder/authorencoder

WORKDIR /usr/local/authoencoder
