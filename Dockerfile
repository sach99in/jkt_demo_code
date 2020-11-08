FROM python:3.6-slim-stretch
RUN apt-get -y update
RUN apt-get -y install build-essential
RUN apt-get -y install libev-dev
RUN mkdir /preds
WORKDIR /preds
COPY src_api/   src
WORKDIR /preds/src
RUN pip3 install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r requirements.txt
RUN python -m spacy download en_core_web_md
EXPOSE 8000
ENTRYPOINT ["python", "main.py"]
