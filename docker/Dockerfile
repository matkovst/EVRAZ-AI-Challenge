FROM continuumio/anaconda3

COPY environment.yaml /home/environment.yaml

RUN apt-get update \
    && apt-get install -y --no-install-recommends git wget unzip

RUN /opt/conda/bin/conda env create -f /home/environment.yaml
RUN echo "conda activate cdcl" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN cd /home \
    && git clone https://github.com/kevinlin311tw/CDCL-human-part-segmentation.git EVRAZ-AI-Challenge \
    && cd EVRAZ-AI-Challenge \
    && bash fetch_data.sh