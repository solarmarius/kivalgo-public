FROM python:3.12

WORKDIR /

COPY ./requirements.txt /requirements.txt

# Install requirements
RUN pip install --no-cache-dir -r /requirements.txt

COPY . /
COPY ./data /data
COPY ./database /database
COPY ./loaders /loaders
COPY ./models /models
COPY ./services /services
COPY ./utils /utils

RUN \
    apt-get update && \
    apt-get install -y sudo curl git && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
    sudo apt-get install git-lfs

RUN cd models

RUN git clone https://huggingface.co/thenlper/gte-small
RUN git clone https://huggingface.co/microsoft/Phi-3.5-mini-instruct

RUN cd ..

ENV PYTHONPATH=/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]