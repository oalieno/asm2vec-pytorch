FROM python:3.10.11-slim

ADD . /asm2vec-pytorch
WORKDIR asm2vec-pytorch

RUN apt-get update && apt-get install -y --no-install-recommends \
    unixodbc-dev \
    unixodbc \
    libpq-dev && \
    pip install -r requirements.txt && \
    python setup.py install

CMD ["/bin/sh"]
