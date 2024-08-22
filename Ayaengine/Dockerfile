FROM python:3.10-slim

RUN pip install --upgrade pip

RUN apt-get -o Acquire::Check-Valid-Until=false update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    tesseract-ocr \
    ghostscript \ 
    && rm -rf /var/lib/apt/lists/*

RUN apt-get  -o Acquire::Check-Valid-Until=false update && apt-get install -y  \
    libtesseract-dev \
    libleptonica-dev \ 
    tesseract-ocr-fas  \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-jpn \
    ghostscript \
    poppler-utils


WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python","model.py"]
