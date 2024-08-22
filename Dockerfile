FROM python:3.10-slim

RUN pip install --upgrade pip

RUN apt-get -o Acquire::Check-Valid-Until=false update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8000

CMD ["chainlit","run","-w","app.py"]
