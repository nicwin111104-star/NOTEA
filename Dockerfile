FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

COPY . .

RUN git lfs install
RUN git lfs pull || true

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}