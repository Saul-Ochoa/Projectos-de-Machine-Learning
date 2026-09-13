FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/

# data/ y models/ se montan como volúmenes en runtime (ver README):
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models <image> python src/models/train_model.py
