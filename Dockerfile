# Stage 1: Web UI Build
FROM node:18-alpine AS ui
WORKDIR /app/webui
COPY webui/package.json webui/package-lock.json* ./
RUN npm ci
COPY webui/ .
RUN npm run build

# Stage 2: Backend (Flask + PyTorch)
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ ./server/
COPY config.yaml ./
COPY --from=ui /app/webui/dist ./server/static

WORKDIR /app/server
EXPOSE 8000
CMD ["python", "app.py"]
