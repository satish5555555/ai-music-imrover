# Stage 1: Web UI Build
FROM node:18-alpine AS ui
WORKDIR /app/webui
COPY webui/package.json webui/package-lock.json* ./
RUN npm ci
COPY webui/ .
RUN npm run build

# Stage 2: Backend (FastAPI + PyTorch CPU)
# Use PyTorch official image that supports both x86_64 and arm64
FROM pytorch/pytorch:2.2.2-cpu

WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ ./server/
COPY config.yaml ./
COPY --from=ui /app/webui/dist ./server/static

WORKDIR /app/server
EXPOSE 8000
CMD ["python", "run_uvicorn.py"]

