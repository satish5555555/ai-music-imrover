# =============================================================
# STAGE 1 — Build Web UI (React / Vite)
# =============================================================
FROM node:18-alpine AS ui
WORKDIR /app/webui

# Copy package manifests first
COPY webui/package.json webui/package-lock.json* ./

# --- SSL / Proxy Safe Config for npm ---
RUN npm config set strict-ssl false
RUN npm config set registry "https://registry.npmjs.org/"
RUN npm config set fetch-retries 5
RUN npm config set fetch-retry-maxtimeout 60000

# Install dependencies
RUN if [ ! -f package-lock.json ]; then npm install --package-lock-only; fi
RUN npm ci || npm install

# Copy rest of the frontend source
COPY webui/ .

# --- Force Vite install if missing (prevents 'vite: command not found') ---
RUN npm install vite --save-dev || true

# --- Build frontend using Vite ---
RUN echo "🏗️  Building frontend..." && npm run build || (npm run build)

# --- Verify that dist folder was created ---
RUN test -d dist && ls -la dist || (echo "❌ Vite build failed — dist missing!" && exit 1)

# =============================================================
# STAGE 2 — Backend (FastAPI + PyTorch)
# =============================================================
FROM python:3.11-slim

WORKDIR /app

# --- System dependencies for audio, networking, and SSL ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

# --- Environment Variables for SSL bypass (safe for build env) ---
ENV NODE_TLS_REJECT_UNAUTHORIZED=0
ENV PIP_NO_VERIFY_CERTS=1
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV PYTHONWARNINGS=ignore

# --- SSL Safe Config for pip ---
RUN pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org download.pytorch.org pytorch.org github.com"
RUN pip config set global.cert /etc/ssl/certs/ca-certificates.crt
RUN pip config set global.disable-pip-version-check true
RUN pip config set global.timeout 300
RUN pip config set global.retries 5

# --- Copy and install Python dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================
# FIXED SECTION — Install PyTorch safely behind SSL proxies
# =============================================================
RUN echo "Installing PyTorch and Torchaudio safely..." && \
    pip install torch==2.3.1 torchaudio==2.3.1 --no-cache-dir \
        --trusted-host pypi.org \
        --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org \
        --trusted-host pytorch.org \
        --trusted-host github.com \
        --timeout 300 || \
    (echo "⚠️ SSL issue detected, retrying with SSL disabled..." && \
     PYTHONWARNINGS=ignore \
     PIP_NO_VERIFY_CERTS=1 \
     pip install torch==2.3.1 torchaudio==2.3.1 --no-cache-dir \
        --trusted-host pypi.org \
        --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org \
        --trusted-host pytorch.org \
        --trusted-host github.com \
        --timeout 300)

# =============================================================
# Copy backend source and built UI
# =============================================================
COPY server/ ./server/
COPY config.yaml ./
COPY --from=ui /app/webui/dist ./server/static

WORKDIR /app/server

# Expose FastAPI port
EXPOSE 8000

# =============================================================
# HEALTHCHECK — ensures container is marked healthy only if API works
# =============================================================
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

# =============================================================
# Start backend (production mode)
# =============================================================
CMD ["python", "run_uvicorn.py"]

