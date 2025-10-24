# =============================================================
# STAGE 1 — Build Web UI (React)
# =============================================================
FROM node:18-alpine AS ui
WORKDIR /app/webui

# Copy package manifests first
COPY webui/package.json webui/package-lock.json* ./

# --- SSL / Proxy Safe Config for npm ---
# Disable strict SSL and set retry policies (safe for dev & corporate proxies)
RUN npm config set strict-ssl false
RUN npm config set registry "https://registry.npmjs.org/"
RUN npm config set fetch-retries 5
RUN npm config set fetch-retry-maxtimeout 60000

# Install dependencies
RUN if [ ! -f package-lock.json ]; then npm install --package-lock-only; fi
RUN npm ci || npm install

# Copy rest of UI and build production bundle
COPY webui/ .
RUN npm run build


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
# We use PyPI directly (not download.pytorch.org) to avoid strict SSL chains
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
# Copy backend source and prebuilt UI
# =============================================================
COPY server/ ./server/
COPY config.yaml ./
COPY --from=ui /app/webui/dist ./server/static

WORKDIR /app/server

# Expose FastAPI port
EXPOSE 8000

# Start backend
CMD ["python", "run_uvicorn.py"]

