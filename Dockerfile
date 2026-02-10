# OnsetLab - Multi-stage build
# Stage 1: Build frontend static files
# Stage 2: Python backend with SDK

# ---- Stage 1: Build website (landing page) ----
FROM node:20-slim AS website-builder

WORKDIR /build/website
COPY website/package.json website/package-lock.json ./
RUN npm ci
COPY website/ ./
# vite.config.js outputs to ../web/backend/static/site (i.e. /build/web/backend/static/site)
RUN npm run build

# ---- Stage 2: Build playground (frontend app) ----
FROM node:20-slim AS playground-builder

WORKDIR /build/web/frontend
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci
COPY web/frontend/ ./
# vite.config.ts outputs to ../backend/static/playground (i.e. /build/web/backend/static/playground)
RUN npm run build

# ---- Stage 3: Python backend ----
FROM python:3.11-slim

WORKDIR /app

# Install Node.js (needed for MCP servers that run via npx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY web/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy OnsetLab SDK
COPY onsetlab/ /app/onsetlab/

# Copy backend application
COPY web/backend/app/ /app/app/

# Copy built static files from frontend stages
COPY --from=website-builder /build/web/backend/static/site/ /app/static/site/
COPY --from=playground-builder /build/web/backend/static/playground/ /app/static/playground/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
