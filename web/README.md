# OnsetLab Playground

Web-based demo for trying OnsetLab without installation.

## Architecture

```
frontend/     → React + Vite + Tailwind (Vercel)
backend/      → FastAPI + OnsetLab SDK (Railway)
```

## Local Development

### 1. Get FREE Groq API Key

1. Go to https://console.groq.com
2. Sign up (free, no credit card needed!)
3. Create API key

### 2. Backend

```bash
cd backend

# Set environment variable
export GROQ_API_KEY="your-groq-api-key"

# Run server (from web/ directory)
cd ..
source ../myvenv/bin/activate
uvicorn backend.app.main:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env

# Run dev server
npm run dev
```

Open http://localhost:5173

## Deployment

### Backend → Railway

1. Create new project on [Railway](https://railway.app)
2. Connect GitHub repo
3. Set root directory to `web/backend`
4. Add environment variables:
   - `GROQ_API_KEY` (from console.groq.com)
   - `CORS_ORIGINS` (your Vercel URL)

### Frontend → Vercel

1. Create new project on [Vercel](https://vercel.com)
2. Connect GitHub repo
3. Set root directory to `web/frontend`
4. Add environment variable:
   - `VITE_API_URL` (your Railway URL)

## Environment Variables

### Backend
| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key (FREE at console.groq.com) |
| `CORS_ORIGINS` | Allowed origins (comma-separated) |
| `RATE_LIMIT_REQUESTS` | Requests per session (default: 5) |

### Frontend
| Variable | Description |
|----------|-------------|
| `VITE_API_URL` | Backend API URL |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/session` | Get/create session |
| GET | `/api/tools` | List available tools |
| GET | `/api/models` | List available models |
| POST | `/api/chat` | Send message |
| POST | `/api/export` | Export agent config |

## Getting Groq API Key (FREE!)

1. Sign up at https://console.groq.com
2. Click "API Keys" in sidebar
3. Create new key
4. **No credit card required!**

## Costs

| Component | Cost |
|-----------|------|
| Vercel (frontend) | Free |
| Railway (backend) | ~$5/mo |
| Groq API | **FREE** |
| **Total** | **~$5/mo** |
