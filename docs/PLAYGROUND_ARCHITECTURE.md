# OnsetLab Playground - Architecture

A web-based demo allowing users to try OnsetLab without installation.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER BROWSER                                │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     REACT FRONTEND (Vercel)                        │ │
│  │                                                                    │ │
│  │  • Chat interface                                                  │ │
│  │  • Tool configuration panel                                        │ │
│  │  • Real-time plan visualization                                    │ │
│  │  • Export functionality                                            │ │
│  │  • Session-based rate limiting (5 requests)                        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTPS (REST + WebSocket)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND (Railway)                         │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   Sessions   │  │    Agent     │  │   Export     │  │  Rate       │ │
│  │   Manager    │  │  Orchestrator│  │   Generator  │  │  Limiter    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
│                           │                                              │
│                           ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     ONSETLAB SDK                                    │ │
│  │  • Router (DIRECT/REWOO/REACT selection)                           │ │
│  │  • Planner, Executor, Verifier, Solver                             │ │
│  │  • Built-in Tools                                                  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTPS
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TOGETHER.AI API                                  │
│                                                                          │
│  • Hosted small language models                                          │
│  • phi3.5, qwen2.5:3b, llama3.2:3b equivalents                          │
│  • ~$0.0002 per request                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Frontend (React + Vite + Tailwind)

**Location:** `web/frontend/`

**Features:**
- Chat interface with message history
- Tool toggle panel (enable/disable tools)
- Model selector dropdown
- Real-time plan visualization (shows REWOO steps)
- GitHub token input (optional MCP)
- Export dropdown (config/docker/script)
- Request counter (5/5 remaining)
- "Get the SDK" CTA after limit

**Key Components:**
```
web/frontend/
├── src/
│   ├── components/
│   │   ├── Chat.tsx           # Main chat interface
│   │   ├── Message.tsx        # Single message bubble
│   │   ├── PlanViewer.tsx     # REWOO plan visualization
│   │   ├── ToolPanel.tsx      # Tool toggles
│   │   ├── ModelSelector.tsx  # Model dropdown
│   │   ├── ExportModal.tsx    # Export options
│   │   └── LimitBanner.tsx    # Rate limit warning
│   ├── hooks/
│   │   ├── useChat.ts         # Chat state management
│   │   └── useSession.ts      # Session/rate limiting
│   ├── api/
│   │   └── client.ts          # API calls to backend
│   ├── App.tsx
│   └── main.tsx
├── package.json
└── vite.config.ts
```

**State Management:**
```typescript
interface AppState {
  messages: Message[];
  tools: { [name: string]: boolean };
  model: string;
  requestsRemaining: number;
  sessionId: string;
  isLoading: boolean;
  currentPlan: PlanStep[] | null;
}
```

---

### 2. Backend (FastAPI + OnsetLab SDK)

**Location:** `web/backend/`

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Send message, get response |
| GET | `/api/session` | Get/create session |
| GET | `/api/models` | List available models |
| GET | `/api/tools` | List available tools |
| POST | `/api/export` | Generate export package |
| GET | `/api/health` | Health check |

**Key Files:**
```
web/backend/
├── app/
│   ├── main.py              # FastAPI app
│   ├── routes/
│   │   ├── chat.py          # Chat endpoint
│   │   ├── session.py       # Session management
│   │   └── export.py        # Export generation
│   ├── services/
│   │   ├── agent_service.py # OnsetLab integration
│   │   ├── model_service.py # together.ai integration
│   │   └── session_store.py # In-memory session store
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   └── config.py            # Environment config
├── requirements.txt
└── Dockerfile
```

**Session Management:**
```python
# In-memory store (simple, no database needed)
sessions: Dict[str, Session] = {}

@dataclass
class Session:
    id: str
    requests_used: int = 0
    requests_limit: int = 5
    created_at: datetime
    tools: List[str]
    model: str
    messages: List[dict]
```

---

### 3. Model Service (together.ai Integration)

**Why together.ai:**
- Cheapest for small models (~$0.10-0.20 per 1M tokens)
- Compatible API format
- No GPU management needed

**Integration:**
```python
# Drop-in replacement for OllamaModel
class TogetherModel:
    def __init__(self, model: str):
        self.model = self._map_model(model)
        self.api_key = os.environ["TOGETHER_API_KEY"]
    
    def _map_model(self, name: str) -> str:
        """Map our model names to together.ai models."""
        mapping = {
            "phi3.5": "microsoft/phi-3.5-mini-instruct",
            "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
            "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
        }
        return mapping.get(name, mapping["phi3.5"])
    
    def generate(self, prompt: str, **kwargs) -> str:
        response = requests.post(
            "https://api.together.xyz/v1/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.1),
            }
        )
        return response.json()["choices"][0]["text"]
```

---

### 4. API Schemas

**Chat Request:**
```typescript
interface ChatRequest {
  message: string;
  session_id: string;
  tools: string[];      // ["Calculator", "DateTime", ...]
  model: string;        // "phi3.5"
  github_token?: string; // Optional MCP
}
```

**Chat Response:**
```typescript
interface ChatResponse {
  answer: string;
  plan: PlanStep[];
  results: { [step: string]: string };
  strategy: "direct" | "rewoo" | "react";
  slm_calls: number;
  requests_remaining: number;
}

interface PlanStep {
  id: string;           // "#E1"
  tool: string;         // "Calculator"
  params: object;       // {"expression": "15*0.15"}
  result?: string;      // "2.25"
  status: "pending" | "running" | "done" | "error";
}
```

**Export Request:**
```typescript
interface ExportRequest {
  session_id: string;
  format: "config" | "docker" | "binary";
}
```

---

## Data Flow

### Chat Flow

```
1. User types message
       │
       ▼
2. Frontend sends POST /api/chat
   {message, session_id, tools, model}
       │
       ▼
3. Backend checks rate limit
   - If exceeded → return 429 + CTA
   - If OK → continue
       │
       ▼
4. Backend creates Agent with selected tools
       │
       ▼
5. Agent.run(message)
   - Router selects strategy
   - Planner creates plan (if REWOO)
   - Executor runs tools
   - Solver generates answer
       │
       ▼
6. Backend returns response
   {answer, plan, results, strategy, requests_remaining}
       │
       ▼
7. Frontend displays:
   - Plan steps (animated)
   - Tool results
   - Final answer
   - Updated request counter
```

### Export Flow

```
1. User clicks "Export" → selects format
       │
       ▼
2. Frontend sends POST /api/export
   {session_id, format}
       │
       ▼
3. Backend reconstructs agent config from session
   - Model
   - Tools
   - Settings
       │
       ▼
4. Backend uses OnsetLab packaging module
   - ConfigExporter / DockerExporter / BinaryExporter
       │
       ▼
5. Returns downloadable file(s)
   - config: YAML file
   - docker: ZIP with Dockerfile, etc.
   - binary: Python script
```

---

## Rate Limiting

**Strategy:** Session-based, no authentication required

```python
RATE_LIMIT = 5  # requests per session

@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    session_id = request.cookies.get("session_id")
    
    if not session_id:
        session_id = create_session()
    
    session = sessions.get(session_id)
    
    if session.requests_used >= RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": "You've used all 5 free requests!",
                "cta": {
                    "text": "Run unlimited locally",
                    "pip": "pip install onsetlab",
                    "github": "https://github.com/riyanshibohra/onsetlab"
                }
            }
        )
    
    response = await call_next(request)
    
    if request.url.path == "/api/chat":
        session.requests_used += 1
    
    return response
```

---

## Deployment

### Frontend (Vercel)

```bash
# vercel.json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite"
}
```

**Environment Variables:**
- `VITE_API_URL` = Railway backend URL

### Backend (Railway)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Environment Variables:**
- `TOGETHER_API_KEY` = together.ai API key
- `CORS_ORIGINS` = Vercel frontend URL

---

## Cost Breakdown

| Component | Provider | Plan | Est. Cost |
|-----------|----------|------|-----------|
| Frontend | Vercel | Free | $0 |
| Backend | Railway | Hobby | $5/mo |
| Model API | together.ai | Pay-as-you-go | $5-15/mo* |
| Domain | Optional | - | $0-12/yr |
| **Total** | | | **$10-25/mo** |

*Assuming ~5000-15000 requests/month

---

## Security Considerations

1. **No persistent storage** - Sessions expire, no database
2. **GitHub tokens** - Only stored in session memory, not logged
3. **CORS** - Restricted to frontend domain only
4. **Rate limiting** - Prevents abuse
5. **Input validation** - Sanitize all user input

---

## Build Order

### Phase 1: Backend Core
1. FastAPI skeleton with health endpoint
2. Session management (in-memory)
3. together.ai integration (model service)
4. Agent service (OnsetLab SDK wrapper)
5. Chat endpoint
6. Rate limiting middleware

### Phase 2: Frontend Core
1. React + Vite + Tailwind setup
2. Chat component (messages, input)
3. API client (fetch wrapper)
4. Session hook (cookie management)
5. Basic styling

### Phase 3: Features
1. Tool panel (toggles)
2. Model selector
3. Plan visualization
4. Export modal
5. Rate limit banner

### Phase 4: Deploy
1. Backend → Railway
2. Frontend → Vercel
3. Environment variables
4. Test end-to-end

---

## File Structure

```
web/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py
│   │   │   ├── session.py
│   │   │   └── export.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── agent_service.py
│   │   │   ├── model_service.py
│   │   │   └── session_store.py
│   │   └── models/
│   │       ├── __init__.py
│   │       └── schemas.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── railway.json
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat.tsx
│   │   │   ├── Message.tsx
│   │   │   ├── PlanViewer.tsx
│   │   │   ├── ToolPanel.tsx
│   │   │   ├── ModelSelector.tsx
│   │   │   ├── ExportModal.tsx
│   │   │   └── LimitBanner.tsx
│   │   ├── hooks/
│   │   │   ├── useChat.ts
│   │   │   └── useSession.ts
│   │   ├── api/
│   │   │   └── client.ts
│   │   ├── types/
│   │   │   └── index.ts
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── vercel.json
│
└── README.md
```

---

## Next Steps

1. **Create backend skeleton** → `web/backend/`
2. **Implement together.ai model service**
3. **Wrap OnsetLab SDK for web use**
4. **Create frontend** → `web/frontend/`
5. **Connect and test locally**
6. **Deploy to Railway + Vercel**

Ready to start with Phase 1: Backend Core?
