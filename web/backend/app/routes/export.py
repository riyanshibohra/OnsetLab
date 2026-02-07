"""Export endpoint."""

import tempfile
import zipfile
import io
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..models.schemas import ExportRequest
from ..services.session_store import session_store

router = APIRouter(prefix="/api", tags=["export"])


# Config template
CONFIG_TEMPLATE = """version: '1.0'
onsetlab:
  model: {model}
  settings:
    memory: true
    verify: true
    routing: true
    react_fallback: true
tools:
{tools}
mcp_servers: []
"""


@router.post("/export")
async def export_agent(request: Request, body: ExportRequest):
    """
    Export agent configuration in specified format.
    
    Returns downloadable file.
    """
    session_id = request.cookies.get("session_id")
    session = session_store.get(session_id) if session_id else None
    
    # Use session config or request config
    model = body.model
    tools = body.tools if body.tools else (session.tools if session else ["Calculator", "DateTime"])
    
    if body.format == "config":
        return _export_config(model, tools)
    elif body.format == "docker":
        return _export_docker(model, tools)
    elif body.format == "docker-vllm":
        return _export_docker_vllm(model, tools)
    elif body.format == "binary":
        return _export_binary(model, tools)
    else:
        raise HTTPException(400, f"Unknown format: {body.format}")


def _export_config(model: str, tools: list) -> StreamingResponse:
    """Export as YAML config."""
    tools_yaml = "\n".join(f"  - name: {t}\n    class: {t}" for t in tools)
    
    content = CONFIG_TEMPLATE.format(
        model=model,
        tools=tools_yaml,
    )
    
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/x-yaml",
        headers={"Content-Disposition": "attachment; filename=onsetlab_agent.yaml"},
    )


def _export_docker(model: str, tools: list) -> StreamingResponse:
    """Export as Docker ZIP package."""
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # agent_config.yaml
        tools_yaml = "\n".join(f"  - name: {t}\n    class: {t}" for t in tools)
        config_content = CONFIG_TEMPLATE.format(model=model, tools=tools_yaml)
        zf.writestr("agent_config.yaml", config_content)
        
        # Dockerfile
        dockerfile = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
ENV OLLAMA_HOST=http://host.docker.internal:11434
CMD ["python", "entrypoint.py", "--serve"]
"""
        zf.writestr("Dockerfile", dockerfile)
        
        # requirements.txt
        requirements = """onsetlab
pyyaml
fastapi
uvicorn
"""
        zf.writestr("requirements.txt", requirements)
        
        # docker-compose.yml
        compose = f"""version: '3.8'
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
  agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
    depends_on:
      - ollama
volumes:
  ollama_data:
"""
        zf.writestr("docker-compose.yml", compose)
        
        # entrypoint.py (simplified)
        entrypoint = f'''#!/usr/bin/env python3
import argparse
import yaml

def main():
    from onsetlab import Agent
    from onsetlab.tools import {", ".join(tools)}
    
    agent = Agent(
        model="{model}",
        tools=[{", ".join(f"{t}()" for t in tools)}],
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    if args.serve:
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn
        
        app = FastAPI(title="OnsetLab Agent")
        
        class Query(BaseModel):
            message: str
        
        @app.post("/chat")
        def chat(q: Query):
            result = agent.run(q.message)
            return {{"answer": result.answer}}
        
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        while True:
            q = input("You: ")
            if q.lower() in ["quit", "exit"]:
                break
            result = agent.run(q)
            print(f"Agent: {{result.answer}}")

if __name__ == "__main__":
    main()
'''
        zf.writestr("entrypoint.py", entrypoint)
        
        # README.md
        readme = """# OnsetLab Agent (Docker)

## Quick Start

```bash
docker-compose up --build
```

## API

```bash
curl -X POST http://localhost:8000/chat \\
  -H "Content-Type: application/json" \\
  -d '{"message": "What is 15 + 27?"}'
```
"""
        zf.writestr("README.md", readme)
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=onsetlab_docker.zip"},
    )


def _export_docker_vllm(model: str, tools: list) -> StreamingResponse:
    """Export as Docker ZIP with vLLM engine (GPU-accelerated)."""
    HF_MODEL_MAP = {
        "phi3.5": "microsoft/Phi-3.5-mini-instruct",
        "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen3-a3b": "Qwen/Qwen3-A3B",
        "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "gemma3:4b": "google/gemma-3-4b-it",
    }
    hf_model = HF_MODEL_MAP.get(model, model)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # agent_config.yaml
        tools_yaml = "\n".join(f"  - name: {t}\n    class: {t}" for t in tools)
        config_content = CONFIG_TEMPLATE.format(model=model, tools=tools_yaml)
        zf.writestr("agent_config.yaml", config_content)

        # Dockerfile (CUDA + vLLM)
        zf.writestr("Dockerfile", f"""FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip python3-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8100
ENV MODEL_NAME={hf_model}
ENV VLLM_PORT=8100
COPY start.sh .
RUN chmod +x start.sh
ENTRYPOINT ["./start.sh"]
""")

        # requirements.txt
        zf.writestr("requirements.txt", """onsetlab
pyyaml
fastapi
uvicorn
vllm>=0.6.0
openai
""")

        # start.sh
        zf.writestr("start.sh", f"""#!/bin/bash
echo "Starting vLLM for $MODEL_NAME ..."
python3 -m vllm.entrypoints.openai.api_server \\
    --model "$MODEL_NAME" --port ${{VLLM_PORT:-8100}} \\
    --max-model-len 4096 --gpu-memory-utilization 0.9 &
for i in $(seq 1 60); do
    curl -s http://localhost:${{VLLM_PORT:-8100}}/health > /dev/null 2>&1 && break
    sleep 2
done
echo "vLLM ready. Starting agent..."
python3 entrypoint.py --serve
""")

        # entrypoint.py
        zf.writestr("entrypoint.py", f'''#!/usr/bin/env python3
import argparse, os, yaml
from openai import OpenAI

class VLLMModel:
    def __init__(self, model, base_url="http://localhost:8100/v1"):
        self.model_name = model
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
    def generate(self, prompt, max_tokens=512, temperature=0.0):
        r = self.client.completions.create(model=self.model_name, prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        return r.choices[0].text.strip()

def main():
    from onsetlab import Agent
    from onsetlab.tools import {", ".join(tools)}
    agent = Agent(model=VLLMModel("{hf_model}"), tools=[{", ".join(f"{t}()" for t in tools)}])
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.serve:
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn
        app = FastAPI(title="OnsetLab Agent (vLLM)")
        class Query(BaseModel):
            message: str
        @app.post("/chat")
        def chat(q: Query):
            result = agent.run(q.message)
            return {{"answer": result.answer}}
        @app.get("/health")
        def health():
            return {{"status": "ok", "engine": "vllm"}}
        uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
''')

        # docker-compose.yml
        zf.writestr("docker-compose.yml", f"""version: '3.8'
services:
  agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME={hf_model}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
""")

        # README
        zf.writestr("README.md", f"""# OnsetLab Agent (vLLM Engine)

GPU-accelerated inference — 5-10x faster than Ollama.

**Model:** `{hf_model}`

## Requirements
- NVIDIA GPU with CUDA 12.1+
- Docker with NVIDIA Container Toolkit

## Quick Start
```bash
docker-compose up --build
```

## API
```bash
curl -X POST http://localhost:8000/chat \\
  -H "Content-Type: application/json" \\
  -d '{{"message": "What is 15 + 27?"}}'
```
""")

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=onsetlab_vllm_docker.zip"},
    )


def _export_binary(model: str, tools: list) -> StreamingResponse:
    """Export as standalone Python script."""
    script = f'''#!/usr/bin/env python3
"""OnsetLab Agent - Generated from Playground"""

CONFIG = {{
    "model": "{model}",
    "tools": {tools},
}}

def main():
    try:
        from onsetlab import Agent
        from onsetlab.tools import {", ".join(tools)}
    except ImportError:
        print("Install OnsetLab: pip install onsetlab")
        return
    
    agent = Agent(
        model=CONFIG["model"],
        tools=[{", ".join(f"{t}()" for t in tools)}],
    )
    
    import sys
    if len(sys.argv) > 1:
        result = agent.run(" ".join(sys.argv[1:]))
        print(result.answer)
    else:
        print(f"OnsetLab Agent ({{agent.model_name}})")
        print("Type 'quit' to exit\\n")
        while True:
            try:
                q = input("You: ").strip()
                if q.lower() in ["quit", "exit", "q"]:
                    break
                if not q:
                    continue
                result = agent.run(q)
                print(f"Agent: {{result.answer}}\\n")
            except (KeyboardInterrupt, EOFError):
                break
        print("Goodbye!")

if __name__ == "__main__":
    main()
'''
    
    return StreamingResponse(
        io.BytesIO(script.encode()),
        media_type="text/x-python",
        headers={"Content-Disposition": "attachment; filename=onsetlab_agent.py"},
    )
