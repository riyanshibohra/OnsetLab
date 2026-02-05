# OnsetLab Packaging

Export your agents in multiple formats for deployment.

## Export Formats

| Format | Output | Use Case |
|--------|--------|----------|
| `config` | YAML/JSON file | Share config, version control, recreate agents |
| `docker` | Complete Docker setup | Production deployment, containers |
| `binary` | Standalone Python script | Simple distribution, single-file deployment |

## Quick Start

### Using the SDK

```python
from onsetlab import Agent
from onsetlab.tools import Calculator, DateTime

# Create agent
agent = Agent(
    model="phi3.5",
    tools=[Calculator(), DateTime()],
)

# Export as config
agent.export("config", "my_agent.yaml")

# Export as Docker
agent.export("docker", "./docker_agent/")

# Export as standalone script
agent.export("binary", "my_agent.py")
```

### Using the CLI

```bash
# Export config
python -m onsetlab export --format config --output my_agent.yaml

# Export Docker setup
python -m onsetlab export --format docker --output ./docker_agent/

# Export Docker with bundled Ollama
python -m onsetlab export --format docker --output ./docker_agent/ --include-ollama

# Export standalone script
python -m onsetlab export --format binary --output my_agent.py
```

---

## Config Export

Exports agent configuration as YAML or JSON.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `format` | auto-detect | `"yaml"` or `"json"` |
| `include_mcp_auth` | `False` | Include MCP auth tokens (security risk!) |

### Example Output

```yaml
version: '1.0'
onsetlab:
  model: phi3.5
  settings:
    memory: true
    verify: true
    routing: true
    react_fallback: true
    max_replans: 1
tools:
  - name: Calculator
    class: Calculator
  - name: DateTime
    class: DateTime
mcp_servers: []
```

### Loading Config

```python
from onsetlab.packaging import ConfigExporter

config = ConfigExporter.load("my_agent.yaml")
# Use config to recreate agent
```

---

## Docker Export

Creates a complete Docker setup with:
- `Dockerfile` - Container definition
- `docker-compose.yml` - Multi-service orchestration
- `agent_config.yaml` - Agent configuration
- `entrypoint.py` - Agent runner script
- `requirements.txt` - Python dependencies
- `README.md` - Instructions

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `include_ollama` | `False` | Bundle Ollama in the container |
| `api_mode` | `True` | Configure for API server mode |

### Basic Docker (External Ollama)

```bash
agent.export("docker", "./my_agent/")
cd my_agent
docker-compose up --build
```

This creates two services:
1. **ollama** - Runs Ollama server
2. **agent** - Your OnsetLab agent (connects to Ollama)

### Docker with Bundled Ollama

```bash
agent.export("docker", "./my_agent/", include_ollama=True)
cd my_agent
docker-compose up --build
```

Single container with both Ollama and your agent.

### API Usage

Once running, the agent exposes a REST API:

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 15 + 27?"}'
```

Response:
```json
{
  "answer": "42",
  "strategy": "rewoo",
  "slm_calls": 2
}
```

---

## Binary/Script Export

Creates a standalone Python script with embedded configuration.

### Requirements

The exported script requires:
- Python 3.8+
- OnsetLab installed (`pip install onsetlab`)
- Ollama running locally

### Usage

```bash
# Interactive mode
python my_agent.py

# Single query
python my_agent.py "What is the current time?"

# API server mode
python my_agent.py --serve --port 8000
```

### PyInstaller (Optional)

For actual binary distribution:

```python
agent.export("binary", "my_agent", use_pyinstaller=True)
```

Requires: `pip install pyinstaller`

---

## Deployment Guide

### Local Development

```bash
# Export and run
python -m onsetlab export --format binary --output agent.py
python agent.py
```

### Docker Deployment

```bash
# Export
python -m onsetlab export --format docker --output ./deployment/

# Build and run
cd deployment
docker-compose up -d

# Test
curl http://localhost:8000/health
```

### Production with Kubernetes

```bash
# Export Docker setup
agent.export("docker", "./k8s-agent/", include_ollama=True)

# Build and push
docker build -t myregistry/onsetlab-agent:v1 ./k8s-agent/
docker push myregistry/onsetlab-agent:v1

# Deploy
kubectl apply -f k8s-deployment.yaml
```

### Cloud Deployment

The Docker export works with:
- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Apps
- DigitalOcean App Platform
- Railway, Fly.io, Render

---

## Tips

1. **Config versioning**: Export config to YAML and commit to git for reproducibility
2. **Docker caching**: The model is pulled at runtime; pre-pull for faster cold starts
3. **API mode**: Use for integrations; supports any HTTP client
4. **Bundled Ollama**: Larger image (~2GB) but fully self-contained
5. **External Ollama**: Smaller agent container, share Ollama across agents
