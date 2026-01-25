"""
FastAPI Server
==============
REST API endpoints for the meta-agent.

Run with:
    cd OnsetLab && python3 meta_agent/api/server.py
    
Or:
    cd OnsetLab/meta_agent && python3 -m api.server
"""

import os
import sys
from pathlib import Path

# Add parent directories to path for imports
_this_file = Path(__file__).resolve()
_meta_agent_dir = _this_file.parent.parent  # meta_agent/
_onsetlab_dir = _meta_agent_dir.parent       # OnsetLab/

if str(_meta_agent_dir) not in sys.path:
    sys.path.insert(0, str(_meta_agent_dir))
if str(_onsetlab_dir) not in sys.path:
    sys.path.insert(0, str(_onsetlab_dir))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import traceback

# Load .env file if it exists
from dotenv import load_dotenv
env_path = _meta_agent_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from {env_path}")

from meta_agent.graph import run_meta_agent
from meta_agent.utils.gist_upload import upload_to_gist_async


# =============================================================================
# Request/Response Models
# =============================================================================

class GenerateRequest(BaseModel):
    """Request body for /api/generate-agent endpoint."""
    problem_statement: str = Field(
        ...,
        description="Description of what the agent should do",
        example="I need an agent that checks my Google Calendar and sends Slack reminders"
    )
    anthropic_api_key: str = Field(
        ...,
        description="Anthropic API key for Claude LLM calls"
    )
    tavily_api_key: str = Field(
        ...,
        description="Tavily API key for web search"
    )
    github_token: Optional[str] = Field(
        default=None,
        description="GitHub token for uploading notebook to Gist (optional)"
    )
    upload_to_gist: bool = Field(
        default=False,
        description="Whether to upload the notebook to GitHub Gist"
    )


class MCPServerResponse(BaseModel):
    """MCP server in response."""
    service: str
    package: str
    auth_type: str
    env_var: Optional[str] = None
    tools: list[dict] = []
    setup_url: Optional[str] = None
    confidence: float = 0.0


class APIEndpointResponse(BaseModel):
    """Detailed API endpoint in response."""
    name: str
    method: str
    path: str
    description: str = ""
    parameters: dict = {}
    required_params: list[str] = []
    request_body: Optional[dict] = None
    response_schema: Optional[dict] = None


class APIServerResponse(BaseModel):
    """API server fallback in response with detailed endpoint info."""
    service: str
    reason: str
    api_docs_url: Optional[str] = None
    base_url: str = ""
    auth_type: str = "bearer"
    auth_header: Optional[str] = None
    env_var: str = ""
    endpoints: list[APIEndpointResponse] = []


class TokenGuideResponse(BaseModel):
    """Token setup guide in response."""
    service: str
    auth_type: str
    steps: list[str]
    env_var: str


class GenerateResponse(BaseModel):
    """Response body for /api/generate-agent endpoint."""
    success: bool = True
    colab_notebook: str = Field(
        default="",
        description="The generated Colab notebook as JSON string"
    )
    colab_notebook_url: Optional[str] = Field(
        default=None,
        description="URL to open the notebook in Colab (if uploaded to Gist)"
    )
    gist_url: Optional[str] = Field(
        default=None,
        description="URL to the GitHub Gist (if uploaded)"
    )
    mcp_servers: list[MCPServerResponse] = Field(
        default=[],
        description="List of discovered MCP servers"
    )
    api_servers: list[APIServerResponse] = Field(
        default=[],
        description="List of services needing API implementation"
    )
    token_guides: list[TokenGuideResponse] = Field(
        default=[],
        description="Setup instructions for each service"
    )
    tool_count: int = Field(
        default=0,
        description="Total number of tools discovered"
    )
    errors: list[str] = Field(
        default=[],
        description="Any errors encountered during processing"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "0.1.0"


# =============================================================================
# FastAPI App
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="OnsetLab Meta-Agent API",
        description="""
        Discover MCP servers and generate Colab notebooks for building AI agents.
        
        The meta-agent:
        1. Parses your problem statement to identify required services
        2. Searches for MCP servers for each service
        3. Falls back to API implementation if no good MCP exists
        4. Generates a complete Colab notebook
        
        The generated notebook uses the OnsetLab SDK to:
        - Generate synthetic training data
        - Fine-tune a small language model
        - Package the agent for local deployment
        """,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware for frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict to your frontend domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse()
    
    @app.get("/", tags=["Health"])
    async def root():
        """Root endpoint with API info."""
        return {
            "name": "OnsetLab Meta-Agent API",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
        }
    
    # =========================================================================
    # Generate Agent
    # =========================================================================
    
    @app.post(
        "/api/generate-agent",
        response_model=GenerateResponse,
        responses={
            200: {"description": "Successfully generated notebook"},
            400: {"model": ErrorResponse, "description": "Invalid request"},
            500: {"model": ErrorResponse, "description": "Internal server error"},
        },
        tags=["Agent Generation"],
    )
    async def generate_agent(request: GenerateRequest):
        """
        Generate a Colab notebook for building an AI agent.
        
        This endpoint:
        1. Parses the problem statement to identify required services
        2. Searches for MCP servers for each service
        3. Falls back to API implementation if no good MCP exists
        4. Generates a Colab notebook with all discovered tools
        5. Optionally uploads the notebook to GitHub Gist
        
        The notebook can be opened in Google Colab to:
        - Configure authentication tokens
        - Generate training data
        - Fine-tune a model
        - Download the packaged agent
        
        **Required API Keys:**
        - Anthropic API key: For Claude LLM calls (MCP discovery)
        - Tavily API key: For web search to find MCP servers
        
        **Optional:**
        - GitHub token: For uploading notebook to Gist (get shareable Colab URL)
        
        **Example Problem Statements:**
        - "I need an agent that manages my Google Calendar and sends Slack reminders"
        - "Build an agent that can create GitHub issues and track Linear tasks"
        - "Create an assistant that searches my Notion workspace and sends emails"
        """
        try:
            # Validate inputs
            if not request.problem_statement.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Problem statement cannot be empty"
                )
            
            if not request.anthropic_api_key.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Anthropic API key is required"
                )
            
            if not request.tavily_api_key.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Tavily API key is required"
                )
            
            if request.upload_to_gist and not request.github_token:
                raise HTTPException(
                    status_code=400,
                    detail="GitHub token is required for Gist upload"
                )
            
            # Run the meta-agent
            result = await run_meta_agent(
                problem_statement=request.problem_statement,
                anthropic_api_key=request.anthropic_api_key,
                tavily_api_key=request.tavily_api_key,
            )
            
            # Upload to Gist if requested
            colab_notebook_url = None
            gist_url = None
            errors = result.get("errors", [])
            
            if request.upload_to_gist and result.get("colab_notebook"):
                try:
                    gist_result = await upload_to_gist_async(
                        notebook_json=result["colab_notebook"],
                        github_token=request.github_token,
                        description=f"OnsetLab Agent: {request.problem_statement[:50]}...",
                    )
                    colab_notebook_url = gist_result["colab_url"]
                    gist_url = gist_result["gist_url"]
                    print(f"📤 Uploaded to Gist: {gist_url}")
                    print(f"🔗 Colab URL: {colab_notebook_url}")
                except Exception as e:
                    errors.append(f"Failed to upload to Gist: {str(e)}")
                    print(f"❌ Gist upload failed: {e}")
            
            # Build response
            return GenerateResponse(
                success=True,
                colab_notebook=result.get("colab_notebook", ""),
                colab_notebook_url=colab_notebook_url,
                gist_url=gist_url,
                mcp_servers=[
                    MCPServerResponse(**server) 
                    for server in result.get("mcp_servers", [])
                ],
                api_servers=[
                    APIServerResponse(
                        service=api.get("service", ""),
                        reason=api.get("reason", ""),
                        api_docs_url=api.get("api_docs_url"),
                        base_url=api.get("base_url", ""),
                        auth_type=api.get("auth_type", "bearer"),
                        auth_header=api.get("auth_header"),
                        env_var=api.get("env_var", ""),
                        endpoints=[
                            APIEndpointResponse(**ep) 
                            for ep in api.get("endpoints", [])
                        ]
                    )
                    for api in result.get("api_servers", [])
                ],
                token_guides=[
                    TokenGuideResponse(**guide)
                    for guide in result.get("token_guides", [])
                ],
                tool_count=len(result.get("tool_schemas", [])),
                errors=errors,
            )
            
        except HTTPException:
            raise
        except Exception as e:
            # Log the full traceback
            traceback.print_exc()
            
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate agent: {str(e)}"
            )
    
    return app


# Create default app instance
app = create_app()


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting OnsetLab Meta-Agent API...")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🏥 Health: http://localhost:8000/health")
    
    uvicorn.run(
        "meta_agent.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
