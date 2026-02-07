"""Chat endpoint."""

import asyncio
import concurrent.futures
from fastapi import APIRouter, HTTPException, Request, Response
from typing import Optional

from ..models.schemas import ChatRequest, ChatResponse, PlanStep, PipelineTrace, RateLimitError
from ..services.session_store import session_store
from ..services.agent_service import create_agent
from ..services.mcp_manager import mcp_manager

# Thread pool for running agent (MCP tool calls need their own event loop)
_agent_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="agent"
)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    response: Response,
    body: ChatRequest,
):
    """
    Process a chat message and return agent response.
    
    Rate limited to 5 requests per session.
    """
    # Get or create session
    session_id = request.cookies.get("session_id")
    session = session_store.get_or_create(session_id)
    
    # Set session cookie if new
    if session_id != session.id:
        response.set_cookie(
            key="session_id",
            value=session.id,
            httponly=True,
            samesite="lax",
            max_age=86400,  # 24 hours
        )
    
    # Check rate limit
    if session.is_rate_limited:
        raise HTTPException(
            status_code=429,
            detail=RateLimitError().model_dump(),
        )
    
    # Update session config
    session_store.update_config(
        session.id,
        model=body.model,
        tools=body.tools if body.tools else None,
    )
    
    # Get tools to use
    tools = body.tools if body.tools else session.tools
    
    try:
        # Get MCP tools from session (if any MCP servers are connected)
        mcp_tools = mcp_manager.get_tools(session.id)

        # Build conversation context from recent messages
        # (so the agent understands "the same repo", "do it again", etc.)
        context = None
        if session.messages:
            recent = session.messages[-6:]  # last 3 exchanges
            context_parts = []
            for msg in recent:
                role = msg["role"].upper()
                context_parts.append(f"{role}: {msg['content']}")
            context = "\n".join(context_parts)

        # Create agent and run
        agent = create_agent(
            model_name=body.model,
            tool_names=tools,
            context=context,
            mcp_tools=mcp_tools if mcp_tools else None,
        )
        
        # Run agent in a thread to avoid event-loop conflicts
        # (MCP tool calls use SyncMCPClient which needs its own event loop)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_agent_executor, agent.run, body.message)
        
        # Increment request count (only on success)
        session_store.increment_requests(session.id)
        
        # Store message in session
        session_store.add_message(session.id, "user", body.message)
        session_store.add_message(session.id, "assistant", result.answer)
        
        # Convert plan steps (safely handle malformed params)
        plan_steps = []
        for step in result.plan:
            params = step.get("params", {})
            if not isinstance(params, dict):
                params = {"_raw": str(params)}
            plan_steps.append(
                PlanStep(
                    id=step.get("id", ""),
                    tool=step.get("tool", ""),
                    params=params,
                    result=step.get("result"),
                    status=step.get("status", "done"),
                )
            )
        
        # Get updated session for remaining count
        session = session_store.get(session.id)
        
        # Build pipeline trace for UI
        trace_data = None
        if result.trace:
            t = result.trace
            trace_data = PipelineTrace(
                router_decision=t.router_decision,
                router_reason=t.router_reason,
                tools_total=t.tools_total,
                tools_filtered=t.tools_filtered,
                tools_selected=t.tools_selected or [],
                tool_rules=t.tool_rules,
                planner_think=t.planner_think,
                planner_prompt=t.planner_prompt,
                fallback_used=t.fallback_used,
                fallback_reason=t.fallback_reason,
            )

        return ChatResponse(
            answer=result.answer,
            plan=plan_steps,
            results=result.results,
            strategy=result.strategy,
            slm_calls=result.slm_calls,
            requests_remaining=session.requests_remaining if session else 0,
            trace=trace_data,
        )
        
    except ValueError as e:
        import logging as _log
        _log.getLogger(__name__).error(f"ValueError in chat: {e}")
        print(f"CHAT ValueError: {e}")  # visible in terminal
        # Missing API key
        if "OPENROUTER_API_KEY" in str(e):
            raise HTTPException(
                status_code=503,
                detail={"error": "config_error", "message": "Server not configured. Please set OPENROUTER_API_KEY."},
            )
        raise HTTPException(
            status_code=400,
            detail={"error": "validation_error", "message": str(e)},
        )
    except RuntimeError as e:
        import logging as _log
        _log.getLogger(__name__).error(f"RuntimeError in chat: {e}")
        print(f"CHAT RuntimeError: {e}")
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            raise HTTPException(
                status_code=503,
                detail={"error": "api_error", "message": "Invalid API key. Please check OPENROUTER_API_KEY."},
            )
        if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
            raise HTTPException(
                status_code=504,
                detail={"error": "timeout", "message": "The model took too long to respond. Try again or use a smaller query."},
            )
        raise HTTPException(
            status_code=500,
            detail={"error": "api_error", "message": f"Model API error: {error_msg}"},
        )
    except Exception as e:
        import traceback
        import logging
        logger = logging.getLogger(__name__)
        error_details = traceback.format_exc()
        logger.error(f"Agent error: {error_details}")
        print(f"AGENT ERROR: {error_details}")  # Also print to stdout
        raise HTTPException(
            status_code=500,
            detail={"error": "agent_error", "message": str(e), "trace": error_details[:500]},
        )
