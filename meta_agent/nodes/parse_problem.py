"""
Parse Problem Node
==================
Extracts required services from the problem statement using Claude.
"""

import json
import anthropic

from meta_agent.state import MetaAgentState


PARSE_PROBLEM_PROMPT = """You are an expert at analyzing problem statements for AI agent development.

Given a problem statement describing what an AI agent should do, extract the list of external services/APIs that the agent will need to interact with.

For each service, provide a normalized identifier (lowercase, underscores for spaces).

Common services include:
- google_calendar (Google Calendar)
- gmail (Gmail/Google Mail)
- slack (Slack messaging)
- github (GitHub repositories, issues, PRs)
- notion (Notion pages, databases)
- linear (Linear issues, projects)
- discord (Discord messaging)
- twitter / x (Twitter/X posts)
- spotify (Spotify music)
- google_drive (Google Drive files)
- google_sheets (Google Sheets)
- jira (Jira issues)
- trello (Trello boards)
- asana (Asana tasks)
- hubspot (HubSpot CRM)
- salesforce (Salesforce CRM)
- stripe (Stripe payments)
- twilio (Twilio SMS/calls)
- sendgrid (SendGrid email)
- aws_s3 (AWS S3 storage)
- postgres (PostgreSQL database)
- mongodb (MongoDB database)
- elasticsearch (Elasticsearch)
- openai (OpenAI API)
- anthropic (Anthropic Claude API)

If a service is mentioned but not in the list above, create a reasonable identifier for it.

Respond with a JSON object in this exact format:
{
    "services": ["service1", "service2"],
    "reasoning": "Brief explanation of why these services are needed"
}

Only include services that require external API access. Do not include general programming concepts or libraries."""


def parse_problem(state: MetaAgentState) -> dict:
    """
    Parse the problem statement to identify required services.
    
    Uses Claude to extract service names like:
    - "google_calendar"
    - "gmail"
    - "slack"
    - "github"
    
    Args:
        state: Current MetaAgentState
        
    Returns:
        State update with identified_services list
    """
    problem_statement = state["problem_statement"]
    api_key = state["anthropic_api_key"]
    
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": f"{PARSE_PROBLEM_PROMPT}\n\nProblem Statement:\n{problem_statement}"}
            ],
        )
        
        # Parse the response - Claude returns content as a list
        response_text = response.content[0].text
        
        # Extract JSON from response (Claude might wrap it in markdown)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text.strip())
        services = result.get("services", [])
        reasoning = result.get("reasoning", "")
        
        print(f"📋 Identified {len(services)} services: {', '.join(services)}")
        if reasoning:
            print(f"   Reasoning: {reasoning}")
        
        return {
            "identified_services": services,
            "current_service_index": 0,
        }
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse LLM response: {e}")
        return {
            "identified_services": [],
            "current_service_index": 0,
            "errors": [f"Failed to parse problem statement: {e}"],
        }
    except Exception as e:
        print(f"❌ Error calling Claude: {e}")
        return {
            "identified_services": [],
            "current_service_index": 0,
            "errors": [f"Claude error: {e}"],
        }
