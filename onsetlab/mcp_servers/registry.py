"""
MCP Server Tool Registry

Comprehensive tool schemas for 5 core MCP servers.
Each tool follows the JSON Schema format compatible with:
- OpenAI function calling
- Anthropic tool use  
- Fine-tuning data generation
"""

from typing import Dict, List, Any

# =============================================================================
# GOOGLE CALENDAR TOOLS (5 tools)
# =============================================================================

GOOGLE_CALENDAR_TOOLS = [
    {
        "name": "calendar_get_current_time",
        "description": "Get the current date and time. Use this to understand 'today', 'tomorrow', 'next week' etc.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "calendar_list_events",
        "description": "List calendar events within a time range. Returns event titles, times, and IDs.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format (optional, defaults to start_date)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of events to return (default: 10)"
                }
            },
            "required": ["start_date"]
        }
    },
    {
        "name": "calendar_create_event",
        "description": "Create a new calendar event with title, date, time, and optional details.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Event title/summary"
                },
                "start_time": {
                    "type": "string",
                    "description": "Start time in ISO format (YYYY-MM-DDTHH:MM:SS)"
                },
                "end_time": {
                    "type": "string",
                    "description": "End time in ISO format (YYYY-MM-DDTHH:MM:SS)"
                },
                "description": {
                    "type": "string",
                    "description": "Event description (optional)"
                },
                "location": {
                    "type": "string",
                    "description": "Event location (optional)"
                },
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of attendee email addresses (optional)"
                }
            },
            "required": ["title", "start_time", "end_time"]
        }
    },
    {
        "name": "calendar_delete_event",
        "description": "Delete a calendar event by its ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "The unique ID of the event to delete"
                }
            },
            "required": ["event_id"]
        }
    },
    {
        "name": "calendar_update_event",
        "description": "Update an existing calendar event. Only provided fields will be updated.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "The unique ID of the event to update"
                },
                "title": {
                    "type": "string",
                    "description": "New event title (optional)"
                },
                "start_time": {
                    "type": "string",
                    "description": "New start time in ISO format (optional)"
                },
                "end_time": {
                    "type": "string",
                    "description": "New end time in ISO format (optional)"
                },
                "description": {
                    "type": "string",
                    "description": "New description (optional)"
                },
                "location": {
                    "type": "string",
                    "description": "New location (optional)"
                }
            },
            "required": ["event_id"]
        }
    }
]

# =============================================================================
# GITHUB TOOLS (6 tools)
# =============================================================================

GITHUB_TOOLS = [
    {
        "name": "github_search_repos",
        "description": "Search for GitHub repositories by name, topic, or description.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'machine learning python')"
                },
                "sort": {
                    "type": "string",
                    "enum": ["stars", "forks", "updated", "best-match"],
                    "description": "Sort order (default: best-match)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 10)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "github_get_repo",
        "description": "Get detailed information about a specific repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner (username or org)"
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name"
                }
            },
            "required": ["owner", "repo"]
        }
    },
    {
        "name": "github_list_issues",
        "description": "List issues in a repository with optional filters.",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner"
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name"
                },
                "state": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "description": "Filter by state (default: open)"
                },
                "labels": {
                    "type": "string",
                    "description": "Comma-separated list of label names"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max issues to return (default: 10)"
                }
            },
            "required": ["owner", "repo"]
        }
    },
    {
        "name": "github_create_issue",
        "description": "Create a new issue in a repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner"
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name"
                },
                "title": {
                    "type": "string",
                    "description": "Issue title"
                },
                "body": {
                    "type": "string",
                    "description": "Issue description/body (optional)"
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels to assign (optional)"
                },
                "assignees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Usernames to assign (optional)"
                }
            },
            "required": ["owner", "repo", "title"]
        }
    },
    {
        "name": "github_list_pull_requests",
        "description": "List pull requests in a repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner"
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name"
                },
                "state": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "description": "Filter by state (default: open)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max PRs to return (default: 10)"
                }
            },
            "required": ["owner", "repo"]
        }
    },
    {
        "name": "github_create_pull_request",
        "description": "Create a new pull request.",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner"
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name"
                },
                "title": {
                    "type": "string",
                    "description": "PR title"
                },
                "head": {
                    "type": "string",
                    "description": "The branch with your changes"
                },
                "base": {
                    "type": "string",
                    "description": "The branch to merge into (e.g., 'main')"
                },
                "body": {
                    "type": "string",
                    "description": "PR description (optional)"
                }
            },
            "required": ["owner", "repo", "title", "head", "base"]
        }
    }
]

# =============================================================================
# SLACK TOOLS (5 tools)
# =============================================================================

SLACK_TOOLS = [
    {
        "name": "slack_send_message",
        "description": "Send a message to a Slack channel or user.",
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Channel name (e.g., '#general') or user ID"
                },
                "text": {
                    "type": "string",
                    "description": "Message content"
                },
                "thread_ts": {
                    "type": "string",
                    "description": "Thread timestamp to reply in a thread (optional)"
                }
            },
            "required": ["channel", "text"]
        }
    },
    {
        "name": "slack_list_channels",
        "description": "List available Slack channels.",
        "parameters": {
            "type": "object",
            "properties": {
                "types": {
                    "type": "string",
                    "enum": ["public", "private", "all"],
                    "description": "Type of channels to list (default: public)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max channels to return (default: 20)"
                }
            },
            "required": []
        }
    },
    {
        "name": "slack_search_messages",
        "description": "Search for messages across Slack channels.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (supports Slack search syntax)"
                },
                "channel": {
                    "type": "string",
                    "description": "Limit search to specific channel (optional)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max messages to return (default: 10)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "slack_get_channel_history",
        "description": "Get recent messages from a channel.",
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Channel name or ID"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of messages to retrieve (default: 20)"
                }
            },
            "required": ["channel"]
        }
    },
    {
        "name": "slack_add_reaction",
        "description": "Add an emoji reaction to a message.",
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Channel where the message is"
                },
                "timestamp": {
                    "type": "string",
                    "description": "Message timestamp"
                },
                "emoji": {
                    "type": "string",
                    "description": "Emoji name without colons (e.g., 'thumbsup')"
                }
            },
            "required": ["channel", "timestamp", "emoji"]
        }
    }
]

# =============================================================================
# GMAIL TOOLS (5 tools)
# =============================================================================

GMAIL_TOOLS = [
    {
        "name": "gmail_search_emails",
        "description": "Search emails using Gmail search syntax.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'from:boss', 'is:unread', 'subject:meeting')"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max emails to return (default: 10)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "gmail_send_email",
        "description": "Compose and send an email.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line"
                },
                "body": {
                    "type": "string",
                    "description": "Email body content"
                },
                "cc": {
                    "type": "string",
                    "description": "CC recipients (comma-separated, optional)"
                },
                "bcc": {
                    "type": "string",
                    "description": "BCC recipients (comma-separated, optional)"
                }
            },
            "required": ["to", "subject", "body"]
        }
    },
    {
        "name": "gmail_reply_to_email",
        "description": "Reply to an existing email thread.",
        "parameters": {
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "ID of the email thread to reply to"
                },
                "body": {
                    "type": "string",
                    "description": "Reply message content"
                },
                "reply_all": {
                    "type": "boolean",
                    "description": "Reply to all recipients (default: false)"
                }
            },
            "required": ["thread_id", "body"]
        }
    },
    {
        "name": "gmail_get_email",
        "description": "Get the full content of a specific email.",
        "parameters": {
            "type": "object",
            "properties": {
                "email_id": {
                    "type": "string",
                    "description": "The unique ID of the email"
                }
            },
            "required": ["email_id"]
        }
    },
    {
        "name": "gmail_label_email",
        "description": "Add or remove labels from an email.",
        "parameters": {
            "type": "object",
            "properties": {
                "email_id": {
                    "type": "string",
                    "description": "The unique ID of the email"
                },
                "add_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels to add (e.g., ['STARRED', 'IMPORTANT'])"
                },
                "remove_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels to remove (e.g., ['UNREAD'])"
                }
            },
            "required": ["email_id"]
        }
    }
]

# =============================================================================
# NOTION TOOLS (6 tools)
# =============================================================================

NOTION_TOOLS = [
    {
        "name": "notion_search",
        "description": "Search for pages and databases in Notion.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text"
                },
                "filter": {
                    "type": "string",
                    "enum": ["page", "database", "all"],
                    "description": "Filter by type (default: all)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 10)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "notion_create_page",
        "description": "Create a new page in a Notion database or as a child of another page.",
        "parameters": {
            "type": "object",
            "properties": {
                "parent_id": {
                    "type": "string",
                    "description": "Database ID or parent page ID"
                },
                "title": {
                    "type": "string",
                    "description": "Page title"
                },
                "content": {
                    "type": "string",
                    "description": "Initial page content (optional)"
                },
                "properties": {
                    "type": "object",
                    "description": "Additional database properties as key-value pairs (optional)"
                }
            },
            "required": ["parent_id", "title"]
        }
    },
    {
        "name": "notion_update_page",
        "description": "Update properties of an existing Notion page.",
        "parameters": {
            "type": "object",
            "properties": {
                "page_id": {
                    "type": "string",
                    "description": "The page ID to update"
                },
                "properties": {
                    "type": "object",
                    "description": "Properties to update as key-value pairs"
                }
            },
            "required": ["page_id", "properties"]
        }
    },
    {
        "name": "notion_query_database",
        "description": "Query a Notion database with filters and sorts.",
        "parameters": {
            "type": "object",
            "properties": {
                "database_id": {
                    "type": "string",
                    "description": "The database ID to query"
                },
                "filter": {
                    "type": "object",
                    "description": "Filter conditions (optional)"
                },
                "sorts": {
                    "type": "array",
                    "description": "Sort conditions (optional)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 10)"
                }
            },
            "required": ["database_id"]
        }
    },
    {
        "name": "notion_append_blocks",
        "description": "Append content blocks to an existing page.",
        "parameters": {
            "type": "object",
            "properties": {
                "page_id": {
                    "type": "string",
                    "description": "The page ID to append to"
                },
                "content": {
                    "type": "string",
                    "description": "Text content to append"
                },
                "block_type": {
                    "type": "string",
                    "enum": ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list", "numbered_list", "to_do", "code"],
                    "description": "Type of block to create (default: paragraph)"
                }
            },
            "required": ["page_id", "content"]
        }
    },
    {
        "name": "notion_get_page",
        "description": "Get the content and properties of a Notion page.",
        "parameters": {
            "type": "object",
            "properties": {
                "page_id": {
                    "type": "string",
                    "description": "The page ID to retrieve"
                }
            },
            "required": ["page_id"]
        }
    }
]

# =============================================================================
# SERVER CONFIGURATIONS
# =============================================================================

MCP_SERVERS = {
    "google_calendar": {
        "name": "Google Calendar",
        "description": "Manage calendar events, schedules, and meetings",
        "package": "@anthropic/mcp-server-google-calendar",
        "auth_type": "oauth",
        "tools": GOOGLE_CALENDAR_TOOLS,
        "tool_count": len(GOOGLE_CALENDAR_TOOLS)
    },
    "github": {
        "name": "GitHub",
        "description": "Manage repositories, issues, and pull requests",
        "package": "@anthropic/mcp-server-github",
        "auth_type": "token",
        "tools": GITHUB_TOOLS,
        "tool_count": len(GITHUB_TOOLS)
    },
    "slack": {
        "name": "Slack",
        "description": "Send messages, search channels, and manage workspace communication",
        "package": "@anthropic/mcp-server-slack",
        "auth_type": "oauth",
        "tools": SLACK_TOOLS,
        "tool_count": len(SLACK_TOOLS)
    },
    "gmail": {
        "name": "Gmail",
        "description": "Search, read, send, and manage emails",
        "package": "@anthropic/mcp-server-gmail",
        "auth_type": "oauth",
        "tools": GMAIL_TOOLS,
        "tool_count": len(GMAIL_TOOLS)
    },
    "notion": {
        "name": "Notion",
        "description": "Manage pages, databases, and documentation",
        "package": "@anthropic/mcp-server-notion",
        "auth_type": "token",
        "tools": NOTION_TOOLS,
        "tool_count": len(NOTION_TOOLS)
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_server_config(server_id: str) -> Dict[str, Any]:
    """Get configuration for a specific MCP server."""
    if server_id not in MCP_SERVERS:
        raise ValueError(f"Unknown server: {server_id}. Available: {list(MCP_SERVERS.keys())}")
    return MCP_SERVERS[server_id]


def get_tools_for_server(server_id: str) -> List[Dict[str, Any]]:
    """Get all tools for a specific MCP server."""
    config = get_server_config(server_id)
    return config["tools"]


def get_all_tools() -> List[Dict[str, Any]]:
    """Get all tools from all servers with server prefix."""
    all_tools = []
    for server_id, config in MCP_SERVERS.items():
        for tool in config["tools"]:
            # Add server info to each tool
            tool_with_server = {
                **tool,
                "server": server_id,
                "server_name": config["name"]
            }
            all_tools.append(tool_with_server)
    return all_tools


def get_tool_summary() -> str:
    """Get a human-readable summary of all available tools."""
    lines = ["Available MCP Servers and Tools:", "=" * 50]
    
    for server_id, config in MCP_SERVERS.items():
        lines.append(f"\n📦 {config['name']} ({config['tool_count']} tools)")
        lines.append(f"   {config['description']}")
        for tool in config["tools"]:
            lines.append(f"   • {tool['name']}: {tool['description'][:60]}...")
    
    return "\n".join(lines)


# Quick stats
TOTAL_TOOLS = sum(server["tool_count"] for server in MCP_SERVERS.values())
TOTAL_SERVERS = len(MCP_SERVERS)


if __name__ == "__main__":
    print(get_tool_summary())
    print(f"\n📊 Total: {TOTAL_SERVERS} servers, {TOTAL_TOOLS} tools")
