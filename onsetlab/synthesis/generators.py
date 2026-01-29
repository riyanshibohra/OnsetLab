"""
LLM Prompt Generators for Training Data
========================================
Builds prompts for single-tool, clarification, follow-up, refusal, and casual
example generation. Used by DataGenerator; no API calls here—pure prompt construction.
"""


def build_single_tool_prompt(
    problem_statement: str,
    tools_desc: str,
    batch_size: int,
) -> str:
    """Build prompt for generating single-tool (successful tool call) examples."""
    return f"""Generate {batch_size} training examples for an AI assistant.

Context: {problem_statement}

TOOLS (use EXACT names and correct TYPES):
{tools_desc}

CRITICAL - ENUM VALUES ARE CASE-SENSITIVE:
- If a param shows enum(OPEN|CLOSED), you MUST use "OPEN" or "CLOSED" exactly (uppercase)
- Do NOT use lowercase like "open" or "closed" - the API will reject it
- Always copy enum values EXACTLY as shown in the param list above

CRITICAL RULES:
1. Tool name must be EXACTLY from the list above
2. The USER QUERY must CONTAIN all required parameter values explicitly
3. DO NOT make up/hallucinate values that the user didn't provide!
4. TYPES ARE CRITICAL:
   - array params MUST use [...] syntax: "labels": ["bug", "urgent"]
   - number params MUST NOT be quoted: "limit": 10 (NOT "limit": "10")
   - boolean params use true/false: "active": true

GOOD examples (user provides all info):
- "Create issue in facebook/react titled 'Bug in hooks'" → owner=facebook, repo=react, title=Bug in hooks ✓
- "List open issues in kubernetes/kubernetes with bug label" → all params from query ✓
- "Send 'Deploy complete' to channel C123456" → channel and message from query ✓

BAD examples (DO NOT generate these - info is missing):
- "Create an issue about the bug" → NO owner/repo specified! ✗
- "Comment on issue 73" → NO owner/repo specified! ✗
- "Send a message to the team" → NO channel specified! ✗

Output JSON array:
[
  {{"query": "user request with ALL required info", "tool": "exact_tool_name", "parameters": {{"param": "value_from_query"}}}}
]

Generate {batch_size} diverse examples where the user query contains ALL required information:"""


def build_clarification_prompt(
    problem_statement: str,
    tools_required_desc: str,
    batch_size: int,
) -> str:
    """Build prompt for clarification examples (user missing required params)."""
    return f"""Generate {batch_size} examples where user wants to use a tool but is MISSING required information.

Context: {problem_statement}

TOOLS AND THEIR REQUIRED PARAMS:
{tools_required_desc}

PATTERN:
1. User makes a request but leaves out 1-2 required parameters
2. Assistant asks a SPECIFIC question to get the missing info
3. Do NOT guess or use placeholders - ask!

EXAMPLES:
- User: "create an issue" → Missing: repo, title → Ask: "Which repository should I create the issue in, and what should the title be?"
- User: "send a message to the team" → Missing: channel, message → Ask: "Which Slack channel should I send to, and what's the message?"
- User: "schedule a meeting tomorrow" → Missing: time, title → Ask: "What time tomorrow, and what should I call the meeting?"

Output JSON array:
[
  {{"query": "incomplete user request", "response": "friendly question asking for missing info", "missing_params": ["param1", "param2"]}}
]

Generate {batch_size} diverse clarification examples:"""


def build_follow_up_prompt(
    tools_required_desc: str,
    batch_size: int,
) -> str:
    """Build prompt for follow-up (context-aware) examples."""
    return f"""Generate follow-up query training examples for a conversational agent.

TOOLS AVAILABLE:
{tools_required_desc}

SCENARIO: The user previously asked a question, got a response, and now asks a SHORT follow-up
that references the previous context. The follow-up query should be BRIEF (like real users).

FORMAT - The query MUST include conversation context in this EXACT format:
[Conversation Context]
User asked: <previous query>
Agent called <tool_name> -> <brief result summary>
[Current Query]
<short follow-up question>

EXAMPLES OF GOOD FOLLOW-UP PATTERNS:
1. "what about the open ones" (filter change)
2. "show me just 5" (limit change)  
3. "same but for the other repo" (param swap)
4. "any from last week?" (time filter)
5. "what's the most recent?" (sort/limit)
6. "now search in the body too" (param addition)

OUTPUT FORMAT (JSON array):
[
  {{"query": "[Conversation Context]\\nUser asked: find issues in owner/repo\\nAgent called list_issues -> Found 5 issues: #1 Bug, #2 Feature...\\n[Current Query]\\nwhat about the open ones", "tool": "list_issues", "params": {{"owner": "owner", "repo": "repo", "state": "OPEN"}}}}
]

RULES:
1. The follow-up query must be SHORT and natural (3-8 words)
2. Carry forward ALL relevant params from the previous query
3. Apply the modification from the follow-up (filter, limit, sort, etc.)
4. Use EXACT tool names and valid param values from the list above
5. ENUM VALUES ARE CASE-SENSITIVE: Use "OPEN"/"CLOSED" exactly as shown, NOT lowercase

Generate {batch_size} diverse follow-up examples:"""


def build_refusal_prompt(
    problem_statement: str,
    tool_names: list,
    batch_size: int,
) -> str:
    """Build prompt for refusal examples (out of scope requests)."""
    tools_str = ", ".join(tool_names[:10])
    return f"""Generate {batch_size} examples where user asks for something the assistant CANNOT do.

Context: {problem_statement}
Available tools: {tools_str}

PATTERN:
1. User asks for something outside the available tools
2. Assistant politely declines and mentions what they CAN help with

EXAMPLES:
- User: "Can you book a flight for me?" → "I can't book flights, but I can help you with GitHub, Slack, or scheduling meetings."
- User: "Send an email to john@example.com" → "I don't have email capabilities, but I can send Slack messages if that helps."
- User: "What's the weather?" → "I can't check weather, but I can search the web or help with your calendar."

Output JSON array:
[
  {{"query": "out of scope request", "response": "polite decline mentioning available capabilities"}}
]

Generate {batch_size} diverse refusal examples:"""


def build_casual_prompt(
    tool_list_str: str,
    batch_size: int,
) -> str:
    """Build prompt for casual conversation examples (no tool call)."""
    return f"""Generate {batch_size} casual conversation examples that DON'T need any tool call.

The agent has these tools: {tool_list_str}

CRITICAL: These are messages where the user is NOT asking to DO something.
The assistant should respond with friendly text, NO tool calls.

CATEGORIES TO COVER (mix all types):

1. GREETINGS:
   - "Hey!" → "Hey there! How can I help you today?"
   - "Good morning" → "Good morning! What can I help you with?"
   - "hi" → "Hi! What can I do for you?"

2. THANKS:
   - "Thanks!" → "You're welcome! Let me know if you need anything else."
   - "That was helpful" → "Glad I could help! Anything else?"
   - "perfect" → "Great! Let me know if you need anything else."

3. CAPABILITY QUESTIONS:
   - "What can you do?" → "I can help with GitHub issues and Slack. I can list issues, create them, add comments, and send Slack messages. What would you like me to do?"
   - "What are you?" → "I'm an assistant that helps manage GitHub issues and Slack messages."
   - "help" → "I'm here to help! I can work with GitHub issues and Slack. What do you need?"

4. ACKNOWLEDGMENTS:
   - "ok" → "Got it! Let me know when you need something."
   - "I see" → "Yep! Anything else?"
   - "cool" → "Glad that works! What's next?"

5. CONFUSION:
   - "I don't understand" → "No problem! What would you like me to explain?"
   - "wait what" → "Sorry if I was unclear. How can I help?"

Output JSON array:
[
  {{"query": "casual message", "response": "friendly response"}}
]

Generate {batch_size} DIVERSE examples from ALL categories:"""
