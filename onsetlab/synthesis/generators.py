"""
LLM Prompt Generators for Training Data
========================================
Builds prompts for single-tool, clarification, follow-up, refusal, and casual
example generation. Used by DataGenerator; no API calls here—pure prompt construction.

IMPORTANT: All prompts are GENERIC - no hardcoded service names (GitHub, Slack, etc.)
The actual tools are passed in via tools_desc which is dynamically built.
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
   - string params: "name": "value"
   - array params: "labels": ["item1", "item2"] (use [...] syntax)
   - number params: "limit": 10 (NO quotes around numbers)
   - boolean params: "active": true (use true/false, no quotes)
   - object params: "data": {{"key": "value", "nested": "field"}} (nested dict structure)

PATTERN FOR GOOD EXAMPLES:
- User query explicitly contains ALL required parameter values
- Tool parameters are extracted directly from what the user said
- No guessing or placeholder values

PATTERN FOR BAD EXAMPLES (DO NOT GENERATE):
- User query is vague and missing required info
- Parameters contain made-up values not in the query
- Placeholder text like "your_value" or "example"

IMPORTANT - OBJECT TYPE PARAMETERS:
If a param shows "object({{...}})", you MUST generate a NESTED DICT, not a string!
Example: event:object({{summary:string, start:string}}) means generate:
  "event": {{"summary": "Team meeting", "start": "2024-03-15T09:00:00"}}
NOT: "event": "Team meeting at 9am" ← WRONG!

Output JSON array (note: object params use nested dicts):
[
  {{"query": "user request", "tool": "tool_name", "parameters": {{"stringParam": "value", "objectParam": {{"nestedKey": "nestedValue"}}}}}}
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

The clarification should:
- Be friendly and helpful
- Ask specifically for the missing parameter(s)
- Not assume or guess values

Output JSON array:
[
  {{"query": "incomplete user request", "response": "friendly question asking for missing info", "missing_params": ["param1", "param2"]}}
]

Generate {batch_size} diverse clarification examples using the tools above:"""


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

COMMON FOLLOW-UP PATTERNS (generic):
1. "what about the [filter]" (filter change)
2. "show me just [N]" (limit change)
3. "same but for [other target]" (param swap)
4. "any from [time period]?" (time filter)
5. "what's the most recent?" (sort/limit)
6. "now include [extra field]" (param addition)

OUTPUT FORMAT (JSON array):
[
  {{"query": "[Conversation Context]\\nUser asked: <prev query>\\nAgent called <tool> -> <result>\\n[Current Query]\\n<follow-up>", "tool": "tool_name", "params": {{"param": "value"}}}}
]

RULES:
1. The follow-up query must be SHORT and natural (3-8 words)
2. Carry forward ALL relevant params from the previous query
3. Apply the modification from the follow-up (filter, limit, sort, etc.)
4. Use EXACT tool names and valid param values from the list above
5. ENUM VALUES ARE CASE-SENSITIVE: Use values exactly as shown in the tool list

Generate {batch_size} diverse follow-up examples using the tools above:"""


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

The refusal should:
- Be polite and helpful
- Explain the limitation briefly
- Redirect to available capabilities (use the tools listed above)

COMMON OUT-OF-SCOPE REQUESTS (adapt to context):
- Asking to perform actions the tools don't support
- Requesting integrations not available
- Asking for general knowledge unrelated to the tools

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

3. CAPABILITY QUESTIONS (use the actual tools above in responses):
   - "What can you do?" → Describe the available tools
   - "What are you?" → Describe yourself based on the context
   - "help" → List what you can help with

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

IMPORTANT: For capability questions, use the ACTUAL tools listed above in your responses.

Generate {batch_size} DIVERSE examples from ALL categories:"""
