"""
LLM Prompt Generators for Training Data
========================================
Builds prompts for single-tool, clarification, follow-up, refusal, and casual
example generation. Used by DataGenerator; no API calls here—pure prompt construction.

IMPORTANT: All prompts are GENERIC - no hardcoded service names (GitHub, Slack, etc.)
When a skill is provided, it guides the generation with exact formats and examples.
Otherwise, tools_desc is used (fallback for backward compatibility).
"""


def build_single_tool_prompt(
    problem_statement: str,
    tools_desc: str,
    batch_size: int,
    skill: str = None,
) -> str:
    """Build prompt for generating single-tool (successful tool call) examples."""
    
    # If skill is provided, use it for detailed guidance
    if skill:
        return f"""Generate {batch_size} training examples for an AI assistant.

Context: {problem_statement}

=== SKILL REFERENCE (FOLLOW THIS EXACTLY) ===
{skill}
=== END SKILL REFERENCE ===

GENERATION RULES:
1. Use the EXACT tool names, parameter names, and formats shown in the skill above
2. Use the "Correct Examples" from the skill as TEMPLATES for your generations
3. Use the "Triggers" phrases as inspiration for user queries
4. AVOID the patterns shown in "Wrong Examples"
5. The USER QUERY must CONTAIN all required parameter values explicitly
6. DO NOT make up values - extract from the user query

CRITICAL - PARAMETER FORMATS:
- Follow the EXACT formats shown in the skill's parameter tables
- For OBJECT parameters (like "event"), use the EXACT nested structure shown
- Copy the format from the skill's correct examples

Output JSON array:
[
  {{"query": "user request", "tool": "tool_name", "parameters": {{...}}}}
]

Generate {batch_size} diverse examples following the skill's formats exactly:"""
    
    # Fallback: no skill provided, use basic prompt
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

DATE-RELATIVE QUERIES (IMPORTANT):
For tools that need dates, include a [Current: DATE] context in the query:
- "[Current: Monday, February 10, 2025] what's on my calendar today" → Use 2025-02-10 for date params
- "[Current: Friday, March 15, 2024] meetings this week" → Use dates relative to that Friday
- The model MUST learn to extract dates from the [Current: ...] context

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

For tools that require dates, include [Current: DATE] context in query and extract the date for params.
Example pattern: "[Current: Monday, Feb 10, 2025] <request about today/this week>" → use 2025-02-10

Generate {batch_size} diverse examples where the user query contains ALL required information:"""


def build_clarification_prompt(
    problem_statement: str,
    tools_required_desc: str,
    batch_size: int,
    skill: str = None,
) -> str:
    """Build prompt for clarification examples (user missing required params)."""
    
    # If skill provided, use its clarification patterns
    skill_section = ""
    if skill:
        skill_section = f"""
=== SKILL REFERENCE (for clarification patterns) ===
{skill}
=== END SKILL REFERENCE ===

Use the "Clarification" sections from the skill above to guide what questions to ask.
"""
    
    return f"""Generate {batch_size} examples where user wants to use a tool but is MISSING required information.

Context: {problem_statement}

TOOLS AND THEIR REQUIRED PARAMS:
{tools_required_desc}
{skill_section}
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
- NEVER include "I'm sorry, I can't help with that" alone - always explain WHY and what you CAN do

CATEGORIES OF OUT-OF-SCOPE REQUESTS:

1. REAL-TIME INFO (can't provide current time/weather/news):
   - "what time is it" → "I don't have access to the current time. What would you like help with?"
   - "what's the weather" → "I can't check weather, but I can help with [mention your actual tools]"
   - "what's in the news" → "I don't have access to news, but I can help with [your tools]"

2. ACTIONS NOT SUPPORTED:
   - Asking to do things the tools can't do
   - Requesting features that don't exist

3. WRONG DOMAIN:
   - General knowledge questions unrelated to your tools
   - Math, coding, trivia that isn't tool-related

4. VAGUE REQUESTS THAT NEED CLARIFICATION:
   - "do that thing" → "I'd be happy to help! Could you be more specific about what you'd like me to do?"
   - "change it" → "Sure! What would you like me to change?"

Output JSON array:
[
  {{"query": "out of scope request", "response": "polite decline WITH explanation of what you CAN do"}}
]

IMPORTANT: Responses must be HELPFUL, not just apologetic. Always redirect to capabilities.

Generate {batch_size} diverse refusal examples covering ALL categories above:"""


def build_casual_prompt(
    tool_list_str: str,
    batch_size: int,
) -> str:
    """Build prompt for casual conversation examples (no tool call)."""
    return f"""Generate {batch_size} casual conversation examples that DON'T need any tool call.

The agent has these tools: {tool_list_str}

CRITICAL RULES:
1. These are messages where the user is NOT asking to DO something
2. Respond with FRIENDLY, WARM text - never apologize or refuse
3. NEVER use phrases like "I can't help with that" - these are friendly exchanges!
4. Keep responses SHORT and conversational (1-2 sentences max)

CATEGORIES (generate equal mix):

1. GREETINGS (40% of examples) - Always respond warmly:
   - "hey" → "Hey! What can I help you with?"
   - "hi" → "Hi there! How can I assist you today?"
   - "hello" → "Hello! What can I do for you?"
   - "hey there" → "Hey! Ready to help. What do you need?"
   - "good morning" → "Good morning! What can I help with today?"
   - "sup" → "Hey! What's up? How can I help?"
   - "yo" → "Hey! What do you need?"
   - "hiya" → "Hi! What can I do for you?"

2. THANKS (20% of examples):
   - "thanks" → "You're welcome! Need anything else?"
   - "thank you" → "Happy to help! Let me know if you need more."
   - "thanks a lot" → "Anytime! What else can I do?"
   - "perfect" → "Great! Let me know if you need anything else."
   - "awesome" → "Glad that worked! Anything else?"
   - "that helps" → "Great! Let me know what else you need."

3. ACKNOWLEDGMENTS (20% of examples):
   - "ok" → "Sounds good! Let me know when you need something."
   - "got it" → "Great! What's next?"
   - "I see" → "Yep! Anything else I can help with?"
   - "cool" → "Nice! What else can I do?"
   - "sounds good" → "Perfect! Let me know if you need anything."
   - "alright" → "Great! What else?"

4. CAPABILITY QUESTIONS (20% of examples) - Mention actual tools:
   - "what can you do" → Briefly describe using {tool_list_str}
   - "help" → List your capabilities using the tools above
   - "what are you" → Describe yourself as an assistant with the tools above

Output JSON array:
[
  {{"query": "casual message", "response": "friendly SHORT response"}}
]

NEVER INCLUDE APOLOGIES OR REFUSALS. These are friendly exchanges.

Generate {batch_size} DIVERSE examples with 40% greetings, 20% thanks, 20% acknowledgments, 20% capability questions:"""
