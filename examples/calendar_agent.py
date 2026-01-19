"""
Example: Building a Calendar Agent with OnsetLab
================================================

This example shows how to use the OnsetLab SDK to generate training data
for a calendar management agent.

Prerequisites:
- pip install onsetlab
- OpenAI or Anthropic API key
"""

import os
from onsetlab import AgentBuilder
from onsetlab.utils import ToolSchema, load_tools_from_file

# Load tool schemas from JSON file
tools = load_tools_from_file("calendar_tools.json")

print(f"Loaded {len(tools)} tools:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description}")

# Option 1: Just generate training data (no fine-tuning)
if False:  # Set to True to run
    from onsetlab.synthesis import generate_training_data
    
    training_path, prompt_path = generate_training_data(
        problem_statement="I need an assistant that manages my Google Calendar",
        tools=tools,
        api_key=os.getenv("OPENAI_API_KEY"),
        num_examples=300,
        output_path="calendar_training_data.jsonl"
    )
    
    print(f"\nGenerated:")
    print(f"  Training data: {training_path}")
    print(f"  System prompt: {prompt_path}")

# Option 2: Generate just a system prompt (no API needed)
if True:
    from onsetlab.synthesis import generate_minimal_prompt
    
    prompt = generate_minimal_prompt(
        problem_statement="I need an assistant that manages my Google Calendar",
        tools=tools
    )
    
    print("\n" + "=" * 60)
    print("Generated System Prompt:")
    print("=" * 60)
    print(prompt)

# Option 3: Full pipeline with AgentBuilder (coming soon)
# This will be available after all SDK components are implemented
#
# from onsetlab import AgentBuilder, MCPServerConfig
# 
# builder = AgentBuilder(
#     problem_statement="I need an assistant that manages my Google Calendar",
#     tools=tools,
#     mcp_servers=[
#         MCPServerConfig(
#             package="@cocal/google-calendar-mcp",
#             auth_type="oauth",
#             env_var="GOOGLE_OAUTH_CREDENTIALS"
#         )
#     ],
#     api_key=os.getenv("OPENAI_API_KEY")
# )
# 
# agent = builder.build()
# agent.save("./my_calendar_agent")
