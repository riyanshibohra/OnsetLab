"""
Interactive test - see exactly what the data generator produces.

Run this to see:
1. Sample size recommendations
2. Example chat formats (what training data looks like)
3. Optionally: Generate real data with your API key
"""

import json

# Sample tools (subset from our registry)
SAMPLE_TOOLS = [
    {
        "name": "search_repositories",
        "description": "Search for GitHub repositories",
        "parameters": {"query": {"type": "string"}, "page": {"type": "number"}},
        "required_params": ["query"]
    },
    {
        "name": "create_issue",
        "description": "Create a new issue in a repository",
        "parameters": {"owner": {"type": "string"}, "repo": {"type": "string"}, "title": {"type": "string"}},
        "required_params": ["owner", "repo", "title"]
    },
    {
        "name": "list_issues",
        "description": "List issues in a repository",
        "parameters": {"owner": {"type": "string"}, "repo": {"type": "string"}, "state": {"type": "string"}},
        "required_params": ["owner", "repo"]
    },
    {
        "name": "conversations_add_message",
        "description": "Send a message to a Slack channel",
        "parameters": {"channel_id": {"type": "string"}, "text": {"type": "string"}},
        "required_params": ["channel_id", "text"]
    },
    {
        "name": "memory_set",
        "description": "Store a value in memory",
        "parameters": {"key": {"type": "string"}, "value": {"type": "string"}},
        "required_params": ["key", "value"]
    },
    {
        "name": "memory_get",
        "description": "Retrieve a value from memory",
        "parameters": {"key": {"type": "string"}},
        "required_params": ["key"]
    },
]


def show_sample_sizes():
    """Show sample size recommendations for different tool counts."""
    from onsetlab.synthesis.data_generator import print_dataset_recommendation
    
    print("\n" + "="*60)
    print("📊 SAMPLE SIZE RECOMMENDATIONS")
    print("="*60)
    
    # Our sample has 6 tools
    print_dataset_recommendation(6)
    
    # Full registry has 54 tools
    print("\nFor the FULL registry (54 tools):")
    print_dataset_recommendation(54)


def show_example_outputs():
    """Show what training examples look like."""
    from onsetlab.synthesis.data_generator import DataGenerator, DataGenConfig
    
    print("\n" + "="*60)
    print("📝 EXAMPLE TRAINING DATA")
    print("="*60)
    
    config = DataGenConfig()
    generator = DataGenerator(
        tools=SAMPLE_TOOLS,
        problem_statement="Help users manage GitHub and Slack",
        api_key="fake-key",  # Won't make API calls
        config=config,
        system_prompt="You are a helpful assistant for GitHub and Slack tasks."
    )
    
    # Example 1: Successful tool call
    print("\n" + "-"*60)
    print("EXAMPLE 1: Successful Tool Call")
    print("-"*60)
    
    example1 = {
        "query": "Create an issue titled 'Bug fix' in riyanshibohra/tunekit",
        "tool": "create_issue",
        "parameters": {"owner": "riyanshibohra", "repo": "tunekit", "title": "Bug fix"}
    }
    
    chat1 = generator._to_chat_format(example1, "tool_call")
    print("\nTraining example (JSON):")
    print(json.dumps(chat1, indent=2))
    
    print("\n👤 Conversation flow:")
    for msg in chat1['messages']:
        role = msg['role'].upper()
        content = msg['content']
        print(f"\n[{role}]")
        print(content)
    
    # Example 2: Clarification (missing params)
    print("\n" + "-"*60)
    print("EXAMPLE 2: Clarification (Missing Parameters)")
    print("-"*60)
    
    example2 = {
        "query": "Create an issue",
        "response": "I'd be happy to help create an issue! Which repository should I create it in, and what should the title be?"
    }
    
    chat2 = generator._to_chat_format(example2, "clarification")
    print("\nTraining example (JSON):")
    print(json.dumps(chat2, indent=2))
    
    print("\n👤 Conversation flow:")
    for msg in chat2['messages']:
        role = msg['role'].upper()
        content = msg['content']
        print(f"\n[{role}]")
        print(content)
    
    # Example 3: Memory operation
    print("\n" + "-"*60)
    print("EXAMPLE 3: Memory Operation")
    print("-"*60)
    
    example3 = {
        "query": "Remember that my favorite repo is tunekit",
        "tool": "memory_set",
        "parameters": {"key": "favorite_repo", "value": "tunekit"}
    }
    
    chat3 = generator._to_chat_format(example3, "tool_call")
    print("\n👤 Conversation flow:")
    for msg in chat3['messages']:
        role = msg['role'].upper()
        content = msg['content']
        print(f"\n[{role}]")
        print(content)
    
    # Example 4: Casual conversation
    print("\n" + "-"*60)
    print("EXAMPLE 4: Casual Conversation")
    print("-"*60)
    
    example4 = {
        "query": "Thanks, that's exactly what I needed!",
        "response": "You're welcome! Let me know if you need anything else."
    }
    
    chat4 = generator._to_chat_format(example4, "casual")
    print("\n👤 Conversation flow:")
    for msg in chat4['messages']:
        role = msg['role'].upper()
        content = msg['content']
        print(f"\n[{role}]")
        print(content)


def generate_real_data(api_key: str, output_dir: str = "./test_generated_data"):
    """Generate REAL training data using API."""
    from onsetlab.synthesis.data_generator import generate_training_data
    
    print("\n" + "="*60)
    print("🚀 GENERATING REAL TRAINING DATA")
    print("="*60)
    
    datasets = generate_training_data(
        tools=SAMPLE_TOOLS,
        problem_statement="Help users manage GitHub, Slack, and remember their preferences",
        api_key=api_key,
        output_dir=output_dir,
        examples_per_tool=10  # Small for testing (normally 25)
    )
    
    print(f"\n✅ Generated data saved to: {output_dir}")
    print(f"   - train.jsonl: {len(datasets.get('train', []))} examples")
    print(f"   - validation.jsonl: {len(datasets.get('validation', []))} examples")
    print(f"   - test.jsonl: {len(datasets.get('test', []))} examples")
    
    return datasets


if __name__ == "__main__":
    print("\n" + "🧪"*30)
    print("   DATA GENERATOR - INTERACTIVE TEST")
    print("🧪"*30)
    
    # 1. Show sample sizes
    show_sample_sizes()
    
    # 2. Show example outputs
    show_example_outputs()
    
    # 3. Optional: Generate real data
    print("\n" + "="*60)
    print("🔑 GENERATE REAL DATA (Optional)")
    print("="*60)
    print("\nTo generate real training data, run in Python:")
    print("""
    from test_interactive import generate_real_data
    
    # Use your OpenAI or Anthropic key
    datasets = generate_real_data("sk-your-api-key-here")
    """)
