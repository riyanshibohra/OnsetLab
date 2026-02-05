"""
ReAct Fallback - iterative reasoning when REWOO fails.

This is a silent fallback, not the primary execution mode.
Only triggered when REWOO planning/execution fails.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple

from ..model.base import BaseModel
from ..tools.base import BaseTool


REACT_PROMPT = '''Use ONE tool to complete the task.

Tools:
{tools_description}

Format:
Thought: [reason]
Action: tool(param="value")
{context}
Task: {task}
{scratchpad}
Thought:'''


class ReactFallback:
    """
    Simple ReAct implementation for fallback when REWOO fails.
    
    This is NOT the primary execution mode - only used as fallback.
    """
    
    def __init__(
        self,
        model: BaseModel,
        tools: List[BaseTool],
        max_iterations: int = 5,
        debug: bool = False
    ):
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations
        self.debug = debug
    
    def run(
        self,
        task: str,
        context: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, str]]:
        """
        Run ReAct loop.
        
        Returns:
            Tuple of (answer, steps_taken, results)
        """
        # If context mentions a specific tool that failed, prioritize showing it
        focus_tool = None
        if context and "Tool used:" in context:
            import re
            match = re.search(r'Tool used:\s*(\w+)', context)
            if match:
                focus_tool = match.group(1)
        
        tools_desc = self._format_tools(focus_tool=focus_tool)
        scratchpad = ""
        steps_taken = []
        results = {}
        
        context_str = f"{context}\n" if context else ""
        
        for iteration in range(self.max_iterations):
            # Build prompt
            prompt = REACT_PROMPT.format(
                tools_description=tools_desc,
                context=context_str,
                task=task,
                scratchpad=scratchpad
            )
            
            # Generate
            response = self.model.generate(
                prompt,
                temperature=0.0,
                max_tokens=300,
                stop_sequences=["Observation:", "\n\n\n"]
            )
            
            if self.debug:
                print(f"[ReAct {iteration+1}] Response: {response[:200]}...")
            
            # Parse response
            thought, action, is_final, final_answer = self._parse_response(response)
            
            # Add thought to scratchpad
            scratchpad += f" {thought}\n"
            
            # Check for final answer
            if is_final and final_answer:
                return final_answer, steps_taken, results
            
            # Execute action if present
            if action:
                tool_name, params = action
                step_id = f"#R{iteration+1}"
                
                observation = self._execute_action(tool_name, params)
                results[step_id] = observation
                
                steps_taken.append({
                    "id": step_id,
                    "tool": tool_name,
                    "params": params,
                    "result": observation
                })
                
                # Add to scratchpad
                scratchpad += f"Action: {tool_name}({self._format_params(params)})\n"
                scratchpad += f"Observation: {observation[:500]}\n"
                scratchpad += "Thought:"
            else:
                # No action and no final answer - model might be stuck
                scratchpad += "Action: none\nObservation: No action taken.\nThought:"
        
        # Max iterations reached - build helpful failure message
        failed_tools = list(set(s.get("tool", "unknown") for s in steps_taken))
        errors = [r for r in results.values() if "Error" in str(r) or "Cannot" in str(r)]
        
        failure_msg = f"I couldn't complete this task after {self.max_iterations} attempts."
        
        if failed_tools:
            failure_msg += f"\n\nTools attempted: {', '.join(failed_tools)}"
        
        if errors:
            failure_msg += f"\n\nLast error: {errors[-1][:200]}"
        
        failure_msg += "\n\nSuggestions:"
        failure_msg += "\n1. Try a simpler request"
        failure_msg += "\n2. Use a larger model (e.g., llama3.1:8b, mistral:7b)"
        failure_msg += "\n3. Check if the tool parameters match what you're asking for"
        
        return failure_msg, steps_taken, results
    
    def _format_tools(self, focus_tool: Optional[str] = None) -> str:
        """Format tools for prompt. If focus_tool is set, show it first and limit others."""
        lines = []
        
        # If we have a focus tool, show it first with full details
        if focus_tool and focus_tool in self.tools:
            tool = self.tools[focus_tool]
            lines.append(f"PRIMARY TOOL (use this):")
            lines.append(self._format_single_tool(focus_tool, tool, detailed=True))
            lines.append("")
            lines.append("Other tools (if needed):")
            
            # Show only 5 other tools
            count = 0
            for name, t in self.tools.items():
                if name != focus_tool and count < 5:
                    lines.append(self._format_single_tool(name, t, detailed=False))
                    count += 1
        else:
            # No focus - show all tools (limited)
            for name, tool in list(self.tools.items())[:10]:
                lines.append(self._format_single_tool(name, tool, detailed=False))
        
        return "\n".join(lines)
    
    def _format_single_tool(self, name: str, tool: BaseTool, detailed: bool = False) -> str:
        """Format a single tool - SLM optimized."""
        params = tool.parameters
        
        if "properties" in params:
            props = params.get("properties", {})
            required = params.get("required", [])
        else:
            props = params
            required = [p for p, d in params.items() 
                       if isinstance(d, dict) and d.get("required")]
        
        # Only show required params
        req_parts = []
        for p, details in props.items():
            if not isinstance(details, dict):
                continue
            is_req = p in required or details.get("required")
            if not is_req:
                continue
            
            if "enum" in details:
                req_parts.append(f'{p}="{details["enum"][0]}"')
            elif details.get("type") == "integer":
                req_parts.append(f'{p}=N')
            else:
                req_parts.append(f'{p}="..."')
        
        sig = f"{name}({', '.join(req_parts)})" if req_parts else f"{name}()"
        desc = tool.description[:60]
        return f"{sig} - {desc}"
    
    def _parse_response(
        self,
        response: str
    ) -> Tuple[str, Optional[Tuple[str, Dict]], bool, Optional[str]]:
        """
        Parse ReAct response.
        
        Returns:
            Tuple of (thought, action, is_final, final_answer)
        """
        thought = response.strip()
        action = None
        is_final = False
        final_answer = None
        
        # Check for Final Answer
        final_match = re.search(r'Final Answer:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if final_match:
            is_final = True
            final_answer = final_match.group(1).strip()
            # Clean up - stop at newlines
            final_answer = final_answer.split('\n')[0].strip()
            return thought, None, is_final, final_answer
        
        # Check for Action
        action_match = re.search(r'Action:\s*(\w+)\s*\(([^)]*)\)', response)
        if action_match:
            tool_name = action_match.group(1)
            params_str = action_match.group(2)
            params = self._parse_params(params_str)
            action = (tool_name, params)
        
        return thought, action, is_final, final_answer
    
    def _parse_params(self, params_str: str) -> Dict[str, Any]:
        """Parse parameters from action string. Handles strings, numbers, arrays, objects."""
        params = {}
        params_str = params_str.strip()
        
        if not params_str:
            return params
        
        i = 0
        while i < len(params_str):
            # Skip whitespace and commas
            while i < len(params_str) and params_str[i] in ' ,\t':
                i += 1
            if i >= len(params_str):
                break
            
            # Find parameter name
            name_match = re.match(r'(\w+)\s*=\s*', params_str[i:])
            if not name_match:
                break
            
            param_name = name_match.group(1)
            i += name_match.end()
            
            if i >= len(params_str):
                break
            
            char = params_str[i]
            
            if char == '"':
                # Quoted string
                end = params_str.find('"', i + 1)
                if end == -1:
                    end = len(params_str)
                value = params_str[i+1:end]
                i = end + 1
                if value.lower() not in ('none', 'null'):
                    params[param_name] = value
            
            elif char == '[':
                # Array
                depth, j = 1, i + 1
                while j < len(params_str) and depth > 0:
                    if params_str[j] == '[': depth += 1
                    elif params_str[j] == ']': depth -= 1
                    elif params_str[j] == '"':
                        j += 1
                        while j < len(params_str) and params_str[j] != '"':
                            j += 1
                    j += 1
                try:
                    params[param_name] = json.loads(params_str[i:j])
                except:
                    params[param_name] = params_str[i:j]
                i = j
            
            elif char == '{':
                # Object
                depth, j = 1, i + 1
                while j < len(params_str) and depth > 0:
                    if params_str[j] == '{': depth += 1
                    elif params_str[j] == '}': depth -= 1
                    elif params_str[j] == '"':
                        j += 1
                        while j < len(params_str) and params_str[j] != '"':
                            j += 1
                    j += 1
                try:
                    params[param_name] = json.loads(params_str[i:j])
                except:
                    params[param_name] = params_str[i:j]
                i = j
            
            else:
                # Unquoted value
                match = re.match(r'([^,\)\s]+)', params_str[i:])
                if match:
                    value = match.group(1)
                    i += match.end()
                    if value.lower() in ('none', 'null'):
                        continue
                    elif value.lower() == 'true':
                        params[param_name] = True
                    elif value.lower() == 'false':
                        params[param_name] = False
                    elif re.match(r'^-?\d+$', value):
                        params[param_name] = int(value)
                    elif re.match(r'^-?\d+\.\d+$', value):
                        params[param_name] = float(value)
                    else:
                        params[param_name] = value
                else:
                    i += 1
        
        return params
    
    def _execute_action(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Execute a tool action."""
        if tool_name not in self.tools:
            available = ", ".join(list(self.tools.keys())[:5])
            return f"Error: Tool '{tool_name}' not found. Available: {available}"
        
        try:
            tool = self.tools[tool_name]
            result = tool.execute(**params)
            return str(result)[:1000]  # Limit result size
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _format_params(self, params: Dict[str, Any]) -> str:
        """Format params for display."""
        parts = []
        for k, v in params.items():
            if isinstance(v, str):
                parts.append(f'{k}="{v}"')
            else:
                parts.append(f'{k}={v}')
        return ", ".join(parts)
