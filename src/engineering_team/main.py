# src/engineering_team/main.py

from pathlib import Path
import sys
import warnings
import os
import time
from datetime import datetime
import json
from typing import Dict, Any

from engineering_team.crew import EngineeringTeam
from engineering_team.ado_client import (
    post_crew_completed, post_execution_started, post_comment,
    post_crew_initializing, post_team_initializing, post_workflow_started
)
from engineering_team.token_tracker import EnhancedTokenUsageCallbackHandler

 # Use accurate pricing based on actual provider rates
PRICING = {
    "openai/gpt-4o": {"in": 2.50, "out": 10.00},  # $2.50 per 1M input, $10.00 per 1M output
    "openai/gpt-4o-mini": {"in": 0.15, "out": 0.60},  # $0.15 per 1M input, $0.60 per 1M output
    "deepseek/deepseek-chat": {"in": 0.14, "out": 0.28}  # $0.14 per 1M input, $0.28 per 1M output
}
# Use available cost tracking methods for LiteLLM v1.72.0
import litellm
from litellm import cost_calculator
litellm.success_callback = ["azure"]  # or any callback to enable tracking
litellm.failure_callback = ["azure"]
litellm._current_cost = 0  # Initialize

# Enable verbose mode for better debugging
os.environ['LITELLM_LOG'] = 'DEBUG'

# completion_cost_tracker not available in this version, using _current_cost instead
import logging


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

requirements = """
A simple account management system for a trading simulation platform.
The system should allow users to create an account, deposit funds, and withdraw funds.
The system should allow users to record that they have bought or sold shares, providing a quantity.
The system should calculate the total value of the user's portfolio, and the profit or loss from the initial deposit.
The system should be able to report the holdings of the user at any point in time.
The system should be able to report the profit or loss of the user at any point in time.
The system should be able to list the transactions that the user has made over time.
The system should prevent the user from withdrawing funds that would leave them with a negative balance, or
 from buying more shares than they can afford, or selling shares that they don't have.
 The system has access to a function get_share_price(symbol) which returns the current price of a share, and includes a test implementation that returns fixed prices for AAPL, TSLA, GOOGL.
 The system UI should be bilingual in English and Chinese.
"""
module_name = "accounts.py"
class_name = "Account"

# Agent to LLM mapping based on your configuration
AGENT_LLM_MAPPING = {
    # "Engineering Lead": "deepseek-chat",
    # "Backend Engineer": "deepseek-chat", 
    # "Frontend Engineer": "deepseek-chat",
    # "Test Engineer": "deepseek-chat"
    "Engineering Lead": "gpt-4o",
    "Backend Engineer": "deepseek-chat", 
    "Frontend Engineer": "deepseek-chat",
    "Test Engineer": "gpt-4o-mini"
}

# Map agent descriptions to proper role names
AGENT_DESCRIPTION_TO_ROLE = {
    "Engineering Lead for the engineering team, directing the work of the engineer\n": "Engineering Lead",
    "Python Engineer who can write code to achieve the design described by the engineering lead\n": "Backend Engineer",
    "A Gradio expert to who can write a simple frontend to demonstrate a backend\n": "Frontend Engineer",
    "An engineer with Python coding skills who can write unit tests for the given backend module {module_name}\n": "Test Engineer"
}

# Global token usage tracker
GLOBAL_TOKEN_USAGE = {}

def _monkey_patch_litellm():
    """Monkey patch LiteLLM to capture token usage per model"""
    original_completion = litellm.completion
    
    def patched_completion(*args, **kwargs):
        # Get model name from args or kwargs
        model = kwargs.get('model', None)
        if not model and len(args) > 0:
            model = args[0] if isinstance(args[0], str) else None
        
        # Call the original function
        result = original_completion(*args, **kwargs)
        
        # Extract token usage from response
        if hasattr(result, 'usage') and result.usage:
            usage_data = {
                'input_tokens': getattr(result.usage, 'prompt_tokens', 0),
                'output_tokens': getattr(result.usage, 'completion_tokens', 0),
                'total_tokens': getattr(result.usage, 'total_tokens', 0)
            }
            
            # Store usage by model
            if model:
                if model not in GLOBAL_TOKEN_USAGE:
                    GLOBAL_TOKEN_USAGE[model] = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
                
                GLOBAL_TOKEN_USAGE[model]['input_tokens'] += usage_data['input_tokens']
                GLOBAL_TOKEN_USAGE[model]['output_tokens'] += usage_data['output_tokens']
                GLOBAL_TOKEN_USAGE[model]['total_tokens'] += usage_data['total_tokens']
                
                print(f"🔍 DEBUG: Captured token usage for {model}: {usage_data}", flush=True)
        
        return result
    
    # Replace the completion function
    litellm.completion = patched_completion
    print("✅ Monkey-patched LiteLLM to capture token usage", flush=True)

def _setup_agent_with_tracking(agent, token_tracker):
    """Setup agent with token tracking based on agent role"""
    agent_role = getattr(agent, 'role', 'Unknown')
    expected_llm = AGENT_LLM_MAPPING.get(agent_role, 'unknown')
    
    # Set current agent for token attribution using proper role name
    token_tracker.set_current_agent(agent_role)
    print(f"🔍 DEBUG: Setting up tracking for {agent_role}", flush=True)

    # Add callback to agent's LLM if it exists
    if hasattr(agent, 'llm') and agent.llm:
        print(f"🔍 DEBUG: Agent {agent_role} has LLM: {type(agent.llm)}", flush=True)

        if not hasattr(agent.llm, 'callbacks'):
            agent.llm.callbacks = []
            print(f"🔍 DEBUG: Created callbacks list for {agent_role}", flush=True)
        
        # Initialize callbacks if None
        if agent.llm.callbacks is None:
            agent.llm.callbacks = []
            print(f"🔍 DEBUG: Initialized callbacks list for {agent_role}", flush=True)
        
        # Check if token_tracker is already in callbacks to avoid duplicates
        if token_tracker not in agent.llm.callbacks:
            agent.llm.callbacks.append(token_tracker)
            print(f"🔍 DEBUG: Added token tracker to {agent_role}'s LLM. Callbacks: {len(agent.llm.callbacks)}", flush=True)
        else:
            print(f"🔍 DEBUG: Token tracker already in {agent_role}'s callbacks", flush=True)

    else:
        print(f"🔍 DEBUG: Agent {agent_role} has no LLM or callbacks", flush=True)
    return agent

def _process_token_usage_data(raw_usage_data):
    """Process raw token usage data to use proper role names and models"""
    processed_data = {}
    
    for agent_desc, usage_info in raw_usage_data.items():
        # Convert agent description to proper role name
        role_name = AGENT_DESCRIPTION_TO_ROLE.get(agent_desc, agent_desc)
        
        # Get the correct model for this role
        model = AGENT_LLM_MAPPING.get(role_name, "unknown")
        
        # Update the usage data with proper model information
        if isinstance(usage_info, dict) and 'unknown' in usage_info:
            # Replace 'unknown' with actual model
            actual_usage = usage_info['unknown']
            processed_data[role_name] = {
                model: {
                    'input_tokens': actual_usage.get('input_tokens', 0),
                    'output_tokens': actual_usage.get('output_tokens', 0)
                }
            }
        else:
            processed_data[role_name] = usage_info
    
    return processed_data

def _distribute_global_token_usage_by_agent():
    """Distribute global token usage to agents based on expected patterns"""
    distributed_usage = {}
    
    if not GLOBAL_TOKEN_USAGE:
        return {}
    
    # Expected distribution patterns based on agent roles and typical usage
    distribution_patterns = {
        "Engineering Lead": {
            "openai/gpt-4o": {"input_ratio": 0.8, "output_ratio": 0.7}  # Mostly uses GPT-4o
        #    "deepseek/deepseek-chat": {"input_ratio": 0.8, "output_ratio": 0.7}  # Mostly uses GPT-4o
        },
        "Backend Engineer": {
            "deepseek/deepseek-chat": {"input_ratio": 0.6, "output_ratio": 0.7}  # Mostly backend work
        },
        "Frontend Engineer": {
            "deepseek/deepseek-chat": {"input_ratio": 0.4, "output_ratio": 0.3}  # Some frontend work
        },
        "Test Engineer": {
        #    "deepseek/deepseek-chat": {"input_ratio": 1.0, "output_ratio": 1.0}  # All test work
            "openai/gpt-4o-mini": {"input_ratio": 1.0, "output_ratio": 1.0}  # All test work
        }
    }
    
    for agent, patterns in distribution_patterns.items():
        agent_models = {}
        
        for model, ratios in patterns.items():
            if model in GLOBAL_TOKEN_USAGE:
                input_tokens = int(GLOBAL_TOKEN_USAGE[model]['input_tokens'] * ratios['input_ratio'])
                output_tokens = int(GLOBAL_TOKEN_USAGE[model]['output_tokens'] * ratios['output_ratio'])
                
                agent_models[model] = {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens
                }
        
        if agent_models:
            distributed_usage[agent] = agent_models
    
    return distributed_usage

def run():
    print("🔍 DEBUG: run() function started!", flush=True)
    
    inputs = {
        'requirements': requirements,
        'module_name': module_name,
        'class_name': class_name
    }
    
    # Initialize enhanced token tracker
    token_tracker = EnhancedTokenUsageCallbackHandler()
    
    # Start timing
    start_time = time.time()
    
    # Get work item ID from environment
    work_item_id = os.getenv("ADO_WORK_ITEM_ID")
    
    # Post ALL status updates (NEW CODE)
    if work_item_id:
        try:
            post_crew_initializing(int(work_item_id))
            post_team_initializing(int(work_item_id))
            post_workflow_started(int(work_item_id))
            post_execution_started(int(work_item_id), f"Generating {module_name} with requirements")
        except Exception as e:
            print(f"Failed to post status comments to ADO: {e}")
    
    try:
        # Monkey patch LiteLLM to capture token usage
        _monkey_patch_litellm()
        
        # Create crew
        team = EngineeringTeam()
        crew = team.crew()
        
        # Setup all agents with token tracking
        for agent in crew.agents:
            _setup_agent_with_tracking(agent, token_tracker)
        
        # Reset cost tracking before starting
        litellm._current_cost = 0
        print("🔍 DEBUG: Reset LiteLLM cost to 0", flush=True)

        # Run the crew
        result = crew.kickoff(inputs=inputs)
        
        # Get the total cost after execution
        total_cost_after = litellm._current_cost
        print(f"🔍 DEBUG: Cost after execution: ${total_cost_after}", flush=True)
        
        # Process token usage data to use proper role names and models
        processed_usage_data = {}
        if hasattr(token_tracker, 'usage_data'):
            processed_usage_data = _process_token_usage_data(token_tracker.usage_data)
        
        # === DEBUG LOGGING ADDED HERE ===
        print(f"\n=== DEBUG TOKEN USAGE ===", flush=True)
        print(f"LiteLLM total cost: {litellm._current_cost}", flush=True)
        print(f"Global token usage: {GLOBAL_TOKEN_USAGE}", flush=True)
        print(f"Raw token tracker usage data: {token_tracker.usage_data if hasattr(token_tracker, 'usage_data') else 'No usage_data'}", flush=True)
        print(f"Processed usage data: {processed_usage_data}", flush=True)
        print("=== END DEBUG ===\n", flush=True)
        # === END DEBUG LOGGING ===

    except Exception as e:
        print(f"Error during crew execution: {e}")
        result = str(e)
        # Post error to ADO if work item exists
        if work_item_id:
            try:
                post_comment(int(work_item_id), f"❌ Execution failed: {str(e)}")
            except:
                pass
    
    # Calculate stats
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    agent_stats = {
        # "Engineering Lead": "deepseek/deepseek-chat",
        # "Backend Engineer": "deepseek/deepseek-chat",
        # "Frontend Engineer": "deepseek/deepseek-chat",
        # "Test Engineer": "deepseek/deepseek-chat"
        "Engineering Lead": "openai/gpt-4o",
        "Backend Engineer": "deepseek/deepseek-chat",
        "Frontend Engineer": "deepseek/deepseek-chat",
        "Test Engineer": "openai/gpt-4o-mini"
    }

    header_lines = [
        "#" * 80,
        f"# 📦 Generated by AI Crew",
        f"# 🕒 Created: {timestamp}",
        f"# 👥 Crew Members: {len(agent_stats)} ({', '.join(agent_stats.keys())})",
        "# 🤖 LLMs Used:"
    ] + [f"#   - {role} → {model}" for role, model in agent_stats.items()] + [
        f"# ⏱ Duration: {duration} seconds",
        "#" * 80,
        ""
    ]
    
    # Inject header into generated module
    output_file = os.path.join("generated", module_name)
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.readlines()

        if content and "This is a simple demonstration" in content[0]:
            content[0] = "\n".join(header_lines) + "\n"
        else:
            content = header_lines + content

        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(content)

    # Force Gradio app.py to use 0.0.0.0:7860
    if module_name == "app.py" and os.path.exists(output_file):
        with open(output_file, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            f.seek(0)
            new_lines = []
            for line in lines:
                if ".launch(" in line:
                    line = '    interface.launch(server_name="0.0.0.0", server_port=7860)\n'
                new_lines.append(line)
            f.writelines(new_lines)
    
    # Post completion comment to ADO with cost information 
    if work_item_id:
        try:
            print("🔍 DEBUG: Starting cost calculation...", flush=True)
            
            # FIXED: Use proper app URL format
            weave_url = f"https://wandb.ai/run?time={int(start_time)}&duration={duration}"
            app_url = "http://8.219.119.18:7860"  # Fixed app URL format

            llms_used = set()
            by_agent = {}
            by_llm = {}
            total_cost = 0.0

            # Check global token usage first (most reliable)
            if GLOBAL_TOKEN_USAGE:
                print("🔍 DEBUG: Using global token usage data", flush=True)
                print(f"🔍 DEBUG: Global token usage: {GLOBAL_TOKEN_USAGE}", flush=True)
                
                # Distribute token usage to agents
                distributed_usage = _distribute_global_token_usage_by_agent()
                
                for agent, agent_data in distributed_usage.items():
                    agent_total = 0.0
                    agent_models = {}
                    
                    for model, usage in agent_data.items():
                        input_tokens = float(usage.get('input_tokens', 0))
                        output_tokens = float(usage.get('output_tokens', 0))
                        
                        # Convert pricing to per-token cost (divide by 1,000,000 since prices are per 1M tokens)
                        rate = PRICING.get(model, {"in": 0.0, "out": 0.0})
                        cost = (input_tokens / 1000000) * rate["in"] + (output_tokens / 1000000) * rate["out"]
                        cost = round(cost, 6)  # Round to 6 decimal places for precision
                        
                        print(f"🔍 DEBUG: Agent {agent} - Model: {model}, Input: {input_tokens}, Output: {output_tokens}, Cost: ${cost}", flush=True)
                        
                        agent_total += cost
                        agent_models[model] = cost
                        by_llm[model] = by_llm.get(model, 0.0) + cost
                        total_cost += cost
                        llms_used.add(model)
                    
                    if agent_total > 0:
                        by_agent[agent] = {"total": round(agent_total, 6), "models": agent_models}
            
            # If no global token usage, use LiteLLM cost
            elif litellm._current_cost > 0:
                print("🔍 DEBUG: Using LiteLLM cost data", flush=True)
                total_cost = litellm._current_cost
                
                # Distribute the total cost based on expected agent usage
                expected_distribution = {
                    "Engineering Lead": 0.2,
                    "Backend Engineer": 0.4,
                    "Frontend Engineer": 0.3,
                    "Test Engineer": 0.1
                }
                
                for agent, percentage in expected_distribution.items():
                    agent_cost = total_cost * percentage
                    model = AGENT_LLM_MAPPING.get(agent, "unknown")
                    by_agent[agent] = {
                        "total": agent_cost,
                        "models": {model: agent_cost}
                    }
                    by_llm[model] = by_llm.get(model, 0.0) + agent_cost
                    llms_used.add(model)
            
            # Fall back to time-based estimation
            else:
                print("🔍 DEBUG: Using time-based estimation", flush=True)
                time_based_costs = {
                    "Engineering Lead": {"gpt-4o": 0.015 * (duration / 60)},
                    # "Engineering Lead": {"deepseek-chat": 0.015 * (duration / 60)},
                    "Backend Engineer": {"deepseek-chat": 0.008 * (duration / 60)},
                    "Frontend Engineer": {"deepseek-chat": 0.006 * (duration / 60)},
                    # "Test Engineer": {"deepseek-chat": 0.004 * (duration / 60)}
                    "Test Engineer": {"gpt-4o-mini": 0.004 * (duration / 60)}
                }
                
                for agent, models in time_based_costs.items():
                    agent_total = sum(models.values())
                    by_agent[agent] = {"total": agent_total, "models": models}
                    for model, cost in models.items():
                        by_llm[model] = by_llm.get(model, 0.0) + cost
                        total_cost += cost
                        llms_used.add(model)

            # Round total cost for display
            total_cost = round(total_cost, 6)
            
            post_crew_completed(
                work_item_id=int(work_item_id),
                total_time=duration,
                llms_used=sorted(llms_used),
                by_agent=by_agent,
                by_llm=by_llm,
                total_cost=total_cost,
                weave_url=weave_url,
                app_url=app_url
            )
            print(f"✅ Successfully posted cost data to ADO work item {work_item_id}")

        except Exception as e:
            print(f"❌ Failed to post cost info to ADO: {e}")
            import traceback
            traceback.print_exc()
            try:
                post_comment(int(work_item_id), f"✅ Work completed successfully! Total time: {duration}s")
            except Exception as fallback_error:
                print(f"Failed to post fallback comment: {fallback_error}")

    # Print token usage for debugging
    print("\nGlobal Token Usage:")
    for model, usage in GLOBAL_TOKEN_USAGE.items():
        print(f"  {model}: {usage}")
    
    return result

def run_crew(prompt: str):
    team = EngineeringTeam()
    crew = team.crew()
    
    # Add token tracking to run_crew function as well
    token_tracker = EnhancedTokenUsageCallbackHandler()
    
    # Setup agents with tracking
    for agent in crew.agents:
        _setup_agent_with_tracking(agent, token_tracker)
    
    # Reset cost tracking before starting
    litellm._current_cost = 0
    print("🔍 DEBUG: Reset LiteLLM cost to 0", flush=True)
    
    # Run the crew - use prompt parameter
    result = crew.kickoff(inputs={"objective": prompt})
    
    # Log token usage
    print("Token usage statistics:")
    for agent, usage in token_tracker.usage_data.items():
        print(f"  {agent}: {usage}")
    
    return result

if __name__ == "__main__":
    print("🚀 Script starting...", flush=True)
    run()
    print("✅ Script completed", flush=True)