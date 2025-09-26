# server.py
from mcp.server.fastmcp import FastMCP
from langmem import create_prompt_optimizer
from typing import List, Dict, Optional, Any
import json
import datetime
import os

mcp = FastMCP("Demo 🚀")

# Create prompt optimizer with OpenAI
def get_prompt_optimizer(api_key: str):
    """Initialize prompt optimizer with provided OpenAI API key"""
    try:
        if not api_key or not api_key.strip():
            return None
            
        # Validate API key format
        if not api_key.startswith("sk-"):
            print(f"Invalid API key format: {api_key[:10]}...")
            return None
            
        # Temporarily set environment variable for this process only
        # This won't affect other users since each MCP server runs in its own process
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = api_key.strip()
        
        try:
            return create_prompt_optimizer(
                "openai:gpt-4o-mini", 
                kind="prompt_memory"
            )
        finally:
            # Restore original environment variable state
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)
    except Exception as e:
        print(f"Error initializing prompt optimizer: {e}")
        print(f"API key format: {api_key[:10] if api_key else 'None'}...")
        return None

# Storage for optimization history - use persistent file storage
import json
from pathlib import Path

OPTIMIZATION_FILE = Path("tmp/optimization_history.json")

def load_optimization_history() -> List[Dict[str, Any]]:
    """Load optimization history from file"""
    try:
        if OPTIMIZATION_FILE.exists():
            with open(OPTIMIZATION_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading optimization history: {e}")
    return []

def save_optimization_history(history: List[Dict[str, Any]]) -> None:
    """Save optimization history to file"""
    try:
        OPTIMIZATION_FILE.parent.mkdir(exist_ok=True)
        with open(OPTIMIZATION_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving optimization history: {e}")

# Load existing optimization history
optimization_history = load_optimization_history()

@mcp.tool()
async def optimize_prompt(
    base_prompt: str,
    user_question: str,
    assistant_response: str,
    feedback: str,
    openai_api_key: str
) -> str:
    """Optimize a prompt based on conversation and feedback"""
    prompt_optimizer = get_prompt_optimizer(openai_api_key)
    
    if not prompt_optimizer:
        return "Error: Invalid or missing OpenAI API key provided."
    
    conversation = [
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": assistant_response}
    ]
    
    trajectories = [(conversation, {"feedback": feedback})]
    
    try:
        better_prompt = await prompt_optimizer(trajectories, base_prompt)
        
        # Store optimization
        optimization_entry = {
            "id": len(optimization_history),
            "original_prompt": base_prompt,
            "optimized_prompt": better_prompt,
            "feedback": feedback,
            "conversation": conversation,
            "timestamp": datetime.datetime.now().isoformat()
        }
        optimization_history.append(optimization_entry)
        
        # Save to persistent storage
        save_optimization_history(optimization_history)
        
        return f"✅ Optimization Complete!\n\nOriginal: {base_prompt}\n\nOptimized: {better_prompt}\n\nBased on feedback: {feedback}"
        
    except Exception as e:
        error_msg = str(e)
        if "api" in error_msg.lower() and "key" in error_msg.lower():
            return f"❌ OpenAI API Key Error: {error_msg}\n\nPlease verify your API key is correct and has sufficient credits."
        elif "unauthorized" in error_msg.lower() or "401" in error_msg:
            return f"❌ Authentication Error: Invalid OpenAI API key.\n\nPlease check that your API key is correct and active."
        elif "quota" in error_msg.lower() or "429" in error_msg:
            return f"❌ Quota Error: {error_msg}\n\nYour OpenAI account may have exceeded its quota or rate limit."
        else:
            return f"❌ Optimization Error: {error_msg}"

@mcp.tool()
async def batch_optimize_prompt(
    base_prompt: str,
    feedback_json: str,
    openai_api_key: str
) -> str:
    """Optimize prompt with multiple feedback examples (JSON array format)"""
    prompt_optimizer = get_prompt_optimizer(openai_api_key)
    
    if not prompt_optimizer:
        return "Error: Invalid or missing OpenAI API key provided."
    
    try:
        feedback_list = json.loads(feedback_json)
        
        trajectories = []
        for item in feedback_list:
            conversation = [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["response"]}
            ]
            trajectories.append((conversation, {"feedback": item["feedback"]}))
        
        better_prompt = await prompt_optimizer(trajectories, base_prompt)
        
        # Store batch optimization
        optimization_entry = {
            "id": len(optimization_history),
            "original_prompt": base_prompt,
            "optimized_prompt": better_prompt,
            "batch_feedback": feedback_list,
            "type": "batch",
            "timestamp": datetime.datetime.now().isoformat()
        }
        optimization_history.append(optimization_entry)
        
        return f"Batch optimized with {len(feedback_list)} examples:\n\nOriginal: {base_prompt}\n\nOptimized: {better_prompt}"
        
    except json.JSONDecodeError:
        return "Invalid JSON format for feedback data"
    except Exception as e:
        return f"Error in batch optimization: {str(e)}"

@mcp.tool()
def get_optimization_history(limit: int = 5) -> str:
    """Get recent prompt optimization history"""
    if not optimization_history:
        return "No optimizations performed yet"
    
    recent = optimization_history[-limit:]
    result = f"Recent optimizations ({len(recent)}):\n\n"
    
    for i, opt in enumerate(recent, 1):
        result += f"{i}. Optimization {opt['id']}\n"
        result += f"   Type: {'Batch' if opt.get('type') == 'batch' else 'Single'}\n"
        result += f"   Original: {opt['original_prompt']}\n"
        result += f"   Optimized: {opt['optimized_prompt']}\n"
        
        if opt.get('type') != 'batch':
            result += f"   Feedback: {opt['feedback']}\n"
        else:
            result += f"   Examples: {len(opt['batch_feedback'])}\n"
        
        result += f"   Time: {opt['timestamp']}\n\n"
    
    return result

@mcp.tool()
def get_latest_optimized_prompt() -> str:
    """Get the most recent optimized prompt for use in conversations"""
    # Reload from file to get latest optimizations
    global optimization_history
    optimization_history = load_optimization_history()
    
    if not optimization_history:
        return "You are a helpful AI assistant"
    
    latest = optimization_history[-1]
    return latest.get("optimized_prompt", "You are a helpful AI assistant")

@mcp.tool()
def clear_optimizations() -> str:
    """Clear all optimization history"""
    global optimization_history
    count = len(optimization_history)
    optimization_history.clear()
    
    # Clear persistent storage
    save_optimization_history(optimization_history)
    
    return f"Cleared {count} optimization records"

if __name__ == "__main__":
    mcp.run()