"""
Script to run prompts from a CSV file with Claude Computer Use model via Vertex AI
and save results back to the CSV
"""

import asyncio
import base64
import os
import sys
import json
import signal
import psutil
import csv
import argparse
import socket
import re
import tempfile
import shutil
from pathlib import Path, PosixPath
from typing import cast, get_args, List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import StrEnum

import httpx
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)

# Try relative import first since we're in the same package
try:
    from .loop import APIProvider, sampling_loop
    from .tools import ToolResult, ToolVersion
    print("Successfully imported modules with relative imports")
except ImportError:
    # Fall back to absolute import
    try:
        from computer_use_demo.loop import APIProvider, sampling_loop
        from computer_use_demo.tools import ToolResult, ToolVersion
        print("Successfully imported modules with absolute imports")
    except ImportError:
        print("ERROR: Failed to import required modules. Check your Python path.")
        sys.exit(1)

# Configuration for API providers
PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-7-sonnet-20250219",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-7-sonnet@20250219",
}

@dataclass(kw_only=True, frozen=True)
class ModelConfig:
    tool_version: ToolVersion
    max_output_tokens: int
    default_output_tokens: int
    has_thinking: bool = False

SONNET_3_7 = ModelConfig(
    tool_version="computer_use_20250124",
    max_output_tokens=128_000,
    default_output_tokens=1024 * 16,
    has_thinking=True,
)

MODEL_TO_MODEL_CONF: dict[str, ModelConfig] = {
    "claude-3-7-sonnet-20250219": SONNET_3_7,
}

CONFIG_DIR = PosixPath("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"

# Default shared volume path in container
SHARED_VOLUME_CONTAINER_PATH = "/home/computeruse/shared_data"
CONTAINER_PREFIX = "computer-use-demo-instance-"

class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"

def get_provider() -> APIProvider:
    """Get the API provider from environment variables"""
    provider_str = os.getenv("API_PROVIDER", "vertex").lower()
    if provider_str == "anthropic":
        return APIProvider.ANTHROPIC
    elif provider_str == "bedrock":
        return APIProvider.BEDROCK
    elif provider_str == "vertex":
        return APIProvider.VERTEX
    else:
        print(f"WARNING: Unknown provider '{provider_str}', defaulting to Vertex")
        return APIProvider.VERTEX

def validate_auth(provider: APIProvider) -> Optional[str]:
    """Validate authentication for the given provider"""
    if provider == APIProvider.ANTHROPIC:
        if not os.getenv("ANTHROPIC_API_KEY"):
            return "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
    elif provider == APIProvider.BEDROCK:
        try:
            import boto3
            if not boto3.Session().get_credentials():
                return "AWS credentials not found for Bedrock."
        except ImportError:
            return "boto3 is not installed. Install it with 'pip install boto3'."
    elif provider == APIProvider.VERTEX:
        if not os.environ.get("CLOUD_ML_REGION"):
            return "CLOUD_ML_REGION environment variable not set for Vertex AI."
        try:
            import google.auth
            from google.auth.exceptions import DefaultCredentialsError
            
            try:
                google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            except DefaultCredentialsError:
                return "Google Cloud credentials not properly configured."
        except ImportError:
            return "Google Auth libraries not installed. Install with 'pip install google-auth google-cloud-aiplatform'."
    
    return None

def print_message(sender: Sender, message: str | BetaContentBlockParam | ToolResult):
    """Print a message from either the user, assistant, or tool"""
    print(f"\n=== {sender.upper()} ===")
    
    # Handle different message types
    is_tool_result = not isinstance(message, str | dict)
    
    if is_tool_result:
        message = cast(ToolResult, message)
        if message.output:
            print(f"Output: {message.output}")
        if message.error:
            print(f"Error: {message.error}")
        if message.base64_image:
            print(f"[Image captured - {len(message.base64_image) // 4 * 3} bytes]")
    elif isinstance(message, dict):
        if message["type"] == "text":
            print(message["text"])
        elif message["type"] == "thinking":
            print(f"[Thinking]\n{message.get('thinking', '')}")
        elif message["type"] == "tool_use":
            print(f"Tool Use: {message['name']}\nInput: {message['input']}")
        else:
            print(f"Unknown message type: {message['type']}")
    else:
        print(message)

def track_new_processes():
    """Record all currently running processes"""
    return {p.pid for p in psutil.process_iter()}

def cleanup_processes(old_processes, new_processes):
    """Close all processes that were started between recording old and new processes"""
    started_processes = new_processes - old_processes
    print(f"\n=== CLEANING UP {len(started_processes)} NEW PROCESSES ===")
    
    for pid in started_processes:
        try:
            process = psutil.Process(pid)
            if process.is_running():
                print(f"Terminating process {pid}: {process.name()}")
                process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    # Give processes a moment to terminate gracefully
    if started_processes:
        print("Waiting for processes to terminate...")
        psutil.wait_procs([psutil.Process(pid) for pid in started_processes 
                          if psutil.pid_exists(pid)], timeout=3)
    
    # Force kill any remaining processes
    for pid in started_processes:
        try:
            process = psutil.Process(pid)
            if process.is_running():
                print(f"Force killing process {pid}: {process.name()}")
                process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def tool_output_callback(
    tool_output: ToolResult, tool_id: str, tool_state: dict[str, ToolResult]
):
    """Handle a tool output by storing it to state and printing it."""
    tool_state[tool_id] = tool_output
    print_message(Sender.TOOL, tool_output)

def api_response_callback(
    request: httpx.Request,
    response: httpx.Response | object | None,
    error: Exception | None,
):
    """Handle API responses"""
    if error:
        print(f"ERROR: {error}")

def extract_assistant_response(messages):
    """Extract the final text response from the assistant's message"""
    try:
        # Find the last assistant message
        for message in reversed(messages):
            if message.get("role") == "assistant":
                # Extract the text content
                content = message.get("content", [])
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")
                        
                # If we got here but didn't find text content, look for it differently
                # This handles different message formats that might occur
                if isinstance(content, str):
                    return content
                
                # Last attempt if the above didn't work
                return str(content)
    except Exception as e:
        print(f"Error extracting assistant response: {e}")
    
    return "No response extracted"

async def run_prompt(
    prompt_text: str,
    messages: List[Dict[str, Any]],
    model: str,
    provider: APIProvider,
    tool_version: ToolVersion,
    api_key: str,
    tool_state: Dict[str, ToolResult],
    custom_system_prompt: str = "",
    track_processes: bool = False
) -> Tuple[List[Dict[str, Any]], str]:
    """Run a specific prompt and return the updated messages and final response"""
    
    # Track processes before prompt execution if requested
    if track_processes:
        initial_processes = track_new_processes()
    
    # Add the new prompt to messages
    messages.append({
        "role": Sender.USER,
        "content": [BetaTextBlockParam(type="text", text=prompt_text)],
    })
    
    # Print the user's message
    print_message(Sender.USER, prompt_text)
    
    # Run the sampling loop
    try:
        updated_messages = await sampling_loop(
            system_prompt_suffix=custom_system_prompt,
            model=model,
            provider=provider,
            messages=messages,
            output_callback=lambda msg: print_message(Sender.BOT, msg),
            tool_output_callback=lambda tool_output, tool_id: tool_output_callback(tool_output, tool_id, tool_state),
            api_response_callback=api_response_callback,
            api_key=api_key,
            only_n_most_recent_images=3,
            tool_version=tool_version,
            max_tokens=SONNET_3_7.default_output_tokens,
        )
        
        # Extract the assistant's final response
        assistant_response = extract_assistant_response(updated_messages)
        
        # If we're tracking processes, get the new set and return both
        if track_processes:
            final_processes = track_new_processes()
            return updated_messages, assistant_response, initial_processes, final_processes
        
        return updated_messages, assistant_response
    except Exception as e:
        print(f"ERROR during sampling loop: {e}")
        import traceback
        traceback.print_exc()
        
        error_response = f"Error: {str(e)}"
        
        # If we're tracking processes, still return the process sets
        if track_processes:
            final_processes = track_new_processes()
            return messages, error_response, initial_processes, final_processes
        
        return messages, error_response

def get_container_name():
    """Try to get the container name"""
    try:
        # Try to get hostname first
        hostname = socket.gethostname()
        
        # Check if the hostname is a container ID (starts with e505bec54cd6 format)
        if re.match(r'^[0-9a-f]{12}$', hostname):
            # This is a container ID, so we need to find the corresponding CSV
            
            # Use environment variable if set
            container_instance = os.environ.get('CONTAINER_INSTANCE_NUM')
            if container_instance:
                return f"{CONTAINER_PREFIX}{container_instance}"
            
            # Try to extract instance number from Docker environment
            try:
                with open('/proc/self/cgroup', 'r') as f:
                    for line in f:
                        if 'docker' in line:
                            # Look for container name in cgroup info
                            match = re.search(r'instance-([0-9]+)', line)
                            if match:
                                instance_num = match.group(1)
                                return f"{CONTAINER_PREFIX}{instance_num}"
            except:
                pass
        
        # If the hostname already has our prefix format, use it directly
        if hostname.startswith(CONTAINER_PREFIX):
            return hostname
        
        # Fallback - scan shared directory for CSVs
        return None
    except:
        return None

def find_csv_file(shared_dir):
    """Find suitable CSV file in shared directory"""
    try:
        # Get container specific info
        container_name = get_container_name()
        
        # Try hostname-based CSV first
        if container_name:
            hostname_csv = Path(shared_dir) / f"{container_name}.csv"
            if hostname_csv.exists():
                return str(hostname_csv)
        
        # Try to find any instance CSV that exists
        # First check if there's an environment variable with the instance number
        instance_num = os.environ.get('CONTAINER_INSTANCE_NUM')
        if instance_num:
            instance_csv = Path(shared_dir) / f"{CONTAINER_PREFIX}{instance_num}.csv"
            if instance_csv.exists():
                return str(instance_csv)
        
        # Scan the directory for CSVs
        csv_files = list(Path(shared_dir).glob(f"{CONTAINER_PREFIX}*.csv"))
        
        # Try to find one that's not being used
        # This is a simplistic approach - in real implementation you might 
        # want to use a locking mechanism
        for csv_file in csv_files:
            # Check if we can get exclusive access
            try:
                with open(csv_file, 'r+') as f:
                    # If we can open it, return this file
                    return str(csv_file)
            except:
                # If file is locked or inaccessible, try next one
                continue
        
        # If we get here, try any prompts.csv as a fallback
        prompts_csv = Path(shared_dir) / "prompts.csv"
        if prompts_csv.exists():
            return str(prompts_csv)
            
        # Last resort - just return the first CSV we find
        all_csv_files = list(Path(shared_dir).glob("*.csv"))
        if all_csv_files:
            return str(all_csv_files[0])
            
    except Exception as e:
        print(f"Error finding CSV file: {e}")
    
    # If we can't find anything, return None
    return None

def read_prompts_from_csv(csv_path):
    """Read prompts and their row indices from a CSV file"""
    prompts = []
    has_header = False
    
    if not csv_path:
        return prompts, has_header
        
    try:
        with open(csv_path, 'r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            
            # First, check if there's a header
            try:
                header = next(reader)
                if len(header) > 0 and "prompt" in header[0].lower():
                    has_header = True
                else:
                    # This was actually data, add it to prompts with row index 0
                    prompts.append((0, header[0]))
            except StopIteration:
                # Empty file
                pass
                
            # Read rest of prompts with their row indices
            for i, row in enumerate(reader, start=0 if not has_header else 1):
                if row and len(row) > 0 and row[0].strip():
                    prompts.append((i, row[0]))
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    
    return prompts, has_header

def save_result_to_csv(csv_path, row_index, prompt, result, has_header):
    """Save a result back to the CSV file"""
    try:
        # Read all rows from the CSV
        rows = []
        with open(csv_path, 'r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            rows = list(reader)
        
        # Determine if we need to add the result column header
        if has_header:
            if len(rows[0]) < 2:
                rows[0].append("result")
        
        # Make sure the row exists and has enough columns
        while len(rows) <= row_index:
            rows.append([])
        
        if len(rows[row_index]) < 1:
            rows[row_index].append(prompt)
            
        # Add or update the result column
        if len(rows[row_index]) < 2:
            rows[row_index].append(result)
        else:
            rows[row_index][1] = result
            
        # Write back to the CSV file using a temporary file for safety
        with tempfile.NamedTemporaryFile('w', newline='', delete=False) as temp_file:
            writer = csv.writer(temp_file)
            writer.writerows(rows)
            
        # Replace the original file with the updated one
        shutil.move(temp_file.name, csv_path)
        print(f"Successfully saved result for row {row_index} to {csv_path}")
        return True
        
    except Exception as e:
        print(f"Error saving result to CSV: {e}")
        return False

async def main():
    """Main function to run prompts from a CSV file and save results back"""
    global CONTAINER_PREFIX
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Claude prompts from a CSV file')
    parser.add_argument('--csv', type=str, help='Path to CSV file containing prompts')
    parser.add_argument('--shared-dir', type=str, default=SHARED_VOLUME_CONTAINER_PATH,
                      help=f'Path to shared directory (default: {SHARED_VOLUME_CONTAINER_PATH})')
    parser.add_argument('--container-prefix', type=str, default=CONTAINER_PREFIX,
                      help=f'Container name prefix (default: {CONTAINER_PREFIX})')
    parser.add_argument('--instance-num', type=str, help='Container instance number')
    parser.add_argument('--mark-completed', action='store_true', 
                      help='Create a .completed file when done to signal completion')
    args = parser.parse_args()
    
    # Override globals if provided
    
    if args.container_prefix:
        CONTAINER_PREFIX = args.container_prefix
    
    # Set instance number environment variable if provided
    if args.instance_num:
        os.environ['CONTAINER_INSTANCE_NUM'] = args.instance_num
    
    # Determine the CSV file path
    csv_path = None
    if args.csv:
        # Specific CSV path provided
        csv_path = args.csv
    else:
        # Try to find a suitable CSV file
        csv_path = find_csv_file(args.shared_dir)
    
    # Make CSV path absolute if needed
    if csv_path and not Path(csv_path).is_absolute():
        csv_path = str(Path(args.shared_dir) / csv_path)
    
    if csv_path:
        print(f"Looking for prompts in: {csv_path}")
    else:
        print("No suitable CSV file found.")
    
    # Read prompts from CSV file
    prompts_with_indices, has_header = read_prompts_from_csv(csv_path)
    
    # If no prompts found, use default prompts as fallback
    if not prompts_with_indices:
        print(f"Warning: No prompts found in CSV. Using default prompts.")
        default_prompts = [
            "Please open a terminal and run 'ls -la'. Then explain the output.",
            "Create a simple text file named 'hello.txt' with the content 'Hello, World!' and then display its contents."
        ]
        prompts_with_indices = [(i, prompt) for i, prompt in enumerate(default_prompts)]
        has_header = False
    
    print(f"Found {len(prompts_with_indices)} prompts to run.")
    
    # Get API provider from environment
    provider = get_provider()
    print(f"Using API provider: {provider}")
    
    # Validate authentication
    auth_error = validate_auth(provider)
    if auth_error:
        print(f"Authentication error: {auth_error}")
        sys.exit(1)
    
    # Set API key - only needed for Anthropic, can be empty for others
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Get model name for provider
    model = PROVIDER_TO_DEFAULT_MODEL_NAME[provider]
    print(f"Using model: {model}")
    
    # Get tool version
    tool_version = SONNET_3_7.tool_version
    print(f"Using tool version: {tool_version}")
    
    # Optional custom system prompt
    custom_system_prompt = os.getenv("SYSTEM_PROMPT", "")
    
    # Dictionary to store tool results
    tool_state = {}
    
    # Initialize empty messages list
    messages = []
    
    # Track completion status
    all_completed = True
    
    # Run each prompt with process tracking for the first prompt
    for idx, (row_index, prompt) in enumerate(prompts_with_indices):
        print(f"\n\n======= RUNNING PROMPT {idx+1}/{len(prompts_with_indices)} =======")
        print(f"CSV Row Index: {row_index}")
        
        # Track processes only for the first prompt
        track_processes_for_this_prompt = (idx == 0)
        
        try:
            if track_processes_for_this_prompt:
                result = await run_prompt(
                    prompt, 
                    messages, 
                    model, 
                    provider, 
                    tool_version, 
                    api_key, 
                    tool_state,
                    custom_system_prompt,
                    track_processes=True
                )
                messages, assistant_response, initial_processes, final_processes = result
                
                # Clean up processes started during this prompt
                print("\n\n======= CLEANING UP PROMPT PROCESSES =======")
                cleanup_processes(initial_processes, final_processes)
            else:
                messages, assistant_response = await run_prompt(
                    prompt, 
                    messages, 
                    model, 
                    provider, 
                    tool_version, 
                    api_key, 
                    tool_state,
                    custom_system_prompt
                )
            
            # Save the result back to the CSV
            if csv_path:
                success = save_result_to_csv(csv_path, row_index, prompt, assistant_response, has_header)
                if not success:
                    all_completed = False
                    print(f"Failed to save result for prompt {idx+1} (row {row_index})")
            
        except Exception as e:
            print(f"Error running prompt {idx+1}: {e}")
            all_completed = False
            
            # Try to save the error as the result
            if csv_path:
                error_msg = f"Error running prompt: {str(e)}"
                save_result_to_csv(csv_path, row_index, prompt, error_msg, has_header)
        
        # Clear history between prompts
        print("\n\n======= CLEARING HISTORY =======")
        messages = []
        tool_state = {}
    
    print("\n\n======= ALL PROMPTS COMPLETED =======")
    
    # Create a completion marker file if requested
    if args.mark_completed and csv_path and all_completed:
        completed_marker = f"{csv_path}.completed"
        try:
            with open(completed_marker, 'w') as f:
                f.write(f"Completed processing {len(prompts_with_indices)} prompts at {__import__('datetime').datetime.now()}")
            print(f"Created completion marker file: {completed_marker}")
        except Exception as e:
            print(f"Failed to create completion marker: {e}")

if __name__ == "__main__":
    asyncio.run(main())