import os
from langchain.agents import tool


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@tool
def save_to_file(content: str):
    """Save text content to a local file."""
    filepath = os.path.join(OUTPUT_DIR, "agent_output.txt")
    with open(filepath, "w") as f:
        f.write(content)
    return f"File saved to {filepath}"


@tool
def save_json_to_file(content: str):
    """Save JSON content to a local file."""
    filepath = os.path.join(OUTPUT_DIR, "agent_output.json")
    with open(filepath, "w") as f:
        f.write(content)
    return f"File saved to {filepath}"


@tool
def check_file(file_name: str):
    """Verify if a file exists on the system."""
    return os.path.exists(file_name)
