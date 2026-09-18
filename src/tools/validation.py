import json
import os
from langchain.agents import tool

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def validate_output_schema(output):
    expected_keys = ["agenda", "summary", "discussion"]
    if not all(key in output for key in expected_keys):
        return False
    if not isinstance(output["discussion"], list):
        return False
    discussion_keys = ["discussion_point", "person_responsible", "completion_date", "remarks"]
    for item in output["discussion"]:
        if not all(key in item for key in discussion_keys):
            return False
    return True


@tool
def validate_and_save(output: str):
    """Validate the Note-Taker Agent's JSON response and save it if valid. Returns an error if the format is invalid."""
    parsed = json.loads(output)
    if validate_output_schema(parsed):
        filepath = os.path.join(OUTPUT_DIR, "validated_response.json")
        with open(filepath, "w") as f:
            json.dump(parsed, f, indent=4)
        return f"Output is valid and saved to '{filepath}'"
    return "Output is invalid. Ensure it contains 'agenda', 'summary', and 'discussion' fields with the correct structure."
