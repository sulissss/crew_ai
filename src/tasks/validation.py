from textwrap import dedent
from crewai import Task
from src.tasks.note_taker import EXPECTED_OUTPUT_SCHEMA


def create_validation_task(agent, transcript, past_transcript=None, past_response=None):
    description_parts = [
        "Validate the response from the Note-Taker Agent.",
        "Ensure that the Note-Taker Agent provides a 300-word summary.",
        "Remind the Note-Taker Agent to save the response to a local file.",
    ]

    if past_transcript and past_response:
        description_parts.extend([
            f"Past Meeting Transcript: {past_transcript}",
            f"Past Meeting Response: {past_response}",
        ])

    description_parts.append(f"Current Meeting Transcript: {transcript}")

    return Task(
        description="\n".join(description_parts),
        agent=agent,
        expected_output=EXPECTED_OUTPUT_SCHEMA,
    )
