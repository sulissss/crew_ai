from textwrap import dedent
from crewai import Agent
from src.config import get_llm
from src.tools.file_io import save_to_file
from src.tools.validation import validate_and_save


def create_validation_agent():
    return Agent(
        role="Validation Agent",
        goal=dedent("""\
            Compare the meeting transcript with the Note-Taker Agent's response
            and validate the output format. Ensure the response contains a proper
            agenda, a 300-word summary, and correctly structured discussion points."""),
        backstory=dedent("""\
            You are a validator for the Meeting Note-Taker Agent. As the meeting
            transcripts are transcribed from audio to text, there is a high chance
            of transcription errors in the transcript, due to which the meeting
            report may be affected. You validate the responses of the Note-Taker
            Agent against the expected JSON schema and provide feedback."""),
        verbose=True,
        tools=[validate_and_save, save_to_file],
        llm=get_llm(),
    )
