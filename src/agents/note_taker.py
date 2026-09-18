from textwrap import dedent
from crewai import Agent
from src.config import get_llm
from src.tools.file_io import save_to_file, save_json_to_file


def create_note_taker_agent():
    return Agent(
        role="Professional Note-Taker for a Meeting",
        goal="Extract the agenda, summary, and discussion points from meeting transcripts.",
        tools=[save_to_file, save_json_to_file],
        backstory=dedent("""\
            You are a note-taker for meetings. You are always handed out meeting
            transcripts, which are sometimes easier to understand, but sometimes
            they make no sense. This is due to the fact that they are directly
            transcribed from audio to text, which does result in a lot of
            transcription issues due to a difference in accents, distance from
            the microphone, etc. You need to decipher, judging by the context of
            the meeting and using your LLM reasoning, whether the dialogue
            mentioned in the script was truly spoken by the said person or was it
            just a mere transcription error. You also need to filter out any
            explicit content that's spoken in the meetings."""),
        verbose=True,
        llm=get_llm(),
    )


def create_backup_agent():
    return Agent(
        role="Backup Agent",
        goal="Save the responses sent by the other agent(s) at each iteration",
        backstory="As a backup for the program, checking up on your fellow agents and storing their thoughts and responses.",
        verbose=True,
        tools=[save_to_file, save_json_to_file],
        llm=get_llm(),
    )
