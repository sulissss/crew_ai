from crewai import Crew
from src.agents.note_taker import create_note_taker_agent, create_backup_agent
from src.agents.validation import create_validation_agent
from src.tasks.note_taker import create_note_taking_task, create_backup_task
from src.tasks.validation import create_validation_task


def run_meeting_notes(transcript, agenda, validate=False):
    note_taker = create_note_taker_agent()

    note_taking = create_note_taking_task(note_taker, agenda, transcript)

    if validate:
        validator = create_validation_agent()
        validation = create_validation_task(validator, transcript)
        validation.context = [note_taking]

        crew = Crew(
            agents=[note_taker, validator],
            tasks=[note_taking, validation],
        )
    else:
        backup = create_backup_agent()
        backup_task = create_backup_task(backup)
        backup_task.context = [note_taking]

        crew = Crew(
            agents=[note_taker, backup],
            tasks=[note_taking, backup_task],
        )

    return crew.kickoff()
