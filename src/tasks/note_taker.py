from textwrap import dedent
from crewai import Task

EXPECTED_OUTPUT_SCHEMA = dedent("""\
    A JSON object of the following format:
    {
        "agenda": <the-agenda> (one line),
        "summary": <the-summary> (300 words),
        "discussion": [
            {
                "discussion_point": <discussion_point>,
                "person_responsible": <person_responsible>,
                "completion_date": <completion_date>,
                "remarks": <the_remarks>
            }
        ]
    }""")


def create_note_taking_task(agent, agenda, transcript):
    return Task(
        description=dedent(f"""\
            Extract the agenda, summary, and discussion points from the
            following meeting transcript. Do not loop, just perform the
            task once.

            Meeting Agenda: {agenda}
            Meeting Transcript: {transcript}"""),
        agent=agent,
        expected_output=EXPECTED_OUTPUT_SCHEMA,
    )


def create_backup_task(agent):
    return Task(
        description="Collect the response data from other agents and save them to a local file. Then verify if the file was saved correctly.",
        agent=agent,
        expected_output="An affirmation if the data has been saved, along with the file path where it was saved.",
    )
