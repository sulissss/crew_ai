from crewai import Agent, Task, Crew, Process
from langchain_community.llms.ollama import Ollama
from textwrap import dedent
from langchain.agents import tool
import json
import os
os.environ["OPENAI_API_KEY"] = "NA"

llm = Ollama(model = "llama3.1")

@tool
def save_text_to_file(llm_response: str):
    """Save your text here locally"""
    with open("t3_python.txt", 'w') as file:
        file.write(llm_response)
    return "File Saved"

@tool
def save_jsonobject_to_file(llm_response: dict):
    """Save your JSON object here locally"""
    with open("t3_python.json", 'w') as file:
        file.write(llm_response)
    return "File Saved"



agenda_agent = Agent(role="Context-Extractor",
                goal="Find out the agenda of a meeting transcript, and send a more contextually-accurate version of the transcript to the other agents.",
                backstory="""You have been a context-analyser for meeting transcripts. You are always handed out meeting transcripts, which are sometimes
                        easier to understand, but sometimes they make no sense. This is due to the fact that they are directly 
                        transcribed from audio to text, which does result in a lot of transcription issues due to a difference in accents, distance from the microphone, etc.
                        You need to decipher, judging by the context of the meeting and using your LLM reasoning, whether the dialogue mentioned in the script was truly spoken by the said person
                        or was it just a mere transcription error. You also need to filter out any explicit content that's spoken in the meetings.""",
                verbose=True,
                tools=[save_text_to_file, save_jsonobject_to_file],
                llm=llm)

summary_agent = Agent(role="Summary Generator",
                goal="Generate a detailed, 300 word summary from a meeting transcript.",
                backstory="""You are a note-taker for meetings. You are always handed out meeting transcripts, which are sometimes
                        easier to understand, but sometimes they make no sense. This is due to the fact that they are directly 
                        transcribed from audio to text, which does result in a lot of transcription issues due to a difference in accents, distance from the microphone, etc.
                        You need to decipher, judging by the context of the meeting and using your LLM reasoning, whether the dialogue mentioned in the script was truly spoken by the said person
                        or was it just a mere transcription error. You also need to filter out any explicit content that's spoken in the meetings.""",
                verbose=True,
                tools=[save_text_to_file, save_jsonobject_to_file],
                llm=llm)

discussion_points_agent = Agent(role="Action Point Assigner",
                goal="Generate the discussion points from a meeting transcript.",
                backstory="""You are a note-taker for meetings. You are always handed out meeting transcripts, which are sometimes
                        easier to understand, but sometimes they make no sense. This is due to the fact that they are directly 
                        transcribed from audio to text, which does result in a lot of transcription issues due to a difference in accents, distance from the microphone, etc.
                        You need to decipher, judging by the context of the meeting and using your LLM reasoning, whether the dialogue mentioned in the script was truly spoken by the said person
                        or was it just a mere transcription error. You also need to filter out any explicit content that's spoken in the meetings.""",
                verbose=True,
                tools=[save_text_to_file, save_jsonobject_to_file],
                llm=llm)

reviewer_agent = Agent(role="Meeting Report Reviewer",
                goal="Review the meeting insights generated from the other agents, and form a JSON object as output under a specific format.",
                backstory="""You are a highly rated, and highly experienced meeting report reviewer. You review the reports generated by each agent, and give feedback accordingly.
                            As being the most experienced, you help them when they need you.""",
                verbose=True,
                tools=[save_text_to_file, save_jsonobject_to_file],
                llm=llm)

meeting_transcript = "Here is the sample meeting transcript..."
with open('/Applications/Documents/Project_2_Meeting_Minutes/that_meeting.txt', 'r') as file:
    meeting_transcript = file.read()

agenda_task = Task(description=f"""
                   Thought Process: Assuming that the transcript does not make any sense, decipher the true agenda and context of the meeting.
                    Use your LLM reasoning to determine whether the dialogues mentioned in the script were truly spoken by a certain person
                    or were they just a mere transcription error. You also need to filter out any explicit content that's spoken in the meetings.
                    Action: Generate an agenda for the meeting, and send a more contextually accurate version of the transcript to the other agents.
                    The Meeting transcript: {meeting_transcript}""",
              expected_output="A single line",
              agent=agenda_agent,
              async_execution=True)

summary_task = Task(description=f"""By using the Agenda of the meeting as context, 
                                    Provide a detailed summary of 300 words for the following meeting transcript: {meeting_transcript}""",
              expected_output="A 300 word paragraph",
              agent=summary_agent,
              async_execution=True)

discussion_task = Task(description=f"""By using the Agenda of the meeting as context,
                                        Provide the discussion points of the following meeting transcript, the person assigned
                                        to those tasks, the deadline of the tasks, and any further remarks on those tasks
                                        for the following meeting transcript: {meeting_transcript}""",
              expected_output="A list of points",
              agent=discussion_points_agent,
              async_execution=True)

reviewer_task = Task(
    description=f""""Review: Review the meeting reports generated by each agents, by checking for concurreny, and using reasoning on your own 
                    based on the following transcript. Give feedback to the agents regarding this.
                    Format JSON object & save: 
                    Compile the output of all agents into a JSON object in the specified format. 
                    Save your result to a file after each JSON object generated.
                    Do not write a response like 'Agent stopped due to iteration limit or time limit.' in the file. 
                    You should only write the desired JSON object in the file.
                    The Meeting Transcript: {meeting_transcript}""",
    agent=reviewer_agent,
    expected_output="""A JSON object of the following format:
                        {
                            "agenda": <the-agenda> (one line),
                            "summary": <the-summary> (300 words),
                            "discussion": [
                                {
                                    "discussion_point": <discussion_point>,
                                    "person_responsible": <person_responsible>,
                                    "completion_date": <completion_date>,
                                    "remarks": <the_remarks>
                                },
                                {
                                    "discussion_point": <discussion_point>,
                                    "person_responsible": <person_responsible>,
                                    "completion_date": <completion_date>,
                                    "remarks": <the_remarks>
                                },
                                {
                                    "discussion_point": <discussion_point>,
                                    "person_responsible": <person_responsible>,
                                    "completion_date": <completion_date>,
                                    "remarks": <the_remarks>
                                }
                            ]
                        }
                        """
)

crew = Crew(agents=[agenda_agent, summary_agent, discussion_points_agent, reviewer_agent],
            tasks=[agenda_task, summary_task, discussion_task, reviewer_task])

result = crew.kickoff()

print("\n\ndone:\n\n")
print(result)
try:
    with open('result_of_agent.txt', 'w') as file:
        file.write(result)
except:
    with open('result_of_agent.json', 'w') as file:
        file.write(result)
print("Result saved to file")