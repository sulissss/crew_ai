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

# Agent to extract agenda
note_taker = Agent(
    role="Professional Note-Taker for a Meeting",
    goal=f"""Extract the agenda, summary, and discussion points from meeting transcript. Save your output in a file, and do not exceed the maximum iterations, i.e. 10""",
    backstory=dedent("""You are a note-taker for meetings. You are always handed out meeting transcripts, which are sometimes
                        easier to understand, but sometimes they make no sense. This is due to the fact that they are directly 
                        transcribed from audio to text, which does result in a lot of transcription issues due to a difference in accents, distance from the microphone, etc.
                        You are the sole agent in this crew."""),
    verbose=True,     
    tools=[save_text_to_file, save_jsonobject_to_file],           
    llm=llm
)

meeting_transcript = "Here is the sample meeting transcript..."
with open('/Applications/Documents/Project_2_Meeting_Minutes/that_meeting.txt', 'r') as file:
    meeting_transcript = file.read()

# Task for Note-Taker Agent
task = Task(
    description=f""""Extract the agenda, summary, and discussion points from the following meeting transcript. 
                    You need to decipher, judging by the context of the meeting and using your LLM reasoning, whether the dialogue mentioned in the script was truly spoken by the said person
                    or was it just a mere transcription error. You also need to filter out any explicit content that's spoken in the meetings.
                    Save your result to a file after each thought that you process.
                    Do not write a response like 'Agent stopped due to iteration limit or time limit.' in the file. 
                    You should only write the desired JSON object in the file.""",
    agent=note_taker,
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
                                    "discussion_point": "It was suggested to use a recursive approach to boost context, where the response is passed as input and then answered recursively.",
                                    "person_responsible": "-",
                                    "completion_date": <completion_date>,
                                    "remarks": <the_remarks>
                                },
                                {
                                    "discussion_point": "Another option mentioned was using API endpoints to improve the chatbot.",
                                    "person_responsible": "-",
                                    "completion_date": <completion_date>,
                                    "remarks": <the_remarks>
                                }
                            ]
                        }
                        """
)

crew = Crew(
    agents=[note_taker],
    tasks=[task],
    # process=Process.sequential
)

print("starting")
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