from crewai import Agent, Task, Crew, Process
from langchain_community.llms.ollama import Ollama
from textwrap import dedent
from langchain.agents import tool
import os
import json

os.environ["OPENAI_API_KEY"] = "NA"

llm = Ollama(model="llama3.1")

def call_back(output: str):
    with open("callback_resp.txt", 'w') as file:
        file.write(output)

class SaveOutput:
    @tool
    def save_to_file(llm_response: str):
        """Save your JSON response here after each iteration"""
        with open("llm_response.txt", 'w') as file:
            file.write(llm_response)

    def tools():
        return [SaveOutput.save_to_file]

def validate_output(output):
    expected_keys = ["agenda", "summary", "discussion"]
    if all(key in output for key in expected_keys) and isinstance(output["discussion"], list):
        for discussion in output["discussion"]:
            if not all(key in discussion for key in ["discussion_point", "person_responsible", "completion_date", "remarks"]):
                return False
        return True
    return False

class ValidationAgent:
    @tool
    def validate_and_save(output: str):
        """Validate the Note-Taker Agent's responses, and save them here if valid. Return an error, if the format is invalid."""
        output = json.loads(output)
        if validate_output(output):
            with open("validated_response.json", "w") as file:
                json.dump(output, file, indent=4)
            return "Output is valid and saved to 'validated_response.json'"
        else:
            return "Output is invalid"

    def tools():
        return [ValidationAgent.validate_and_save]

# Agent to extract agenda
note_taker = Agent(
    role="Professional Note-Taker for a Meeting",
    goal="""Extract the agenda, summary, and discussion points from meeting transcripts. Return your thought-out output after each iteration, and use the SaveOutput tool to store that output in a file.
            The Validation Agent will be validating your responses. Act exactly on that agent's requirements.""",
    backstory=dedent("""You are a note-taker for meetings. You are always handed out meeting transcripts, which are sometimes
                        easier to understand, but sometimes they make no sense. This is due to the fact that they are directly 
                        transcribed from audio to text, which does result in a lot of transcription issues due to a difference in accents, distance from the microphone, etc.
                        You need to decipher, judging by the context of the meeting and using your LLM reasoning, whether the dialogue mentioned in the script was truly spoken by the said person
                        or was it just a mere transcription error. You also need to filter out any explicit content that's spoken in the meetings. All in all, expect the transcript to be
                        senseless, and derive context out of it. That is your job to do! Also, you'll be working with a Validation Agent who'll be monitoring and validating your responses."""),
    verbose=True,     
    tools=SaveOutput.tools(),           
    llm=llm
)

# Validation Agent
validation_agent = Agent(
    role="Validation Agent",
    goal="""Validate the JSON response format of the Note-Taker Agent. If valid, save the response to 'validated_response.json'. If invalid, return an error message.""",
    backstory=dedent("""You are responsible for ensuring the output JSON is correctly formatted according to the specified structure. You will validate the response from the Note-Taker Agent and save it if valid."""),
    verbose=True,
    tools=ValidationAgent.tools(),
    llm=llm
)

# Sample meeting transcript
meeting_transcript = "Here is the sample meeting transcript..."
with open('/Applications/Documents/Project_2_Meeting_Minutes/that_meeting.txt', 'r') as file:
    meeting_transcript = file.read()

# Task for Note-Taker Agent
task = Task(
    description=f""""Extract the agenda, summary, and discussion points from the following meeting transcript: 
                    {meeting_transcript}""",
    agent=note_taker,
    output_file='llm_response.txt',
    expected_output="""A JSON object of the following format. For example, if you have the output for summary, you should
                        insert your output in <the-summary> after "summary". The parenthesis mentions the word/line limit for each output.
                        Write "-" in front of any field who's data is unavailable.  
                        Save the output for each iteration to a file.
                        Mention the file name where you saved the data: 
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

# Task for Validation Agent
validation_task = Task(
    description="Validate the response from the Note-Taker Agent.",
    agent=validation_agent
)

crew = Crew(
    agents=[note_taker, validation_agent],
    tasks=[task, validation_task]
)

print("starting")
result = crew.kickoff()

print("\n\ndone:\n\n")
print(result)
