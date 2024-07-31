from crewai import Agent, Task, Crew
from langchain_community.llms.ollama import Ollama
from textwrap import dedent
import os

os.environ["OPENAI_API_KEY"] = "NA"

llm = Ollama(model="llama3.1")

# Agent to extract agenda
agenda_agent = Agent(role="Agenda Extractor",
                     goal="Extract the agenda from the meeting transcript.",
                     backstory="You are an expert at identifying the agenda from text.",
                     verbose=True,
                     llm=llm)

# Agent to summarize the meeting
summary_agent = Agent(role="Meeting Summarizer",
                      goal="Summarize the meeting transcript.",
                      backstory="You provide concise and accurate summaries of meetings.",
                      verbose=True,
                      llm=llm)

# Agent to extract discussion points
discussion_points_agent = Agent(role="Discussion Points Extractor",
                                goal="Extract the main discussion points from the meeting transcript.",
                                backstory="You identify key discussion points from text.",
                                verbose=True,
                                llm=llm)

# Sample meeting transcript
meeting_transcript = "Here is the sample meeting transcript..."
with open('/Applications/Documents/Project_2_Meeting_Minutes/that_meeting.txt', 'r') as file:
    meeting_transcript = file.read()

# Tasks
agenda_task = Task(description=f"Extract the agenda from the following meeting transcript: {meeting_transcript}",
                   agent=agenda_agent,
                   expected_output="A list of agenda items."
                   )

summary_task = Task(description=f"Using the agenda list from the agenda agent as context, provide a detailed summary of the following meeting transcript: {meeting_transcript}",
                    agent=summary_agent,
                    expected_output="A summary of the meeting as bullet points. Word limit: 300 words."
                    )

discussion_points_task = Task(description=f"Using the agenda list from the agenda agent as context, extract the main discussion points from the following meeting transcript: {meeting_transcript}",
                              agent=discussion_points_agent,
                              expected_output=dedent("""A JSON object of discussion points in the following format:
                                                         discussion_points = [
                                                            {
                                                                "discussion": "Discussed project milestones and deadlines.",
                                                                "person_responsible": "John Doe",
                                                                "completion_date": "July 16, 2024",
                                                                "remarks": "Initial planning phase completed."
                                                            },
                                                            {
                                                                "discussion": "Addressed challenges in the current sprint.",
                                                                "person_responsible": "Jane Smith",
                                                                "completion_date": "July 16, 2024",
                                                                "remarks": "Identified key issues and solutions."
                                                            },
                                                            {
                                                                "discussion": "Reviewed customer feedback and action items.",
                                                                "person_responsible": "Mike Johnson",
                                                                "completion_date": "July 16, 2024",
                                                                "remarks": "Feedback incorporated into the next release."
                                                            }
                                                        ]""")
                                                        )

# Create a crew
crew = Crew(
            agents=[agenda_agent, summary_agent, discussion_points_agent],
            tasks=[agenda_task, summary_task, discussion_points_task],
            verbose=2
        )

# Execute the tasks
result = crew.kickoff()

# Print the results
print("Agenda:", result['outputs'][0])
print("Summary:", result['outputs'][1])
print("Discussion Points:", result['outputs'][2])