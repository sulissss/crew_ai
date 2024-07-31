from dotenv import load_dotenv
load_dotenv()

from crewai import Crew

from tasks import MeetingPreparationTasks
from agents import MeetingPreparationAgents

tasks = MeetingPreparationTasks()
agents = MeetingPreparationAgents()

print("## Welcome to the Meeting Prep Crew")
print('-------------------------------')
# participants = input("What are the emails for the participants (other than you) in the meeting?\n")
# context = input("What is the context of the meeting?\n")
# objective = input("What is your objective for this meeting?\n")

# Create Agents
# researcher_agent = agents.research_agent()
# industry_analyst_agent = agents.industry_analysis_agent()
# meeting_strategy_agent = agents.meeting_strategy_agent()
# summary_and_briefing_agent = agents.summary_and_briefing_agent()

note_taker_agent = agents.note_taker_agent()
backup_agent = agents.backup_agent()

# Create Tasks
# research = tasks.research_task(researcher_agent, participants, context)
# industry_analysis = tasks.industry_analysis_task(industry_analyst_agent, participants, context)
# meeting_strategy = tasks.meeting_strategy_task(meeting_strategy_agent, context, objective)
# summary_and_briefing = tasks.summary_and_briefing_task(summary_and_briefing_agent, context, objective)

meeting_transcript = "Here is the sample meeting transcript..."
with open('/Applications/Documents/Project_2_Meeting_Minutes/that_meeting.txt', 'r') as file:
    meeting_transcript = file.read()

note_taking = tasks.note_taking_task(note_taker_agent, "A project of a RAG-based chatbot", meeting_transcript)
backup = tasks.backup_task(backup_agent)

backup.context = [note_taking]
# meeting_strategy.context = [research, industry_analysis]
# summary_and_briefing.context = [research, industry_analysis, meeting_strategy]

crew = Crew(agents=[note_taker_agent, backup_agent],
            tasks=[note_taking, backup])

# # Create Crew responsible for Copy
# crew = Crew(
# 	agents=[
# 		researcher_agent,
# 		industry_analyst_agent,
# 		meeting_strategy_agent,
# 		summary_and_briefing_agent
# 	],
# 	tasks=[
# 		research,
# 		industry_analysis,
# 		meeting_strategy,
# 		summary_and_briefing
# 	]
# )

game = crew.kickoff()


# Print results
print("\n\n################################################")
print("## Here is the result")
print("################################################\n")
print(game)
