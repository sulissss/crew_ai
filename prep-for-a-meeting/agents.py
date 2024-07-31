from textwrap import dedent
from crewai import Agent
from langchain.agents import tool
from tools.ExaSearchTool import ExaSearchTool
from langchain_community.llms.ollama import Ollama
import os

llm = Ollama(model="llama3.1")

@tool
def save_to_file(llm_response: dict):
    """Save your JSON response here after each iteration"""
    with open("validation_resp.json", 'w') as file:
        file.write(llm_response)
    return "File Saved"

@tool
def check_file(file_name: str):
	"""Verify if the file was saved correctly on the system"""
	return os.path.exists(file_name)

class MeetingPreparationAgents():
	def research_agent(self):
		return Agent(
			role='Research Specialist',
			goal='Conduct thorough research on people and companies involved in the meeting',
			tools=ExaSearchTool.tools(),
			backstory=dedent("""\
					As a Research Specialist, your mission is to uncover detailed information
					about the individuals and entities participating in the meeting. Your insights
					will lay the groundwork for strategic meeting preparation."""),
			verbose=True
		)

	def industry_analysis_agent(self):
		return Agent(
			role='Industry Analyst',
			goal='Analyze the current industry trends, challenges, and opportunities',
			tools=ExaSearchTool.tools(),
			backstory=dedent("""\
					As an Industry Analyst, your analysis will identify key trends,
					challenges facing the industry, and potential opportunities that
					could be leveraged during the meeting for strategic advantage."""),
			verbose=True
		)

	def meeting_strategy_agent(self):
		return Agent(
			role='Meeting Strategy Advisor',
			goal='Develop talking points, questions, and strategic angles for the meeting',
			tools=ExaSearchTool.tools(),
			backstory=dedent("""\
					As a Strategy Advisor, your expertise will guide the development of
					talking points, insightful questions, and strategic angles
					to ensure the meeting's objectives are achieved."""),
			verbose=True
		)

	def summary_and_briefing_agent(self):
		return Agent(
			role='Briefing Coordinator',
			goal='Compile all gathered information into a concise, informative briefing document',
			tools=ExaSearchTool.tools(),
			backstory=dedent("""\
					As the Briefing Coordinator, your role is to consolidate the research,
					analysis, and strategic insights."""),
			verbose=True
		)
	
	def note_taker_agent(self):
		return Agent(
			role="Professional Note-Taker for a Meeting",
			goal=f"""Extract the agenda, summary, and discussion points from meeting transcript.""",
			tools=[save_to_file],           
			backstory=dedent("""You are a note-taker for meetings. You are always handed out meeting transcripts, which are sometimes
								easier to understand, but sometimes they make no sense. This is due to the fact that they are directly 
								transcribed from audio to text, which does result in a lot of transcription issues due to a difference in accents, distance from the microphone, etc.
								You need to decipher, judging by the context of the meeting and using your LLM reasoning, whether the dialogue mentioned in the script was truly spoken by the said person
								or was it just a mere transcription error. You also need to filter out any explicit content that's spoken in the meetings."""),
			verbose=True,     
			llm=llm
		)

	def backup_agent(self):
		return Agent(
			role="A Backup Agent",
			goal="Save the responses sent by the other agent(s) at each iteration",
			backstory="As a Backup for the program, checking up on your fellow agents, and storing their thoughts and responses.",
			verbose=True,
			tools=[save_to_file, check_file],
			llm=llm
		)
