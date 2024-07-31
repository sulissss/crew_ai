from textwrap import dedent
from crewai import Task

class MeetingPreparationTasks():
	def research_task(self, agent, participants, context):
		return Task(
			description=dedent(f"""\
				Conduct comprehensive research on each of the individuals and companies
				involved in the upcoming meeting. Gather information on recent
				news, achievements, professional background, and any relevant
				business activities.

				Participants: {participants}
				Meeting Context: {context}"""),
			expected_output=dedent("""\
				A detailed report summarizing key findings about each participant
				and company, highlighting information that could be relevant for the meeting."""),
			async_execution=True,
			agent=agent
		)

	def industry_analysis_task(self, agent, participants, context):
		return Task(
			description=dedent(f"""\
				Analyze the current industry trends, challenges, and opportunities
				relevant to the meeting's context. Consider market reports, recent
				developments, and expert opinions to provide a comprehensive
				overview of the industry landscape.

				Participants: {participants}
				Meeting Context: {context}"""),
			expected_output=dedent("""\
				An insightful analysis that identifies major trends, potential
				challenges, and strategic opportunities."""),
			async_execution=True,
			agent=agent
		)

	def meeting_strategy_task(self, agent, context, objective):
		return Task(
			description=dedent(f"""\
				Develop strategic talking points, questions, and discussion angles
				for the meeting based on the research and industry analysis conducted

				Meeting Context: {context}
				Meeting Objective: {objective}"""),
			expected_output=dedent("""\
				Complete report with a list of key talking points, strategic questions
				to ask to help achieve the meetings objective during the meeting."""),
			agent=agent
		)

	def summary_and_briefing_task(self, agent, context, objective):
		return Task(
			description=dedent(f"""\
				Compile all the research findings, industry analysis, and strategic
				talking points into a concise, comprehensive briefing document for
				the meeting.
				Ensure the briefing is easy to digest and equips the meeting
				participants with all necessary information and strategies.

				Meeting Context: {context}
				Meeting Objective: {objective}"""),
			expected_output=dedent("""\
				A well-structured briefing document that includes sections for
				participant bios, industry overview, talking points, and
				strategic recommendations."""),
			agent=agent
		)
	
	def note_taking_task(self, agent, agenda, transcript):
		return Task(
			description=f""""Extract the agenda, summary, and discussion points from the following meeting transcript. Do not loop, just perform the task once.
							Meeting Agenda: {agenda}
							Meeting Transcript: {transcript}""",
			agent=agent,
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
	
	def backup_task(self, agent):
		return Task(description=f"Collect the response data from other agents, and save them to a local file. Then, verify if the file was saved correctly.",
			  agent=agent,
			  expected_output="An affirmation if the data has been saved, along with the file path where it was saved.")
