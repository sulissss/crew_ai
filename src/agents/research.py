from textwrap import dedent
from crewai import Agent
from src.tools.exa_search import ExaSearchTool


def create_research_agent():
    return Agent(
        role="Research Specialist",
        goal="Conduct thorough research on people and companies involved in the meeting",
        tools=ExaSearchTool.tools(),
        backstory=dedent("""\
            As a Research Specialist, your mission is to uncover detailed information
            about the individuals and entities participating in the meeting. Your insights
            will lay the groundwork for strategic meeting preparation."""),
        verbose=True,
    )


def create_industry_analysis_agent():
    return Agent(
        role="Industry Analyst",
        goal="Analyze the current industry trends, challenges, and opportunities",
        tools=ExaSearchTool.tools(),
        backstory=dedent("""\
            As an Industry Analyst, your analysis will identify key trends,
            challenges facing the industry, and potential opportunities that
            could be leveraged during the meeting for strategic advantage."""),
        verbose=True,
    )


def create_meeting_strategy_agent():
    return Agent(
        role="Meeting Strategy Advisor",
        goal="Develop talking points, questions, and strategic angles for the meeting",
        tools=ExaSearchTool.tools(),
        backstory=dedent("""\
            As a Strategy Advisor, your expertise will guide the development of
            talking points, insightful questions, and strategic angles
            to ensure the meeting's objectives are achieved."""),
        verbose=True,
    )


def create_briefing_agent():
    return Agent(
        role="Briefing Coordinator",
        goal="Compile all gathered information into a concise, informative briefing document",
        tools=ExaSearchTool.tools(),
        backstory=dedent("""\
            As the Briefing Coordinator, your role is to consolidate the research,
            analysis, and strategic insights into a comprehensive briefing."""),
        verbose=True,
    )
