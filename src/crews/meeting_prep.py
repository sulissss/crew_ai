from crewai import Crew
from src.agents.research import (
    create_research_agent,
    create_industry_analysis_agent,
    create_meeting_strategy_agent,
    create_briefing_agent,
)
from src.tasks.research import (
    create_research_task,
    create_industry_analysis_task,
    create_meeting_strategy_task,
    create_briefing_task,
)


def run_meeting_prep(participants, context, objective):
    researcher = create_research_agent()
    analyst = create_industry_analysis_agent()
    strategist = create_meeting_strategy_agent()
    briefer = create_briefing_agent()

    research = create_research_task(researcher, participants, context)
    analysis = create_industry_analysis_task(analyst, participants, context)
    strategy = create_meeting_strategy_task(strategist, context, objective)
    briefing = create_briefing_task(briefer, context, objective)

    strategy.context = [research, analysis]
    briefing.context = [research, analysis, strategy]

    crew = Crew(
        agents=[researcher, analyst, strategist, briefer],
        tasks=[research, analysis, strategy, briefing],
    )

    return crew.kickoff()
