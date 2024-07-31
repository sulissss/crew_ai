from crewai import Agent, Task, Crew, Process
from langchain_community.llms.ollama import Ollama
from textwrap import dedent
from langchain.agents import tool
import json
import os
os.environ["OPENAI_API_KEY"] = "NA"

llm = Ollama(model = "llama3.1")

def call_back(output):
    output_str = json.dumps(output, default=str)
    with open("callback_resp.json", "w") as file:
        file.write(output_str)
    print("callback done")


@tool
def save_to_file(llm_response: str):
    """Save your JSON response here after each iteration"""
    with open("math_professor.txt", 'w') as file:
        file.write(llm_response)
    return "File Saved"

agent_1 = Agent(role="Math Professor",
                goal="Solve the given equation.",
                backstory="An esteemed Math Professor at the most prestigious univeristy, you are tasked to solve the math problems given to you and save to file.",
                verbose=True,
                tools=[save_to_file],
                llm=llm)

# agent_2 = Agent(role="JSON object Parser",
#                 goal="Take the output from agents 1 and 3 as input, and convert it to a list of JSON objects",
#                 tools=[],
#                 backstory="""You convert all LLM responses to JSON objects to be processed further. JSON objects are of the form /{'question': <>, 'answer': <> /}. You return the data in the braces only, and you place the input after the question field, and your answer after the output field. For example, if 
#                             the Math Professor agent returns 1+3=4. You insert the following JSON object into a list /{'question': '1+3', 'answer': '4' /}""",
#                 verbose=True,
#                 llm=llm)

# agent_3 = Agent(role="Science Professor",
#                 goal="Solve the given question",
#                 tools=[],
#                 backstory="An esteemed Science Professor at the most prestigious univeristy, you are tasked to solve the scientific problems given to you.",
#                 verbose=True,
#                 max_iter=1,
#                 llm=llm)

task_1 = Task(description="What is 3+5? Store the response locally on the system using the provided tool. Mention the tool used.After performing the tast exit the loop",
              expected_output="A single line equation with the answer",
              agent=agent_1,
              output_file="callback_resp.txt")

# task_2 = Task(description="Create a JSON object",
#               expected_output="A JSON object of the format: 'question': <the-question>, 'answer': <your-answer>",
#               agent=agent_2)

# task_3 = Task(description="How far away is the Sun?",
#               expected_output="A single line answer",
#               agent=agent_3)

crew = Crew(agents=[agent_1],
            tasks=[task_1])

result = crew.kickoff()

print(result)