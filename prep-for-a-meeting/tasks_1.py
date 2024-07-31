# from textwrap import dedent
# from crewai import Task

# class MeetingTasks():
#     def research_task(self, agent, particpants, context):
#         return Task(
            
#         )

import json

with open('/mnt/data/callback_resp.json', 'r') as file:
    file_content = file.read()

# Convert the content back to JSON
json_data = json.loads(file_content)

# Print the JSON data to verify
print(json.dumps(json_data, indent=4))