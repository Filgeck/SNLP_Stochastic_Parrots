from clients import RagClient
from clients import ModelClient
from agents import AgentMulti
from agents import AgentFileSelector
from agents import AgentExampleRetriever
from agents import AgentProgrammer

multi_agent_model_client = ModelClient(model_name="gemini-2.5-pro-exp-03-25")
multi_agent = AgentMulti(
    model_client=multi_agent_model_client,
    max_retries=3,
)

example_prompt = ""
with open("data/example_prompt.txt", "r") as file:
    example_prompt = file.read()

patch = multi_agent.forward(example_prompt)

print("Patch:")
print(patch)
