from agents import Agent
from agents import AgentFileSelector
from agents import AgentExampleRetriever
from agents import AgentProgrammer
from clients import ModelClient, RagClient

class MultiAgent(Agent):
    def __init__(self, model_client: ModelClient, max_retries: int = 3):
        super().__init__(model_client=model_client, max_retries=max_retries)
        self.model_client = model_client

        file_selector_model_client = ModelClient(model_name="gemini-2.5-pro-exp-03-25")
        self.agent1 = AgentFileSelector(model_client=file_selector_model_client, return_full_text=True, strip_line_num=True)

        rag_client = RagClient()
        rag_model_client = ModelClient(model_name="gemini-2.5-pro-exp-03-25")
        self.agent2 = AgentExampleRetriever(model_client=rag_model_client, rag_client=rag_client)

        programmer_model_client = ModelClient(model_name="gemini-2.5-pro-exp-03-25")
        self.agent3 = AgentProgrammer(model_client=programmer_model_client)

        self.agent_name = "Multi Agent"

    def forward(self, prompt: str) -> str | None:
        # remove patches from the prompt
        prompt = prompt.split('</code>', 1)[0] + '</code>'

        files_list, only_selected_files = self.agent1.forward(prompt, "batch")

        # get issue between <issue> and </issue>
        start = prompt.find("<issue>")
        end = prompt.find("</issue>")
        issue = prompt[start + len("<issue>") : end]
        
        rags = self.agent2.forward(issue, 10, 3)

        programmer_prompt = only_selected_files + "\n" + rags

        print(programmer_prompt)

        # SET CLEANUP TO FALSE
        patch = self.agent3.forward(programmer_prompt)

        return patch

        