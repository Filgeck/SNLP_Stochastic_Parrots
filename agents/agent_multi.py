from agents.base import Agent
from agents.agent_file_selector import AgentFileSelector
from agents.agent_example_retriever import AgentExampleRetriever
from agents.agent_programmer import AgentProgrammer
from clients import ModelClient, RagClient


class AgentMulti(Agent):
    def __init__(
        self,
        model_client: ModelClient,
        max_retries: int = 3,
        param_count: str | None = None,
        temp: float | None = None
    ) -> None:
        super().__init__(
            model_client=model_client,
            max_retries=max_retries,
            param_count=param_count,
            agent_name="Multi Agent",
            temp=temp
        )
        self.model_client = model_client

        file_selector_model_client = ModelClient(model_name=model_client.model_name)
        self.agent1 = AgentFileSelector(
            model_client=file_selector_model_client,
            return_full_text=True,
            strip_line_num=True,
        )

        rag_client = RagClient()
        rag_model_client = ModelClient(model_name=model_client.model_name)
        self.agent2 = AgentExampleRetriever(
            model_client=rag_model_client, rag_client=rag_client
        )

        programmer_model_client = ModelClient(model_name=model_client.model_name)
        self.agent3 = AgentProgrammer(model_client=programmer_model_client)

        self.agent_name = "agent_multi"

    def forward(self, prompt: str) -> str | None:
        prompt = prompt.split("</code>", 1)[0] + "</code>"

        files_list, only_selected_files, hint = self.agent1.forward(prompt, "batch")

        # get issue between <issue> and </issue>
        issue = prompt[prompt.find("<issue>") + len("<issue>") : prompt.find("</issue>")]

        rags = self.agent2.forward(issue, 10, 3)

        programmer_prompt = only_selected_files + "\n" + rags + "\n Here is a hint: \n" + hint

        # SET CLEANUP TO FALSE
        patch = self.agent3.forward(programmer_prompt)

        return patch
