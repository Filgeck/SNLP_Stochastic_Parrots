from agents.base import Agent
from clients import ModelClient


class AgentFileRetriever(Agent):
    def __init__(self, model_client: ModelClient, max_retries: int = 3):
        super().__init__(model_client=model_client, max_retries=max_retries)
        self.agent_name = "agent_file_retriever"

    def forward(
        self, issue: str, base_commit: str, environment_setup_commmit: str
    ) -> str:
        """AgentFileRetriever fetches files from the given codebase based on the relevancy of the issue/errors."""
        raise NotImplementedError