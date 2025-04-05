from model_client import ModelClient
from abc import ABC, abstractmethod
import re
from typing import List


class Agent(ABC):
    def __init__(self, model_client: ModelClient):
        self.model_client = model_client
        self.agent_name = "agent"

    @abstractmethod
    def forward(self, *inputs):
        """Process inputs and produce output according to agent's role."""
        pass

    def _extract_patch(self, response_text: str) -> str:
        """Extracts the patch content from the agent response, supporting both <patch> tags and ```patch blocks."""
        # try to match <patch>...</patch>
        match = re.search(
            r"<patch>\s*\n?(.*?)\n?\s*</patch>",
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # else try to match ```patch ... ```
        match = re.search(
            r"```patch\s*\n?(.*?)\n?```",
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # as fallback return the whole response stripped
        return response_text.strip()


class AgentBasic(Agent):
    def __init__(self, model_client: ModelClient):
        self.model_client = model_client
        self.agent_name = "agent_basic"

    def forward(self, prompt):
        """AgentBasic passes the prompt directly."""
        return self._extract_patch(self.model_client.query(prompt))


class AgentFileSelector(Agent):
    def __init__(self, model_client: ModelClient):
        self.model_client = model_client
        self.agent_name = "agent_file_selector"

    def forward(self, problem: str, files: List[str]) -> str:
        """AgentFileSelector select which files to pass on based on if they are relevant to the current problem/errors."""
        raise NotImplementedError


class AgentExampleRetriever(Agent):
    def __init__(self, model_client: ModelClient):
        self.model_client = model_client
        self.agent_name = "agent_example_retriever"

    def forward(self, problem: str, num_retrieve: int, num_select: int) -> str:
        """AgentExampleRetriever fetches examples via RAG of stack exchange solutions to questions that are similar to the current problem/errors."""
        raise NotImplementedError
