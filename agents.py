from model_client import ModelClient
from abc import ABC, abstractmethod
import re
from typing import List, Literal


class Agent(ABC):
    def __init__(self, model_client: ModelClient):
        self.model_client = model_client
        self.agent_name = "agent"

    @abstractmethod
    def forward(self, *inputs):
        """Process inputs and produce output according to agent's role."""
        pass

    def _extract_tag(self, response_text: str, tag_name: str) -> str:
        """Extract content from the agent response, supporting custom tags both in <example> tags and ```example blocks."""
        # try to match <tag_name> </tag_name>
        match = re.search(
            rf"<{tag_name}>\s*\n?(.*?)\n?\s*</{tag_name}>",
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # else try to match ```tag_name ```
        match = re.search(
            rf"```{tag_name}\s*\n?(.*?)\n?```",
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # else return None for retry as it means:
        # - unclosed <tag_name> (incomplete response)
        # - unclosed ```tag_name (incomplete response)
        # - no tag or block (bad response)
        return None

    def _extract_tag_with_retries(self, response_text: str, tag_name: str) -> str:
        """Extract tag, re-querying to model if tag extraction fails."""
        tag_content = self._extract_tag(
            self.model_client.query(response_text), tag_name
        )
        num_retries = 0
        max_retries = self.model_client.max_retries
        while tag_content is None and num_retries < max_retries:
            print(
                f"Failed attempt {num_retries + 1} of {max_retries}.\nRetrying due to empty '{tag_name}' tag/block."
            )
            tag_content = self._extract_tag(
                self.model_client.query(response_text), tag_name
            )
            num_retries += 1
        if tag_content is None:
            raise RuntimeError(
                f"Max retries reached. Unable to extract '{tag_name}' tag/block from model's response."
            )
        return tag_content

    def _extract_patch_with_retries(self, response_text: str) -> str:
        """Extract patch, re-querying to model if patch extraction fails."""
        return self._extract_tag_with_retries(self, response_text, "patch")


class AgentBasic(Agent):
    def __init__(self, model_client: ModelClient):
        self.model_client = model_client
        self.agent_name = "agent_basic"

    def forward(self, prompt):
        """AgentBasic passes the prompt directly."""
        return self._extract_patch_with_retries(prompt)


class AgentFileSelector(Agent):
    def __init__(self, model_client: ModelClient):
        self.model_client = model_client
        self.agent_name = "agent_file_selector"

    def forward(
        self,
        text: str,
        method: Literal["batch", "individual"],
        custom_issue: str = None,
    ) -> List[str]:
        """AgentFileSelector selects which files to pass on based on if they are relevant to the current issue/errors."""

        issue_text = custom_issue or self._extract_tag(custom_issue, "issue")
        files_text = self._extract_tag(text, "code")

        if method == "batch":  # pass all files to the model
            prompt = f"""Your task is to select the files that are relevant to 
            solving the issue, this will be passed on to another model which 
            will use these as context to solve the issue:
            \n<issue>\n{issue_text}\n</issue>
            \n<files>\n{files_text}\n</files>
            \nPlease ensure your response is a list of selected files with path 
            (exactly as they appear above). Return your list of selected files in 
            either <selected> </selected> tags or in a ```selected ``` block. 
            Here is an example:
            \n<selected>\n
            astropy/time/formats.py\n
            astropy/coordinates/distances.py\n
            docs/conf.py\n
            </selected>"""
        elif method == "individual":  # pass each file to the model
            pass


class AgentFileRetriever(Agent):
    def __init__(self, model_client: ModelClient):
        self.model_client = model_client
        self.agent_name = "agent_file_retriever"

    def forward(
        self, issue: str, base_commit: str, environment_setup_commmit: str
    ) -> str:
        """AgentFileRetriever fetches files from the given codebase based on the relevancy of the issue/errors."""
        raise NotImplementedError


class AgentExampleRetriever(Agent):
    def __init__(self, model_client: ModelClient):
        self.model_client = model_client
        self.agent_name = "agent_example_retriever"

    def forward(self, issue: str, num_retrieve: int, num_select: int) -> str:
        """AgentExampleRetriever fetches examples via RAG of stack exchange solutions to questions that are similar to the current issue/errors."""
        raise NotImplementedError
