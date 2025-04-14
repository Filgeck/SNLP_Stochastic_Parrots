from agents.base import Agent
from clients import ModelClient


class AgentBasic(Agent):
    def __init__(
        self,
        model_client: ModelClient,
        max_retries: int = 3,
        param_count: str | None = None
    ) -> None:
        super().__init__(
            model_client=model_client,
            max_retries=max_retries,
            param_count=param_count)
        self.agent_name = "agent_basic"

    def forward(self, prompt):
        """AgentBasic passes the prompt directly."""

        instruction = """Feel free to analyse and edit files as required, however you must absolutely ensure that at the end of your response you enclose your final patch in either <patch> </patch> tags or a ```patch ``` block."""
        prompt = f"{prompt}\n{instruction}"

        def helper(prompt_arg):
            patch = self._query_and_extract(prompt_arg, "patch")
            if patch.startswith("---"):
                return patch
            else:
                raise ValueError(
                    f"Expected patch to start with '---', instead started with {patch[:20]}..."
                )

        return self._func_with_retries(helper, prompt)
