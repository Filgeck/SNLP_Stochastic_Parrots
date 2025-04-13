from agents.base import Agent
from clients import ModelClient, RagClient


class AgentExampleRetriever(Agent):
    def __init__(
        self, model_client: ModelClient, rag_client: RagClient, max_retries: int = 3
    ):
        super().__init__(model_client=model_client, max_retries=max_retries)
        self.rag_client = rag_client
        self.model_client = model_client

    def forward(self, issue_description: str, num_retrieve: int, num_select: int) -> str:
        """AgentExampleRetriever fetches examples via RAG of stack exchange solutions to questions that are similar to the current issue/errors."""


        RAGS_unfiltered = self.rag_client.query(
            issue_description = issue_description,
            num_retrieve=num_retrieve,
        )

        RAGS_string = ""
        for i, example in enumerate(RAGS_unfiltered):
            RAGS_string += f"<example>\n{example}\n</example>\n"

        prompt = f"""Your task is to select the top {num_select} relevant examples that are similar to the issue, 
        this will be passed on to another model (to help it solve the issue) which will use these as context to solve the issue:
        \n<issue>\n{issue_description}\n</issue>
        \n<POTENTIAL_EXAMPLES>\n{RAGS_string}\n</POTENTIAL_EXAMPLES>
        \n Return {num_select} examples in <example> exact text from example </example> tags so I can split them up. Do not solve the issue, just return the examples.
        Make sure to put the examples in the correct format in between the <example> and </example> tags."""

        RAGS_filtered_string = self.model_client.query(prompt)

        # extract all examples in <example> ... </example>
        RAGS_filtered = []
        for example in RAGS_filtered_string.split("<example>")[1:]:
            example = example.split("</example>")[0]
            if len(example) > 0:
                RAGS_filtered.append(example)

        RAGS_filtered_output = ""

        for i, example in enumerate(RAGS_filtered):
            # Add the start and end markers
            example = example.strip()
            if example:
                RAGS_filtered_output += f"[start of example_{i+1}]\n{example}\n[end of example_{i+1}]\n"
        # Add the closing tag
        RAGS_filtered_output = f"<examples>\n{RAGS_filtered_output}</examples>"

        return RAGS_filtered_output
