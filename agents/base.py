from abc import abstractmethod
import re
from typing import List, Tuple

from clients import Retries, ModelClient


class Agent(Retries):
    def __init__(self, model_client: ModelClient, max_retries: int = 3):
        super().__init__(max_retries=max_retries)
        self.model_client = model_client
        self.agent_name = "agent"

    @abstractmethod
    def forward(self, *inputs):
        """Process inputs and produce output according to agent's role."""
        pass

    def _extract_tag(self, prompt: str, tag_name: str) -> str | None:
        """Extract content from the models response, supporting custom tags both in <example> tags and ```example blocks."""
        # try to match <tag_name> </tag_name>
        match = re.search(
            rf"<{tag_name}>\s*\n?(.*?)\n?\s*</{tag_name}>",
            prompt,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # else try to match ```tag_name ```
        match = re.search(
            rf"```{tag_name}\s*\n?(.*?)\n?```",
            prompt,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # else return None for retry as it means:
        # - unclosed <tag_name> (incomplete response)
        # - unclosed ```tag_name (incomplete response)
        # - no tag or block (bad response)
        return None

    def _strip_lines(self, code_block: str) -> str:
        """
        Remove leading digits and the first optional space\n
        code_block: str - input code block (within <code>...</code> brackets, excluding tags)
        """
        return "\n".join(
            [re.sub(r"^\d+\s?", "", line) for line in code_block.splitlines()]
        )

    def _get_files(self, files_text: str, strip_line_num: bool) -> dict:
        # Regex to find all file blocks
        # \[start of (.*?)\]  -> Capture [start of example/test.py]
        # (.*?)               -> Capture the file content (non-greedy)
        # \[end of \1\]       -> Match [end of example/test.py]
        pattern = r"\[start of (.*?)\]\s*\n(.*?)\n\[end of \1\]"
        matches = re.findall(pattern, files_text, re.DOTALL)

        files_dict = {}
        for match in matches:
            file_path = match[0].strip()
            file_content = self._strip_lines(match[1]) if strip_line_num else match[1]
            files_dict[file_path] = file_content

        return files_dict

    def _query_and_extract(self, prompt: str, tag_name: str) -> str:
        """Prompt model, extract tag from models response, re-querying model if tag extraction fails."""

        def helper(prompt_arg: str, tag_name_arg: str) -> str:
            response = self.model_client.query(prompt_arg)
            extracted = self._extract_tag(response, tag_name_arg)
            if extracted is None:
                raise ValueError(
                    f"Failed to extract '{tag_name_arg}' tag/block from response"
                )
            return extracted

        return self._func_with_retries(helper, prompt, tag_name)

    def build_tagged_string(
        self, issue: str | None, code: List[Tuple[str, str]], line_numbers=False
    ) -> str:
        """
        Builds a string with blocks for the issue and for code

        args:
        - issue: the Github issue
        - code: a list of (filename, file contents) pairs
        - line_numbers: whether to provide line numbers in the output string

        returns:
            A string!
        """

        if issue is None:
            s = ""
        else:
            s = f"<issue>\n{issue}</issue>"

        s += "\n<code>\n"

        for name, contents in code:
            s += f"[start of {name}]\n"

            if line_numbers:
                for i, line in enumerate(contents.split("\n")):
                    s += f"{i + 1} {line}\n"
            else:
                s += contents + "\n"

            s += f"[end of {name}]"

        return f"{s}\n</code>"
