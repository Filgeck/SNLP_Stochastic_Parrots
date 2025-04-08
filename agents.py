from clients import Retries, ModelClient
from abc import abstractmethod
import re
from typing import List, Literal, Tuple
import subprocess


class Agent(Retries):
    def __init__(self, model_client: ModelClient, max_retries: int = 3):
        super().__init__(max_retries=max_retries)
        self.model_client = model_client
        self.agent_name = "agent"

    @abstractmethod
    def forward(self, *inputs):
        """Process inputs and produce output according to agent's role."""
        pass

    def _extract_tag(self, prompt: str, tag_name: str) -> str:
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

    def _query_and_extract(self, prompt: str, tag_name: str) -> str:
        """Prompt model, extract tag from models respone, re-querying model if tag extraction fails."""

        def helper(prompt_arg: str, tag_name_arg: str) -> str:
            response = self.model_client.query(prompt_arg)
            extracted = self._extract_tag(response, tag_name_arg)
            if extracted is None:
                raise ValueError(
                    f"Failed to extract '{tag_name_arg}' tag/block from response"
                )
            return extracted

        return self._func_with_retries(helper, prompt, tag_name)


class AgentBasic(Agent):
    def __init__(self, model_client: ModelClient, max_retries: int = 3):
        super().__init__(model_client=model_client, max_retries=max_retries)
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


class AgentFileSelector(Agent):
    def __init__(
        self,
        model_client: ModelClient,
        return_full_text: bool,
        strip_line_num: bool,
        max_retries: int = 3,
    ):
        super().__init__(model_client=model_client, max_retries=max_retries)
        self.agent_name = "agent_file_selector"
        self.return_full_text = return_full_text
        self.strip_line_num = strip_line_num

    def forward(
        self,
        text: str,
        method: Literal["batch", "individual"],
        custom_issue: str = None,
    ) -> Tuple[List[str], str]:
        """AgentFileSelector selects which files to pass on based on if they are relevant to the current issue/errors."""

        issue_text = custom_issue or self._extract_tag(text, "issue")
        files_text = self._extract_tag(text, "code")

        if issue_text is None:
            raise ValueError("Could not extract <issue> tag from input text.")
        if files_text is None:
            raise ValueError("Could not extract <code> tag from input text.")

        if method == "batch":  # pass all files to the model
            selected_file_paths = self._func_with_retries(
                self._func_with_retries(self._select_by_batch, issue_text, files_text)
            )
        elif method == "individual":  # pass each file to the model one by one
            selected_file_paths = self._func_with_retries(
                self._func_with_retries(
                    self._select_by_individual, issue_text, files_text
                )
            )
        else:
            raise ValueError(
                f"Invalid method: {method}. Expected 'batch' or 'individual'."
            )

        return selected_file_paths, self._format_output(
            text, files_text, selected_file_paths
        )

    def _select_by_batch(self, issue_text: str, files_text: str) -> List[str]:
        """Select files by passing all files to the model."""

        prompt = f"""Your task is to select the files that are relevant to 
        solving the issue, this will be passed on to another model which 
        will use these as context to solve the issue:
        \n<issue>\n{issue_text}\n</issue>
        \n<files>\n{files_text}\n</files>
        \nFeel free to reason about which files to select. At the end provide 
        a list of selected files with paths (exactly as they appear above). 
        Return your list of selected files in either <selected> </selected> 
        tags or in a ```selected ``` block.
        \nHere is an example:
        \n<selected>\n
        astropy/time/formats.py\n
        astropy/coordinates/distances.py\n
        docs/conf.py\n
        </selected>"""

        selected = self._query_and_extract(prompt, "selected")

        # check selected files exist in issue_text
        files_dict = self._get_files(files_text)
        for file_path in selected.splitlines():
            if file_path not in files_dict:
                raise ValueError(
                    f"Selected file '{file_path}' does not exist in the provided files."
                )

        return selected.splitlines()

    def _select_by_individual(self, issue_text: str, files_text: str) -> List[str]:
        """Select files by passing each file to the model one by one."""

        selected_file_paths = []
        files_dict = self._get_files(files_text)

        for file_path, file_content in files_dict.items():
            prompt = f"""You will bed given files one by one. Your task is to 
            determine if the following file is relevant to solving the issue, 
            this will be passed on to another model which will use this as 
            context to solve the issue:
            \n<issue>\n{issue_text}\n</issue>
            \nHere is {file_path}:
            \n<file>\n{file_content}\n</file>
            \nFeel free to reason about whether to select the file or not. At 
            the end provide either a `Yes` or `No` answer in either <selected> 
            </selected> tags or in a ```selected ``` block.
            \nHere is an example:
            \n<selected>\n
            Yes
            </selected>"""

            selected = self._query_and_extract(prompt, "selected")

            if selected.lower() == "yes":
                selected_file_paths.append(file_path)
            elif selected.lower() != "no":
                raise ValueError(
                    f"Invalid response from model: {selected}. Expected 'Yes' or 'No'."
                )

        return selected_file_paths

    def _get_files(self, files_text: str) -> dict:
        # Regex to find all file blocks
        # \[start of (.*?)\]  -> Capture [start of example/test.py]
        # (.*?)               -> Capture the file content (non-greedy)
        # \[end of \1\]       -> Match [end of example/test.py]
        pattern = r"\[start of (.*?)\]\s*\n(.*?)\n\[end of \1\]"
        matches = re.findall(pattern, files_text, re.DOTALL)

        files_dict = {}
        for match in matches:
            file_path = match[0].strip()
            original_file_content = match[1]

            if self.strip_line_num:
                lines = original_file_content.splitlines()
                cleaned_lines = []
                for line in lines:
                    # remove leading digits and the first optional space
                    cleaned_line = re.sub(r"^\d+\s?", "", line)
                    cleaned_lines.append(cleaned_line)
                processed_content = "\n".join(cleaned_lines)
            else:
                processed_content = original_file_content

            files_dict[file_path] = processed_content

        return files_dict

    def _format_output(
        self, text: str, files_text: str, selected_file_paths: List[str]
    ) -> str:
        """Reconstructs the text within <code> tags to only include files specified in selected_files_paths."""
        # similar to regex in self._get_files()
        # Group 1: The entire block
        # Group 2: The file path within the start marker
        file_block_pattern = r"(\[start of (.*?)\]\s*?\n.*?\n\[end of \2\])"
        all_file_blocks = re.findall(file_block_pattern, files_text, re.DOTALL)

        # only keep blocks whose paths are in selected_files_paths
        selected_blocks_text = []
        selected_files_set = set(selected_file_paths)

        for full_block, file_path in all_file_blocks:
            if file_path.strip() in selected_files_set:
                selected_blocks_text.append(full_block)

        reconstructed_code_content = "\n".join(selected_blocks_text)

        if not self.return_full_text:
            if reconstructed_code_content:
                return f"<code>\n{reconstructed_code_content}\n</code>"
            else:
                return "<code>\n</code>"

        # regex for <code> </code> block in the full_text
        # Group 1: Content before <code>
        # Group 2: Content inside <code> (to replace)
        # Group 3: Content after </code>
        code_section_pattern = r"(.*?<code>\s*\n?)(.*?)(\n?\s*</code>.*)"
        match = re.search(code_section_pattern, text, re.DOTALL | re.IGNORECASE)

        if not match:
            raise ValueError(
                "Could not find the <code> block in the original text for replacement."
            )

        prefix = match.group(1)
        suffix = match.group(3)

        if reconstructed_code_content:
            final_text = (
                f"{prefix.rstrip()}\\n{reconstructed_code_content}\\n{suffix.lstrip()}"
            )
            # clean up possible extra \n around the replaced <code>
            final_text = final_text.replace(
                prefix.rstrip() + "\n\n", prefix.rstrip() + "\n"
            )
            final_text = final_text.replace(
                "\n\n" + suffix.lstrip(), "\n" + suffix.lstrip()
            )
        else:
            # case where no files are selected (empty <code> block)
            final_text = f"{prefix.rstrip()}\\n{suffix.lstrip()}"

        return final_text


class AgentFileRetriever(Agent):
    def __init__(self, model_client: ModelClient, max_retries: int = 3):
        super().__init__(model_client=model_client, max_retries=max_retries)
        self.agent_name = "agent_file_retriever"

    def forward(
        self, issue: str, base_commit: str, environment_setup_commmit: str
    ) -> str:
        """AgentFileRetriever fetches files from the given codebase based on the relevancy of the issue/errors."""
        raise NotImplementedError


class AgentExampleRetriever(Agent):
    def __init__(self, model_client: ModelClient, max_retries: int = 3):
        super().__init__(model_client=model_client, max_retries=max_retries)
        self.agent_name = "agent_example_retriever"

    def forward(self, issue: str, num_retrieve: int, num_select: int) -> str:
        """AgentExampleRetriever fetches examples via RAG of stack exchange solutions to questions that are similar to the current issue/errors."""
        raise NotImplementedError


class AgentProgrammer(Agent):
    def __init__(self, model_client: ModelClient, max_retries: int = 3):
        super().__init__(model_client=model_client, max_retries=max_retries)
        self.agent_name = "agent_programmer"

    def forward(
        self,
        prompt: str,
        method: Literal["batch", "individual"],
    ) -> str:
        """AgentProgrammer regenerates (fully) files with bugs in them, then generates a patch via a diff between the old and new file"""

        # TODO: Implement

        cache_dir = "agent_cache"

        # could use AgentFileSelector with return_full_text=False, strip_line_num=True,

        def helper():
            pass

        return self._func_with_retries(helper, prompt)

    def _generate_patch(file1: str, file2: str):
        """Generate a patch between two files using the diff command."""

        # diff command to generate a patch
        result = subprocess.run(
            ["diff", "-u", file1, file2],
            capture_output=True,
            text=True,
        )

        # ensure diff applied to original code results in the new code:
        diff_result = subprocess.run(
            ["git", "apply", "--check", "-"],
            input=result.stdout,
            capture_output=True,
            text=True,
        )

        if diff_result.returncode != 0:
            raise RuntimeError(f"Diff check failed: {diff_result.stderr}")

        return result.stdout
