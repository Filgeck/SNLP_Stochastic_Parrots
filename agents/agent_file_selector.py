import re
from typing import List, Literal, Tuple

from agents.base import Agent
from clients import ModelClient


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
        self, text: str, method: Literal["batch", "individual"], custom_issue: str | None = None,
    ) -> Tuple[List[str], str]:
        """
        selects which files to pass on based on if they are relevant to the current issue/errors.

        args:
        - text: the full issues/code text block
        - method: whether to pass each file to the model in a block or sequentially
        - custom_issue: a custom issue not in the text

        returns:
        A tuple containing
        1. The file paths of the relevant files
        2. The string containing the issue and the code, but only with the relevant files
        """

        issue_text = custom_issue or self._extract_tag(text, "issue")
        files_text = self._extract_tag(text, "code")

        if issue_text is None:
            raise ValueError("Could not extract <issue> tag from input text.")
        if files_text is None:
            raise ValueError("Could not extract <code> tag from input text.")

        # Get files dictionary with line numbers stripped if needed
        files_dict = self._get_files(files_text, self.strip_line_num)

        if method == "batch":  # pass all files to the model
            selected_file_paths = self._func_with_retries(
                self._select_by_batch, issue_text, files_text
            )
        elif method == "individual":  # pass each file to the model one by one
            selected_file_paths = self._func_with_retries(
                self._select_by_individual, issue_text, files_text
            )
        else:
            raise ValueError(
                f"Invalid method: {method}. Expected 'batch' or 'individual'."
            )

        return selected_file_paths, self._format_output(
            text, files_text, selected_file_paths, files_dict
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
        files_dict = self._get_files(files_text, self.strip_line_num)
        for file_path in selected.splitlines():
            if file_path not in files_dict:
                raise ValueError(
                    f"Selected file '{file_path}' does not exist in the provided files."
                )

        return selected.splitlines()

    def _select_by_individual(self, issue_text: str, files_text: str) -> List[str]:
        """Select files by passing each file to the model one by one."""

        selected_file_paths = []
        files_dict = self._get_files(files_text, self.strip_line_num)

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
    
    def remove_line_numbers(self, text: str) -> str:
        lines = text.split('\n')
        new_lines = []
        for line in lines:
            new_line = re.sub(r'^\d+\s?', '', line)
            new_lines.append(new_line)
        return '\n'.join(new_lines)

    def _get_files(self, files_text: str, strip_line_num: bool) -> dict:
        """
        Extract file content from a string containing file blocks marked with [start of filename] and [end of filename].
        
        Args:
            files_text: String containing file blocks
            strip_line_num: Whether to strip line numbers from the file content
            
        Returns:
            Dictionary mapping file paths to file contents
        """
        # Regex to find all file blocks
        # \[start of (.*?)\] -> Capture [start of example/test.py]
        # (.*?) -> Capture the file content (non-greedy)
        # \[end of \1\] -> Match [end of example/test.py]
        pattern = r"\[start of (.*?)\](.*?)\[end of \1\]"
        matches = re.findall(pattern, files_text, re.DOTALL)
        files_dict = {}
        for match in matches:
            file_path = match[0].strip()
            file_content = match[1].strip()
            files_dict[file_path] = file_content
        if strip_line_num:
            for file_path, file_content in files_dict.items():
                files_dict[file_path] = self.remove_line_numbers(file_content)
        return files_dict

    def _format_output(
        self, text: str, files_text: str, selected_file_paths: List[str], files_dict: dict
    ) -> str:
        """Reconstructs the text within <code> tags to only include files specified in selected_files_paths.
        Uses the processed files_dict to ensure line numbers are properly handled."""
        
        # only keep blocks whose paths are in selected_files_paths
        selected_blocks_text = []
        selected_files_set = set(selected_file_paths)

        # Use the processed files_dict to reconstruct the file blocks
        for file_path in selected_files_set:
            if file_path in files_dict:
                file_content = files_dict[file_path]
                # Reconstruct the file block with the processed content
                block = f"[start of {file_path}]\n{file_content}\n[end of {file_path}]"
                selected_blocks_text.append(block)

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
            final_text = f"{prefix.rstrip()}\n{reconstructed_code_content}\n{suffix.lstrip()}"
            
            # Clean up potential double newlines around the inserted content
            final_text = final_text.replace(f"{prefix.rstrip()}\n\n", f"{prefix.rstrip()}\n")
            final_text = final_text.replace(f"\n\n{suffix.lstrip()}", f"\n{suffix.lstrip()}")
        else:
            # Case where no files are selected (empty <code> block)
            final_text = f"{prefix.rstrip()}\n{suffix.lstrip()}"


        return final_text