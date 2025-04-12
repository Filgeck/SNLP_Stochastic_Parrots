from clients import Retries, ModelClient, RagClient
from abc import abstractmethod
import re
from typing import List, Literal, Tuple
import subprocess
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


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
        custom_issue: str | None = None,
    ) -> Tuple[List[str], str]:
        """AgentFileSelector selects which files to pass on based on if they are relevant to the current issue/errors."""

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


class AgentProgrammer(Agent):
    def __init__(self, model_client: ModelClient, max_retries: int = 3):
        super().__init__(model_client=model_client, max_retries=max_retries)
        self.agent_name = "agent_programmer"

    def forward(
        self,
        prompt: str,
    ) -> str | None:
        """AgentProgrammer regenerates (fully) files with bugs in them, then generates a patch via a diff between the old and new file"""

        files_dict = self._get_files(prompt, False)

        clean_prompt = prompt

        clean_prompt += "\n\nPlease fix the bugs in the files and return the full fixed files in this format:\n\n" \
            "[start of FILEPATH] abcdef\n[end of FILEPATH]\n" \
            "[start of FILEPATH] abcdef\n[end of FILEPATH]\n" \
            " etc make sure to write out the full file, not just the changes. And do not write line numbers!\n\n"
        
        response = self.model_client.query(clean_prompt)
        
        changed_files = self._get_files(response, False)

        patch = self.create_patch_from_files(files_dict, changed_files, "agent_cache", cleanup=True)

        return patch

        # return self._func_with_retries(helper, prompt)
    
    def _get_files(self, files_text: str, strip_line_num: bool) -> dict:
        """
        Extract file content from a string containing file blocks marked with [start of filename] and [end of filename].
        
        Args:
            files_text: String containing file blocks
            
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
        return files_dict

    def _generate_patch(self, file1: str, file2: str):
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

    
    def create_patch_from_files(self, files_dict: dict, changed_files: dict, cache_dir: str, cleanup=True) -> str:
        """
        Save original and fixed files in the cache directory, generate a patch, and optionally clean up.
        Args:
            files_dict: Dictionary of original files (path -> content)
            changed_files: Dictionary of fixed files (path -> content)
            cache_dir: Directory to save the files
            cleanup: Whether to remove temporary files after generating the patch (default: True)
        Returns:
            Generated patch as a string
        """
        import os
        import subprocess
        
        modified_files = []
        temp_files = {}

        try:
            # Create a temporary directory for our files if it doesn’t exist
            os.makedirs(cache_dir, exist_ok=True)

            # Check *all* file paths we know about (union of original & changed)
            for file_path in set(files_dict.keys()) | set(changed_files.keys()):
                original_content = files_dict.get(file_path, "")
                fixed_content = changed_files.get(file_path, "")

                # -------------------------------------------------------------
                # KEY CHANGE: Treat an empty "fixed_content" as "no change."
                # Also skip if the file is unchanged.
                # -------------------------------------------------------------
                if (fixed_content.strip() == "" 
                    or fixed_content == original_content):
                    # This means "no effective changes" => skip
                    continue

                # Otherwise, we do have changes, so create temp files to diff
                orig_file = os.path.join(cache_dir, f"orig_{os.path.basename(file_path)}")
                fixed_file = os.path.join(cache_dir, f"fixed_{os.path.basename(file_path)}")

                with open(orig_file, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                with open(fixed_file, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)

                temp_files[file_path] = (orig_file, fixed_file)
                modified_files.append(file_path)

            all_patches = []
            for file_path in modified_files:
                orig_file, fixed_file = temp_files[file_path]

                # Run diff and capture output
                result = subprocess.run(
                    ["diff", "-u", orig_file, fixed_file],
                    capture_output=True,
                    text=True
                )

                # diff returns 1 if files differ, which we do expect
                # but anything else is a real error
                if result.returncode not in [0, 1]:
                    raise RuntimeError(f"Diff failed: {result.stderr}")

                diff_output = result.stdout.splitlines()

                # Adjust the headers so they show the actual file paths
                if len(diff_output) >= 2:
                    diff_output[0] = f"--- {file_path}"
                    diff_output[1] = f"+++ {file_path}"

                all_patches.append("\n".join(diff_output))

            print(f"Patched {len(all_patches)} files")
            return "\n".join(all_patches)

        finally:
            # Clean up
            if cleanup:
                for orig_file, fixed_file in temp_files.values():
                    if os.path.exists(orig_file):
                        os.remove(orig_file)
                    if os.path.exists(fixed_file):
                        os.remove(fixed_file)
