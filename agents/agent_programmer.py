import re
import subprocess

from agents.base import Agent
from clients import ModelClient


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

        clean_prompt += (
            "\n\nPlease fix the bugs in the files and return the full fixed files in this format:\n\n"
            "[start of FILEPATH] abcdef\n[end of FILEPATH]\n"
            "[start of FILEPATH] abcdef\n[end of FILEPATH]\n"
            " etc make sure to write out the full file, not just the changes. And do not write line numbers!\n\n"
        )

        response = self.model_client.query(clean_prompt)

        changed_files = self._get_files(response, False)

        patch = self.create_patch_from_files(
            files_dict, changed_files, "agent_cache", cleanup=True
        )

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

    def create_patch_from_files(
        self, files_dict: dict, changed_files: dict, cache_dir: str, cleanup=True
    ) -> str:
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
                if fixed_content.strip() == "" or fixed_content == original_content:
                    # This means "no effective changes" => skip
                    continue

                # Otherwise, we do have changes, so create temp files to diff
                orig_file = os.path.join(
                    cache_dir, f"orig_{os.path.basename(file_path)}"
                )
                fixed_file = os.path.join(
                    cache_dir, f"fixed_{os.path.basename(file_path)}"
                )

                with open(orig_file, "w", encoding="utf-8") as f:
                    f.write(original_content)
                with open(fixed_file, "w", encoding="utf-8") as f:
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
                    text=True,
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
