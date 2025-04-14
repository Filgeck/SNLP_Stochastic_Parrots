import ast
from collections import deque
from pathlib import Path
from pydantic import BaseModel, Field
import subprocess
from typing import Dict, List, Optional, Set, Tuple

from agents.base import Agent
from clients import ModelClient

# Prompts
DESCRIBE_ISSUE = (
    "You are an expert senior developer who is tasked with solving github issues. "
    "You will be provided with a Github issue, and your job is to identify the file "
    "where the error takes place. If you don't know the answer, please respond with "
    "the string '<no-file>'.\n"
    "Please ensure that you provide the full file path (where the repo directory is '/')"
)

# Output schema
class StartingPoint(BaseModel):
    starting_filename: str = Field(description="The filename in which the error occured")


class AgentFileRetriever(Agent):
    def __init__(
        self,
        model_client: ModelClient,
        strip_line_num: bool,
        max_retries: int = 3,
        return_full_text: bool = True
    ):
        super().__init__(model_client=model_client, max_retries=max_retries)
        self.agent_name = "agent_file_retriever"
        self.strip_line_num = strip_line_num
        self.return_full_text = return_full_text


    def forward(self, issue: str, repo_name: str, base_commit: str, n: int = 10) -> Tuple[List[Path], str]:
        """
        Fetches files from the given codebase via import BFS

        args:
        - issue: the text of the Github issue
        - repo_name: the name of the Github repo
        - base_commit: the commit hash where the issue is found
        - n: the number of files to retrieve via BFS

        returns:
        A tuple containing
        1. The file paths of the relevant files
        2. The string containing the issue and the code, but only with the relevant files
        """
        global DESCRIBE_ISSUE

        repo_cache = Path("agent_cache") / "agent_file_retriever"

        if not repo_cache.exists():
            repo_cache.mkdir()

        repo_path = repo_cache / repo_name

        self._checkout_repo_commit(repo_name, repo_path, base_commit)

        ig = ImportGraph(repo_path)

        # Get starting filename
        prompt = f"{DESCRIBE_ISSUE}\n<issue>\n{issue}\n</issue>"
        output = self.model_client.query(prompt, structure=StartingPoint)
        parsed_output = StartingPoint.model_validate_json(output)

        # length-n BFS
        starting_filename = parsed_output.starting_filename

        if starting_filename == "<no-file>":
            raise NotImplementedError("Haven't handled case for no file given")

        starting_module = ig.get_module_name(repo_path / starting_filename)
        relevant_modules = ig.bfs(starting_module, n)

        # Get file paths and contents
        file_paths = []
        file_contents = []

        for module in relevant_modules:
            paths = ig.get_module_files(module)
            contents = [p.read_text(encoding="utf-8", errors="ignore") for p in paths]

            file_paths.extend(paths)
            file_contents.extend(contents)

        file_pairs = [(str(p), c) for p, c in zip(file_paths, file_contents)]
        out_str = self.build_tagged_string(issue, file_pairs)

        return file_paths, out_str


    def _checkout_repo_commit(self, repo_name: str, repo_path: Path, commit_hash: str):
        """
        Clones the repo and checks out the specific commit
        """
        repo_url = f"https://github.com/{repo_name}.git"

        if not repo_path.exists():
            subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)

        subprocess.run(["git", "checkout", commit_hash], cwd=repo_path)


    def _get_repository_files(self, repo_path: Path) -> list[str]:
        """
        Returns the list of the names of all files in the repository

        args:
         - repo_path: the path of the local git repo
        """
        result = subprocess.run(["git", "ls-files"], cwd=repo_path,
                                capture_output=True, text=True, check=True)
        files = result.stdout.strip().splitlines()

        return files


    def _build_output_string(self, issue: str, file_paths: List[str], repo_path: Path) -> str:
        """
        Builds an XML-tagged output string with the contents of the supplied files
        """
        file_contents = []

        for relative_path in file_paths:
            full_path = repo_path / relative_path
            file_contents.append((relative_path, full_path.read_text(encoding="utf-8", errors="ignore")))

        return self.build_tagged_string(issue, file_contents)


# Import exploration
class ImportGraph:
    def __init__(self, repo_root: Path):
        """
        Build an import graph for all Python files under repo_root, but only include modules that
        are part of the project (i.e. in repo_root, and not in virtual environments).
        """
        self.repo_root = repo_root.resolve()

        self.module_mapping: Dict[str, Path] = {}  # Maps module names -> file paths
        self.graph: Dict[str, List[str]] = {}      # Maps module names to list of imported module names
        self.build_graph()


    def get_module_name(self, file_path: Path) -> str:
        """
        Convert a file path to a dotted module name based on its path relative to repo_root.

        For example:
          repo_root/mypackage/module.py  -> "mypackage.module"
          repo_root/mypackage/__init__.py -> "mypackage"
        """
        try:
            rel_path = file_path.relative_to(self.repo_root)
        except ValueError:
            return ""

        parts = list(rel_path.with_suffix("").parts)

        # If the file is named __init__.py, the module name is the package name (drop the '__init__')
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]

        return ".".join(parts)


    def resolve_relative_import(self, current_module: str, level: int, module: Optional[str]) -> str:
        """
        Given the current module's dotted name, the relative import level,
        and the imported module (which may be empty), return the absolute module name.
        """
        parts = current_module.split(".")

        if level > len(parts):
            return ""

        base = parts[:-level]

        if module:
            base.extend(module.split("."))

        return ".".join(base)


    def get_imports_for_file(self, file_path: Path) -> List[str]:
        """
        Iteratively parse a Python file and return a list of imported module names.
        Uses an iterative AST traversal to avoid recursion limits.
        """
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        try:
            tree = ast.parse(source, filename=str(file_path))
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []

        imports: List[str] = []
        stack: List[ast.AST] = [tree]

        while stack:
            node = stack.pop()

            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    if node.module:
                        imports.append(node.module)
                else:
                    # Record relative import: e.g. "..module" (if module is None, then just dots)
                    dots = "." * node.level
                    mod = node.module if node.module else ""
                    imports.append(dots + mod)

            # Add children to stack.
            stack.extend(ast.iter_child_nodes(node))

        return imports


    def build_graph(self):
        """
        Walk all Python files under repo_root (excluding those in directories with "venv" in their path),
        build a mapping of module names, and construct the import graph for project code only.
        """
        # Build module mapping from repo_root: scan all *.py files, excluding those in virtual env folders.
        for file in self.repo_root.rglob("*.py"):
            # Skip files in virtual environment directories.
            if any(venv in file.parts for venv in ["venv", ".venv", "env"]):
                continue
            mod_name = self.get_module_name(file)
            if mod_name:
                self.module_mapping[mod_name] = file.resolve()

        # Now, construct the graph.
        for mod_name, file_path in self.module_mapping.items():
            raw_imports = self.get_imports_for_file(file_path)
            resolved_imports: List[str] = []

            for imp in raw_imports:
                if imp.startswith("."):
                    # Resolve relative import.
                    level = 0
                    while level < len(imp) and imp[level] == '.':
                        level += 1

                    mod_part = imp[level:]
                    absolute = self.resolve_relative_import(mod_name, level, mod_part)

                    # Only add if the absolute module is in our module mapping (project code).
                    if absolute and absolute in self.module_mapping:
                        resolved_imports.append(absolute)
                else:
                    # For absolute imports, only add if it exists in our module mapping.
                    if imp in self.module_mapping:
                        resolved_imports.append(imp)

            self.graph[mod_name] = resolved_imports


    def bfs(self, start_module: str, n: int) -> List[str]:
        """
        Perform BFS on the import graph starting from start_module,
        collecting up to n modules in the project that are reachable.
        """
        queue = deque([start_module])
        visited: Set[str] = {start_module}
        result: List[str] = []

        while queue and len(result) < n:
            current = queue.popleft()

            for neighbor in self.graph.get(current, []):
                if neighbor in self.graph and neighbor not in visited:
                    visited.add(neighbor)
                    result.append(neighbor)
                    queue.append(neighbor)

                    if len(result) >= n:
                        break

        return result


    def get_module_files(self, module_name: str) -> List[Path]:
        """
        Returns all file paths for a given (local) module

        args:
        - module_name: the name of the module
        """
        paths = []

        for m_name, file_path in self.module_mapping.items():
            if m_name == module_name or m_name.startswith(module_name + "."):
                paths.append(file_path)

