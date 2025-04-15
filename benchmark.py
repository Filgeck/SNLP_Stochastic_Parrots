import json
from pathlib import Path
from typing import Callable
from tqdm.auto import tqdm
from datasets import load_dataset, DatasetDict
from agents import Agent, AgentBasic, AgentProgrammer, AgentMulti
from clients import ModelClient
import traceback
import subprocess
import sys

SWE_BENCH_BM25_40K_DATASET = "princeton-nlp/SWE-bench_bm25_40K"
SWE_BENCH_LITE_DATASET = "princeton-nlp/SWE-bench_Lite"
SWE_BENCH_RUN_EVAL_PATH = Path("swebench/swebench/harness/run_evaluation.py")
PREDICTIONS_DIR_PATH = Path("predictions")


class AgentBenchmark:
    def __init__(self, agent: Agent):
        model_name = agent.model_client.model_name.replace("/", "_")
        output_path = Path(
            f"{PREDICTIONS_DIR_PATH}/{agent.agent_name}/{model_name}".replace(
                ":", "-"
            )
        )

        self.agent = agent
        self.preds_file_path = output_path / "all_pred.jsonl"
        self.report_dir_path = output_path / "logs"
        self.run_id = f"{agent.agent_name}_{agent.model_client.model_name}"

    def generate_preds_custom_retrieval(
        self, benchmark_dataset: str, retrieval_func: Callable
    ) -> None:
        """Run benchmark for a given agent and swe-bench dataset, with a custom retrieval method."""
        pass

    def generate_preds_precomputed_retrieval(
        self, benchmark_dataset: str, retrieval_dataset: str
    ) -> None:
        """Run benchmark for a given agent and swe-bench dataset, with a precomputed retrieval dataset."""
        benchmark = load_dataset(benchmark_dataset)
        # benchmark & retrieval must be a DatasetDict since split is None and
        # streaming is False
        assert isinstance(benchmark, DatasetDict)
        retrieval = load_dataset(retrieval_dataset)
        assert isinstance(retrieval, DatasetDict)
        processed_ids = self._find_processed_ids()

        # map each benchmark instance to the retrieved (precomputed RAG) documents
        retrieval_map = {
            item["instance_id"]: item["text"]
            for item in tqdm(
                retrieval["test"],
                desc=f"Retrieving {retrieval_dataset}:",
            )
        }

        self.preds_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.preds_file_path.touch(exist_ok=True)
        with open(self.preds_file_path, "a") as file_out:
            for instance in tqdm(
                benchmark["test"],
                desc=f"Running {benchmark_dataset} benchmark",
            ):
                instance_id = instance["instance_id"]

                if (instance_id, self.run_id) in processed_ids:
                    print("Skipping already processed instance:", instance_id)
                    continue

                prompt = retrieval_map[instance_id]
                # prompt += """\n\nFeel free to analyse and edit files as required, however you must absolutely ensure that at the end of your response you enclose your final patch in either <patch> </patch> tags or a ```patch ``` block."""
                output = self.agent.forward(prompt)

                prediction = {
                    "instance_id": instance_id,
                    "model_patch": output,
                    "model_name_or_path": self.run_id,
                }

                file_out.write(json.dumps(prediction) + "\n")
                file_out.flush()

        self._sort_jsonl_alphanumeric_asc(
            self.preds_file_path,
        )

    def run_benchmark(
        self,
        max_workers: int,
    ) -> None:
        # self.preds_file_path
        # self.run_id

        if (
            not self.preds_file_path.exists()
            or self.preds_file_path.stat().st_size == 0
        ):
            raise FileNotFoundError(
                f"Predictions file {self.preds_file_path} does not exist or is empty."
            )

        if not SWE_BENCH_RUN_EVAL_PATH.exists():
            raise FileNotFoundError(
                "SWE-bench run script not found. Please ensure the path is correct and you run this script from the repo root."
            )

        # check logs directory exists, if not create
        self.report_dir_path.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(SWE_BENCH_RUN_EVAL_PATH.resolve()),
            "--predictions_path",
            str(self.preds_file_path.resolve()),
            "--max_workers",
            str(max_workers),
            "--report_dir",
            str(self.report_dir_path.resolve()),
            "--run_id",
            self.run_id,
            "--namespace",
            "",
            "--force_rebuild",
            "True",
        ]

        print("\nExecuting swe-bench command:\n", " ".join(command))

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            )

            while True:
                output = process.stdout.readline() if process.stdout else ""
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            rc = process.poll()
            if rc == 0:
                print(
                    f"SWE-bench evaluation completed successfully for run_id '{self.run_id}'."
                )
                print(f"Logs and report stored under: {self.report_dir_path}")
            else:
                print(
                    f"ERROR: SWE-bench evaluation failed with exit code {rc} for run_id"
                )
        except Exception as error:
            print(
                f"ERROR: An unexpected error occurred while running the evaluation script: {error}"
            )
            traceback.print_exc()

    def _sort_jsonl_alphanumeric_asc(self, jsonl_file: Path) -> None:
        """Sort a JSONL file by the 'instance_id' field in ascending order."""
        with open(jsonl_file, "r") as file:
            lines = [json.loads(line) for line in file]

        sorted_lines = sorted(lines, key=lambda x: x["instance_id"])

        with open(jsonl_file, "w") as file:
            for line in sorted_lines:
                file.write(json.dumps(line) + "\n")

    def _find_processed_ids(self) -> set:
        """For this given benchmark, return the instance_ids that have already been evaluated."""
        processed_ids = set()
        try:
            if self.preds_file_path.exists():
                with open(self.preds_file_path, "r") as file:
                    existing_preds = [json.loads(line) for line in file]
                processed_ids = {
                    (p["instance_id"], p["model_name_or_path"])
                    for p in existing_preds
                }
            else:
                return set()
        except Exception as e:
            raise RuntimeError(f"Error reading {self.preds_file_path}:\n{e}")
        return processed_ids

# def benchmark_temperature(
#     model_name: str, max_temp: int = 2, step: float = 0.1
# ) -> None:
#     scale = int(1 / step)
#     for i in range(0, max_temp * scale, scale):
#         temp = i / scale
#         model = ModelClient(model_name, max_retries=1)

#         AGENTS_TO_BENCHMARK = [
#             # AgentBasic(model, max_retries=1, param_count=param_count),
#             AgentProgrammer(
#                 model, max_retries=10
#             ),
#             # AgentMulti(model, max_retries=10, param_count=param_count),
#         ]
#         for agent in AGENTS_TO_BENCHMARK:
#             print("Benchmarking agent:", agent.agent_name)
#             benchmark = AgentBenchmark(agent, temp=temp)
#             benchmark.generate_preds_precomputed_retrieval(
#                 SWE_BENCH_LITE_DATASET, SWE_BENCH_BM25_40K_DATASET
#             )
#             benchmark.run_benchmark(max_workers=16)

def benchmark_deepseek_params() -> None:
    params_to_models = {
        # "1.5B": "deepseek/deepseek-r1-distill-qwen-1.5b", # Not enough - gets into loops of it talking to itself
        # "8B": "deepseek/deepseek-r1-distill-llama-8b", # Only has 32k context
        "14B": "deepseek/deepseek-r1-distill-qwen-14b",
        "32B": "deepseek/deepseek-r1-distill-qwen-32b",
        "70B": "deepseek/deepseek-r1-distill-llama-70b",
        "671B": "deepseek/deepseek-r1",
    }
    try:
        for param_count, model_name in params_to_models.items():
            print("Benchmarking model:", model_name)
            print("Param count:", param_count)
            model = ModelClient(model_name, max_retries=1)

            AGENTS_TO_BENCHMARK = [
                # AgentBasic(model, max_retries=1, param_count=param_count),
                AgentProgrammer(
                    model, max_retries=10, param_count=param_count
                ),
                # AgentMulti(model, max_retries=10, param_count=param_count),
            ]


            for agent in AGENTS_TO_BENCHMARK:
                print("Benchmarking agent:", agent.agent_name)
                benchmark = AgentBenchmark(agent)
                try:
                    benchmark.generate_preds_precomputed_retrieval(
                        SWE_BENCH_LITE_DATASET, SWE_BENCH_BM25_40K_DATASET
                    )
                except Exception:
                    traceback.print_exc()
                    print("Skipping agent:", agent.agent_name, model_name)
                    continue
                benchmark.run_benchmark(max_workers=6)

    except KeyboardInterrupt:
        print("Benchmarking interrupted by user.")

if __name__ == "__main__":
    benchmark_deepseek_params()

# if __name__ == "__main__":
#     MODELS_TO_BENCHMARK = [
#         # "anthropic/claude-3.7-sonnet",
#         # "deepseek/deepseek-r1-zero:free",
#         # "x-ai/grok-3-beta",
#         # "deepseek-r1-8b",
#         # "llama3.2",
#         "gemini-2.5-pro-exp-03-25",
#     ]

#     try:
#         for model_name in MODELS_TO_BENCHMARK:
#             model = ModelClient(model_name)

#             AGENTS_TO_BENCHMARK = [
#                 # AgentBasic(model, max_retries=10),
#                 AgentProgrammer(model, max_retries=10),
#                 AgentMulti(model, max_retries=10),
#             ]

#             for agent in AGENTS_TO_BENCHMARK:
#                 benchmark = AgentBenchmark(agent)
#                 benchmark.generate_preds_precomputed_retrieval(
#                     SWE_BENCH_LITE_DATASET, SWE_BENCH_BM25_40K_DATASET
#                 )
#                 benchmark.run_benchmark(max_workers=16)

#     except KeyboardInterrupt:
#         print("Benchmarking interrupted by user.")
