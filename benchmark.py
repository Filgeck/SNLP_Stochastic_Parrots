import json
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_dataset
from agents import Agent, AgentBasic
from clients import ModelClient

SWE_BENCH_BM25_40K_DATASET = "princeton-nlp/SWE-bench_bm25_40K"
SWE_BENCH_LITE_DATASET = "princeton-nlp/SWE-bench_Lite"


class AgentBenchmark:
    def __init__(self, model_name: str, agent: Agent):
        self.model_name = model_name
        self.agent = agent
        self.preds_file = Path(
            f"predictions/{agent.agent_name}/{model_name}/all_pred.jsonl".replace(":", "-")
        )
        self.benchmark_name = f"{agent.agent_name}_{agent.model_client.model_name}"

    def run_benchmark_custom_retrieval(
        self, benchmark_dataset: str, retrieval_func: callable
    ) -> None:
        """Run benchmark for a given agent and swe-bench dataset, with a custom retrieval method."""
        pass

    def run_benchmark_precomputed_retrieval(
        self, benchmark_dataset: str, retrieval_dataset: str
    ) -> None:
        """Run benchmark for a given agent and swe-bench dataset, with a precomputed retrieval dataset."""
        benchmark = load_dataset(benchmark_dataset)
        retrieval = load_dataset(retrieval_dataset)
        processed_ids = self._find_processed_ids()

        # map each benchmark instance to the retrieved (precomputed RAG) documents
        retrieval_map = {
            item["instance_id"]: item["text"]
            for item in tqdm(
                retrieval["test"],
                desc=f"Retrieving {retrieval_dataset}:",
            )
        }

        self.preds_file.parent.mkdir(parents=True, exist_ok=True)
        self.preds_file.touch(exist_ok=True)
        with open(self.preds_file, "a") as file_out:
            for instance in tqdm(
                benchmark["test"],
                desc=f"Running {benchmark_dataset} benchmark",
            ):
                instance_id = instance["instance_id"]

                if (instance_id, self.benchmark_name) in processed_ids:
                    print("Skipping already processed instance:", instance_id)
                    continue

                prompt = retrieval_map.get(instance_id)
                prompt += """\n\nFeel free to analyse and edit files as required, however you must absolutely ensure that at the end of your response you enclose your final patch in either <patch> </patch> tags or a ```patch ``` block."""
                output = self.agent.forward(prompt)

                prediction = {
                    "instance_id": instance_id,
                    "model_patch": output,
                    "model_name_or_path": self.benchmark_name,
                }

                file_out.write(json.dumps(prediction) + "\n")
                file_out.flush()

        self._sort_jsonl_alphanumeric_asc(
            self.preds_file,
        )

    def _sort_jsonl_alphanumeric_asc(self, jsonl_file: Path) -> None:
        """Sort a JSONL file by the 'instance_id' field in ascending order."""
        with open(jsonl_file, "r") as file:
            lines = [json.loads(line) for line in file]

        sorted_lines = sorted(lines, key=lambda x: x["instance_id"])

        with open(jsonl_file, "w") as file:
            for line in sorted_lines:
                file.write(json.dumps(line) + "\n")

    def _find_processed_ids(self) -> set | None:
        """For this given benchmark, return the instance_ids that have already been evaluated."""
        processed_ids = set()
        try:
            if self.preds_file.exists():
                with open(self.preds_file, "r") as file:
                    existing_preds = [json.loads(line) for line in file]
                processed_ids = {
                    (p["instance_id"], p["model_name_or_path"]) for p in existing_preds
                }
            else:
                return set()
        except Exception as e:
            print(f"Error reading {self.preds_file}:\n{e}")
            return set()
        return processed_ids


if __name__ == "__main__":
    MODELS_TO_BENCHMARK = [
        "deepseek-r1:8b",
        "llama3.2",
        "gemini-2.5-pro-exp-03-25",
    ]
    for model_name in MODELS_TO_BENCHMARK:
        model = ModelClient(model_name=model_name)
        agent = AgentBasic(model, max_retries=10)
        benchmark = AgentBenchmark(model_name=model_name, agent=agent)
        benchmark.run_benchmark_precomputed_retrieval(
            SWE_BENCH_LITE_DATASET, SWE_BENCH_BM25_40K_DATASET
        )
