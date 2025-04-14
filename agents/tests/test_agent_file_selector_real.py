import pytest
from pathlib import Path
from agents import AgentFileSelector
from clients import ModelClient

# Directory where your example inputs are stored
TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def model_client():
    return ModelClient(model_name="llama3.2")  # or 'gemini-1.5-pro'


@pytest.fixture
def agent(model_client):
    return AgentFileSelector(
        model_client=model_client, return_full_text=False, strip_line_num=True
    )


@pytest.fixture(params=list(TEST_DATA_DIR.glob("*.txt"))[:1])
def example_input(request):
    return request.param.read_text()


def test_forward_batch_selection(agent, example_input):
    selected_files, output = agent.forward(text=example_input, method="batch")
    print("\n[Batch Selection]")
    print("Selected files:", selected_files)
    # print("Model Output:\n", output)

    assert isinstance(selected_files, list)
    # assert all(path.endswith(".py") for path in selected_files)


def test_forward_individual_selection(agent, example_input):
    selected_files, output = agent.forward(text=example_input, method="individual")
    print("\n[Individual Selection]")
    print("Selected files:", selected_files)
    # print("Model Output:\n", output)

    assert isinstance(selected_files, list)
    # assert all(path.endswith(".py") for path in selected_files)
