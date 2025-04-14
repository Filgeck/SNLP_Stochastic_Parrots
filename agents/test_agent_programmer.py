import unittest
from unittest.mock import MagicMock
from agents.programmer import AgentProgrammer
from clients import ModelClient


class TestAgentProgrammer(unittest.TestCase):
    def setUp(self):
        # Mock ModelClient
        self.model_client = MagicMock(spec=ModelClient)
        self.agent = AgentProgrammer(model_client=self.model_client, max_retries=1)

        # Mock _query_and_extract to simplify testing
        self.agent._query_and_extract = MagicMock()

        # Assume base Agent's _get_files for simplicity
        def mock_get_files(text, strip_line_num):
            pattern = r"\[start of (.*?)\](.*?)\[end of \1\]"
            matches = re.findall(pattern, text, re.DOTALL)
            return {match[0].strip(): match[1].strip() for match in matches}

        self.agent._get_files = MagicMock(side_effect=mock_get_files)

    def test_fix_simple_bug(self):
        prompt = (
            "[start of test.py]\n"
            "def add(a, b):\n"
            "    return a - b  # Bug: should be addition\n"
            "[end of test.py]"
        )
        fixed_response = (
            "[start of test.py]\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "[end of test.py]"
        )
        self.agent._query_and_extract.return_value = fixed_response

        patch = self.agent.forward(prompt)
        self.assertIsNotNone(patch)
        self.assertIn("--- test.py", patch)
        self.assertIn("+++ test.py", patch)
        self.assertIn("-    return a - b", patch)
        self.assertIn("+    return a + b", patch)

    def test_no_changes(self):
        prompt = (
            "[start of test.py]\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "[end of test.py]"
        )
        fixed_response = prompt  # No changes
        self.agent._query_and_extract.return_value = fixed_response

        patch = self.agent.forward(prompt)
        self.assertEqual(patch, "")

    def test_empty_response(self):
        prompt = (
            "[start of test.py]\n"
            "def add(a, b):\n"
            "    return a - b\n"
            "[end of test.py]"
        )
        self.agent._query_and_extract.return_value = ""

        patch = self.agent.forward(prompt)
        self.assertIsNone(patch)

    # Add more test cases (e.g., multiple files, syntax errors, missing files)
    def test_multiple_files(self):
        prompt = (
            "[start of file1.py]\n"
            "def func1():\n"
            "    print('bug')\n"
            "[end of file1.py]\n"
            "[start of file2.py]\n"
            "def func2():\n"
            "    return 1 / 0\n"
            "[end of file2.py]"
        )
        fixed_response = (
            "[start of file1.py]\n"
            "def func1():\n"
            "    print('fixed')\n"
            "[end of file1.py]\n"
            "[start of file2.py]\n"
            "def func2():\n"
            "    return 1\n"
            "[end of file2.py]"
        )
        self.agent._query_and_extract.return_value = fixed_response

        patch = self.agent.forward(prompt)
        self.assertIn("--- file1.py", patch)
        self.assertIn("--- file2.py", patch)
        self.assertIn("-    print('bug')", patch)
        self.assertIn("+    print('fixed')", patch)
        self.assertIn("-    return 1 / 0", patch)
        self.assertIn("+    return 1", patch)


if __name__ == "__main__":
    unittest.main()