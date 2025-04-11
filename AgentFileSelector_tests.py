import unittest
from unittest.mock import MagicMock, patch
from typing import Literal, Tuple, List
import re
import os
import glob

# Import the base classes (or mock them as needed)
from clients import ModelClient, Retries

# Define the AgentFileSelector class (assumed to be imported in actual code)
from agents import Agent, AgentFileSelector

class TestAgentFileSelector(unittest.TestCase):
    """Test cases specifically for the AgentFileSelector class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock ModelClient
        self.model_client = MagicMock(spec=ModelClient)
        
        # Initialize the AgentFileSelector
        self.agent = AgentFileSelector(
            model_client=self.model_client,
            return_full_text=True,
            strip_line_num=True,
            max_retries=3
        )
        
        # Find all test files in the directory
        test_files_dir = "test_data"
        test_file_pattern = os.path.join(test_files_dir, "*.txt")
        test_file_paths = glob.glob(test_file_pattern)
        
        if len(test_file_paths) < 3:
            raise ValueError(f"Expected at least 3 test files in {test_files_dir}, found {len(test_file_paths)}")
        
        # Load the first 3 test files
        self.test_files = []
        self.test_file_paths = test_file_paths[:3]
        
        for file_path in self.test_file_paths:
            try:
                with open(file_path, 'r') as f:
                    self.test_files.append(f.read())
            except FileNotFoundError:
                raise ValueError(f"Test file {file_path} not found.")
        
        # Extract expected files for each test case from the test files
        self.expected_files_by_test = []
        for test_file in self.test_files:
            issue_text = self._extract_issue(test_file)
            files_text = self._extract_code(test_file)
            file_paths = self._extract_file_paths(files_text)
            #print("expected files: ", file_paths)
            
            # For testing purposes, we'll default to selecting the first two files
            # This would be replaced with actual expected files in a real scenario
            #self.expected_files_by_test.append(file_paths[:2])
            self.expected_files_by_test.append(file_paths)
    
    def _extract_issue(self, test_file):
        """Extract issue text from a test file."""
        match = re.search(r'<issue>(.*?)</issue>', test_file, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_code(self, test_file):
        """Extract code text from a test file."""
        match = re.search(r'<code>(.*?)</code>', test_file, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_file_paths(self, files_text):
        """Extract file paths from the files text."""
        file_pattern = r'\[start of (.*?)\]'
        return re.findall(file_pattern, files_text)
    
    def _parse_files_to_dict(self, files_text):
        """Parse files text into a dictionary of file paths and contents."""
        files_dict = {}
        file_pattern = r'\[start of (.*?)\](.*?)\[end of \1\]'
        for file_path, file_content in re.findall(file_pattern, files_text, re.DOTALL):
            files_dict[file_path] = file_content.strip()
        return files_dict

    def test_initialization(self):
        """Test that the AgentFileSelector initializes correctly with the right attributes."""
        self.assertEqual(self.agent.agent_name, "agent_file_selector")
        self.assertTrue(self.agent.return_full_text)
        self.assertTrue(self.agent.strip_line_num)
        self.assertEqual(self.agent.max_retries, 3)

    def test_select_by_batch(self):
        """Test the _select_by_batch method with valid input across test files."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                issue_text = self._extract_issue(test_file)
                files_text = self._extract_code(test_file)
                expected_files = self.expected_files_by_test[test_idx]
                
                # Create patch for _query_and_extract
                with patch.object(
                    Agent, 
                    '_query_and_extract', 
                    return_value="\n".join(expected_files)
                ) as mock_query:
                    
                    # Create patch for _get_files
                    with patch.object(
                        Agent, 
                        '_get_files', 
                        return_value=self._parse_files_to_dict(files_text)
                    ) as mock_get_files:
                        
                        # Call the method
                        result = self.agent._select_by_batch(issue_text, files_text)
                        
                        # Verify results
                        self.assertEqual(result, expected_files)
                        mock_query.assert_called_once()
                        
                        # Check prompt content
                        prompt = mock_query.call_args[0][0]
                        self.assertIn(issue_text, prompt)
                        self.assertIn("<selected>", prompt)

    def test_select_by_batch_invalid_file(self):
        """Test _select_by_batch with invalid file selection."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                issue_text = self._extract_issue(test_file)
                files_text = self._extract_code(test_file)
                
                # Create patch for _query_and_extract
                with patch.object(
                    Agent, 
                    '_query_and_extract', 
                    return_value="nonexistent_file.py"
                ) as mock_query:
                    
                    # Create patch for _get_files
                    with patch.object(
                        Agent, 
                        '_get_files', 
                        return_value=self._parse_files_to_dict(files_text)
                    ) as mock_get_files:
                        
                        # Expect a ValueError when a non-existent file is selected
                        with self.assertRaises(ValueError) as context:
                            self.agent._select_by_batch(issue_text, files_text)
                        
                        self.assertIn("does not exist", str(context.exception))

    def test_select_by_individual(self):
        """Test the _select_by_individual method with valid input."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                issue_text = self._extract_issue(test_file)
                files_text = self._extract_code(test_file)
                expected_files = self.expected_files_by_test[test_idx]
                file_paths = self._extract_file_paths(files_text)
                
                # Create responses based on expected files
                responses = []
                for file_path in file_paths:
                    if file_path in expected_files:
                        responses.append("Yes")
                    else:
                        responses.append("No")
                
                # Create patch for _query_and_extract
                with patch.object(
                    Agent, 
                    '_query_and_extract', 
                    side_effect=responses
                ) as mock_query:
                    
                    # Create patch for _get_files
                    with patch.object(
                        Agent, 
                        '_get_files', 
                        return_value=self._parse_files_to_dict(files_text)
                    ) as mock_get_files:
                        
                        # Call the method
                        result = self.agent._select_by_individual(issue_text, files_text)
                        
                        # Verify results
                        self.assertEqual(result, expected_files)
                        self.assertEqual(mock_query.call_count, len(file_paths))

    def test_select_by_individual_invalid_response(self):
        """Test _select_by_individual with invalid model response."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                issue_text = self._extract_issue(test_file)
                files_text = self._extract_code(test_file)
                file_paths = self._extract_file_paths(files_text)
                
                if not file_paths:
                    self.skipTest("No files found in this test file")
                
                # Create patch for _query_and_extract
                with patch.object(
                    Agent, 
                    '_query_and_extract', 
                    return_value="Maybe"
                ) as mock_query:
                    
                    # Create patch for _get_files
                    with patch.object(
                        Agent, 
                        '_get_files', 
                        return_value={file_paths[0]: "Sample content"}
                    ) as mock_get_files:
                        
                        # Expect a ValueError when model returns invalid response
                        with self.assertRaises(ValueError) as context:
                            self.agent._select_by_individual(issue_text, files_text)
                        
                        self.assertIn("Invalid response from model", str(context.exception))

    def test_forward_batch_method(self):
        """Test the forward method with batch selection."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                issue_text = self._extract_issue(test_file)
                files_text = self._extract_code(test_file)
                expected_files = self.expected_files_by_test[test_idx]
                
                # Create patch for _extract_tag
                with patch.object(
                    Agent, 
                    '_extract_tag', 
                    side_effect=[issue_text, files_text]
                ) as mock_extract_tag:
                    
                    # Create patch for _select_by_batch
                    with patch.object(
                        AgentFileSelector, 
                        '_select_by_batch', 
                        return_value=expected_files
                    ) as mock_select_by_batch:
                        
                        # Create patch for _format_output
                        with patch.object(
                            AgentFileSelector, 
                            '_format_output', 
                            return_value="test output"
                        ) as mock_format_output:
                            
                            # Call the method
                            selected_files, output_text = self.agent.forward(
                                text=test_file,
                                method="batch"
                            )
                            
                            # Verify results
                            self.assertEqual(selected_files, expected_files)
                            self.assertEqual(output_text, "test output")
                            mock_select_by_batch.assert_called_once_with(issue_text, files_text)

    def test_forward_individual_method(self):
        """Test the forward method with individual selection."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                issue_text = self._extract_issue(test_file)
                files_text = self._extract_code(test_file)
                expected_files = self.expected_files_by_test[test_idx]
                
                # Create patch for _extract_tag
                with patch.object(
                    Agent, 
                    '_extract_tag', 
                    side_effect=[issue_text, files_text]
                ) as mock_extract_tag:
                    
                    # Create patch for _select_by_individual
                    with patch.object(
                        AgentFileSelector, 
                        '_select_by_individual', 
                        return_value=expected_files
                    ) as mock_select_by_individual:
                        
                        # Create patch for _format_output
                        with patch.object(
                            AgentFileSelector, 
                            '_format_output', 
                            return_value="test output"
                        ) as mock_format_output:
                            
                            # Call the method
                            selected_files, output_text = self.agent.forward(
                                text=test_file,
                                method="individual"
                            )
                            
                            # Verify results
                            self.assertEqual(selected_files, expected_files)
                            self.assertEqual(output_text, "test output")
                            mock_select_by_individual.assert_called_once_with(issue_text, files_text)

    def test_forward_invalid_method(self):
        """Test forward method with an invalid selection method."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                issue_text = self._extract_issue(test_file)
                files_text = self._extract_code(test_file)
                
                # Create patch for _extract_tag
                with patch.object(
                    Agent, 
                    '_extract_tag', 
                    side_effect=[issue_text, files_text]
                ) as mock_extract_tag:
                    
                    with self.assertRaises(ValueError) as context:
                        self.agent.forward(
                            text=test_file,
                            method="invalid_method"
                        )
                    
                    self.assertIn("Invalid method", str(context.exception))

    def test_forward_missing_issue_tag(self):
        """Test forward method with missing issue tag."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                files_text = self._extract_code(test_file)
                
                # Create patch for _extract_tag
                with patch.object(
                    Agent, 
                    '_extract_tag', 
                    side_effect=[None, files_text]
                ) as mock_extract_tag:
                    
                    with self.assertRaises(ValueError) as context:
                        self.agent.forward(
                            text=test_file,
                            method="batch"
                        )
                    
                    self.assertIn("Could not extract <issue> tag", str(context.exception))

    def test_forward_missing_code_tag(self):
        """Test forward method with missing code tag."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                issue_text = self._extract_issue(test_file)
                
                # Create patch for _extract_tag
                with patch.object(
                    Agent, 
                    '_extract_tag', 
                    side_effect=[issue_text, None]
                ) as mock_extract_tag:
                    
                    with self.assertRaises(ValueError) as context:
                        self.agent.forward(
                            text=test_file,
                            method="batch"
                        )
                    
                    self.assertIn("Could not extract <code> tag", str(context.exception))

    def test_forward_with_custom_issue(self):
        """Test forward method with a custom issue."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                files_text = self._extract_code(test_file)
                expected_files = self.expected_files_by_test[test_idx]
                custom_issue = "This is a custom issue"
                
                # Create patch for _extract_tag
                with patch.object(
                    Agent, 
                    '_extract_tag', 
                    return_value=files_text
                ) as mock_extract_tag:
                    
                    # Create patch for _select_by_batch
                    with patch.object(
                        AgentFileSelector, 
                        '_select_by_batch', 
                        return_value=expected_files
                    ) as mock_select_by_batch:
                        
                        # Create patch for _format_output
                        with patch.object(
                            AgentFileSelector, 
                            '_format_output', 
                            return_value="test output"
                        ) as mock_format_output:
                            
                            # Call the method
                            selected_files, _ = self.agent.forward(
                                text=test_file,
                                method="batch",
                                custom_issue=custom_issue
                            )
                            
                            # Verify that the custom issue was used instead of extracted issue
                            mock_select_by_batch.assert_called_once_with(custom_issue, files_text)

    def test_format_output_return_full_text(self):
        """Test _format_output with return_full_text=True."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                files_text = self._extract_code(test_file)
                expected_files = self.expected_files_by_test[test_idx]
                
                # Set up the agent with return_full_text=True
                agent = AgentFileSelector(
                    model_client=self.model_client,
                    return_full_text=True,
                    strip_line_num=True
                )
                
                # Call the method
                output = agent._format_output(
                    test_file,
                    files_text,
                    expected_files
                )
                
                # Verify output
                self.assertIn("<issue>", output)
                self.assertIn("<code>", output)
                for file_path in expected_files:
                    self.assertIn(f"[start of {file_path}]", output)
                
                # Find files that were not selected
                all_files = self._extract_file_paths(files_text)
                unselected_files = [f for f in all_files if f not in expected_files]
                
                for file_path in unselected_files:
                    if file_path: # Skip empty paths
                        self.assertNotIn(f"[start of {file_path}]", output)

    def test_format_output_return_code_only(self):
        """Test _format_output with return_full_text=False."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                files_text = self._extract_code(test_file)
                expected_files = self.expected_files_by_test[test_idx]
                
                if not expected_files:
                    self.skipTest("No expected files for this test case")
                
                # Set up the agent with return_full_text=False
                agent = AgentFileSelector(
                    model_client=self.model_client,
                    return_full_text=False,
                    strip_line_num=True
                )
                
                # Call the method
                output = agent._format_output(
                    test_file,
                    files_text,
                    expected_files
                )
                
                # Verify output
                self.assertIn("<code>", output)
                self.assertIn("</code>", output)
                self.assertNotIn("<issue>", output)
                
                for file_path in expected_files:
                    self.assertIn(f"[start of {file_path}]", output)
                
                # Find files that were not selected
                all_files = self._extract_file_paths(files_text)
                unselected_files = [f for f in all_files if f not in expected_files]
                
                for file_path in unselected_files:
                    if file_path: # Skip empty paths
                        self.assertNotIn(f"[start of {file_path}]", output)

    def test_format_output_no_selected_files(self):
        """Test _format_output with no selected files."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                files_text = self._extract_code(test_file)
                
                # Call the method with empty selected files list
                output = self.agent._format_output(
                    test_file,
                    files_text,
                    []
                )
                
                # Verify output has empty code block
                if self.agent.return_full_text:
                    self.assertIn("<code>", output)
                    self.assertIn("</code>", output)
                    
                    all_files = self._extract_file_paths(files_text)
                    for file_path in all_files:
                        if file_path: # Skip empty paths
                            self.assertNotIn(f"[start of {file_path}]", output)
                else:
                    self.assertEqual(output.strip(), "<code>\n</code>")

    def test_format_output_no_code_block(self):
        """Test _format_output with input that doesn't contain code block."""
        # Create test input without code block
        input_without_code = "<issue>Test issue</issue>"
        
        # Expect ValueError when no code block is found
        with self.assertRaises(ValueError) as context:
            self.agent._format_output(
                input_without_code,
                "Some files text",
                ["file1.py"]
            )
        
        self.assertIn("Could not find the <code> block", str(context.exception))
        
    def test_strip_line_num_true(self):
        """Test that line numbers are stripped when strip_line_num=True."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                files_text = self._extract_code(test_file)
                
                files_text_with_line_nums = files_text
                # Add line numbers to the files_text for testing
                #files_text_with_line_nums = self._add_line_numbers_to_files(files_text)
                #print(files_text_with_line_nums)
                
                # Initialize the agent with strip_line_num=True
                agent = AgentFileSelector(
                    model_client=self.model_client,
                    return_full_text=True,
                    strip_line_num=True
                )
                
                # Get files with line numbers, but strip_line_num=True should remove them
                files_dict = agent._get_files(files_text_with_line_nums, agent.strip_line_num)
                
                # Verify line numbers were stripped from all files
                for file_path, content in files_dict.items():
                    # Each line should not start with a number followed by a space
                    for line in content.splitlines():
                        if line.strip():  # Skip empty lines
                            self.assertFalse(re.match(r'^\d+ ', line), 
                                            f"Line still has number prefix: '{line}'")

    def test_strip_line_num_false(self):
        """Test that line numbers are preserved when strip_line_num=False."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                files_text = self._extract_code(test_file)
                
                files_text_with_line_nums = files_text
                # Add line numbers to the files_text for testing
                #files_text_with_line_nums = self._add_line_numbers_to_files(files_text)
                #print(files_text_with_line_nums)
                
                # Initialize the agent with strip_line_num=False
                agent = AgentFileSelector(
                    model_client=self.model_client,
                    return_full_text=True,
                    strip_line_num=False
                )
                
                # Get files with line numbers, strip_line_num=False should preserve them
                files_dict = agent._get_files(files_text_with_line_nums, agent.strip_line_num)
                
                # Verify line numbers were preserved in all files
                for file_path, content in files_dict.items():
                    # Find non-empty lines
                    content_lines = [line for line in content.splitlines() if line.strip()]
                    if content_lines:
                        # At least some lines should have line numbers
                        lines_with_numbers = [line for line in content_lines if re.match(r'^\d+ ', line)]
                        self.assertTrue(len(lines_with_numbers) > 0, 
                                    f"No line numbers found in content for {file_path}")

    def _add_line_numbers_to_files(self, files_text):
        """Helper method to add line numbers to files for testing."""
        # Find all file blocks
        file_block_pattern = r"(\[start of (.*?)\]\s*?\n)(.*?)(\n\[end of \2\])"
        
        def add_numbers_to_block(match):
            start = match.group(1)
            file_path = match.group(2)
            content = match.group(3)
            end = match.group(4)
            
            # Add line numbers to each line of content
            lines = content.splitlines()
            numbered_lines = []
            for i, line in enumerate(lines, 1):
                if line.strip():  # Only number non-empty lines
                    numbered_lines.append(f"{i} {line}")
                else:
                    numbered_lines.append(line)
            
            numbered_content = "\n".join(numbered_lines)
            return f"{start}{numbered_content}{end}"
        
        return re.sub(file_block_pattern, add_numbers_to_block, files_text, flags=re.DOTALL)

    def test_select_by_batch_with_strip_line_num(self):
        """Test that _select_by_batch correctly handles line numbers based on strip_line_num."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                issue_text = self._extract_issue(test_file)
                files_text = self._extract_code(test_file)
                
                files_text_with_line_nums = files_text
                # Add line numbers to files_text
                #files_text_with_line_nums = self._add_line_numbers_to_files(files_text)
                #print(files_text_with_line_nums)
                
                # Expected files for this test
                expected_files = self.expected_files_by_test[test_idx]
                
                # Create a dictionary from the files_text for both tests
                files_dict_with_nums = self._parse_files_to_dict(files_text_with_line_nums)
                
                # Create a stripped version for the strip=True test
                files_dict_stripped = {}
                for path, content in files_dict_with_nums.items():
                    stripped_content = '\n'.join([re.sub(r'^\d+ ', '', line) for line in content.splitlines()])
                    files_dict_stripped[path] = stripped_content
                
                # Test with strip_line_num=True
                agent_strip = AgentFileSelector(
                    model_client=self.model_client,
                    return_full_text=True,
                    strip_line_num=True
                )
                
                # Use side_effect to properly mock _get_files
                with patch.object(Agent, '_get_files', return_value=files_dict_stripped) as mock_get_files:
                    # Mock _query_and_extract to return fixed result
                    with patch.object(Agent, '_query_and_extract', return_value="\n".join(expected_files)) as mock_query:
                        
                        # Call the method
                        result = agent_strip._select_by_batch(issue_text, files_text_with_line_nums)
                        
                        # Verify strip_line_num parameter was passed
                        mock_get_files.assert_called_once()
                        self.assertEqual(mock_get_files.call_args[0][1], True)
                
                # Test with strip_line_num=False
                agent_no_strip = AgentFileSelector(
                    model_client=self.model_client,
                    return_full_text=True,
                    strip_line_num=False
                )
                
                # Use side_effect to properly mock _get_files
                with patch.object(Agent, '_get_files', return_value=files_dict_with_nums) as mock_get_files:
                    # Mock _query_and_extract to return fixed result
                    with patch.object(Agent, '_query_and_extract', return_value="\n".join(expected_files)) as mock_query:
                        
                        # Call the method
                        result = agent_no_strip._select_by_batch(issue_text, files_text_with_line_nums)
                        
                        # Verify strip_line_num parameter was passed
                        mock_get_files.assert_called_once()
                        self.assertEqual(mock_get_files.call_args[0][1], False)
                        
                        # Verify at least some file has line numbers
                        found_line_numbers = False
                        for file_path, content in files_dict_with_nums.items():
                            content_lines = [line for line in content.splitlines() if line.strip()]
                            if any(re.match(r'^\d+ ', line) for line in content_lines):
                                found_line_numbers = True
                                break
                        
                        self.assertTrue(found_line_numbers, "No line numbers found in any file content")

    def test_select_by_individual_with_strip_line_num(self):
        """Test that _select_by_individual correctly handles line numbers based on strip_line_num."""
        for test_idx, test_file in enumerate(self.test_files):
            with self.subTest(f"Testing with test file {test_idx + 1}"):
                issue_text = self._extract_issue(test_file)
                files_text = self._extract_code(test_file)
                file_paths = self._extract_file_paths(files_text)
                
                if not file_paths:
                    self.skipTest("No files found in this test file")
                
                files_text_with_line_nums = files_text
                # Add line numbers to files_text
                #files_text_with_line_nums = self._add_line_numbers_to_files(files_text)
                #print(files_text_with_line_nums)
                
                # Create a dictionary from the files_text for both tests
                files_dict_with_nums = self._parse_files_to_dict(files_text_with_line_nums)
                
                # Create a stripped version for the strip=True test
                files_dict_stripped = {}
                for path, content in files_dict_with_nums.items():
                    stripped_content = '\n'.join([re.sub(r'^\d+ ', '', line) for line in content.splitlines()])
                    files_dict_stripped[path] = stripped_content
                
                # Expected files for this test
                expected_files = self.expected_files_by_test[test_idx]
                
                # Get responses based on expected files
                responses = []
                for file_path in file_paths:
                    if file_path in expected_files:
                        responses.append("Yes")
                    else:
                        responses.append("No")
                
                # Test with strip_line_num=True
                agent_strip = AgentFileSelector(
                    model_client=self.model_client,
                    return_full_text=True,
                    strip_line_num=True
                )
                
                # Use return_value to properly mock _get_files
                with patch.object(Agent, '_get_files', return_value=files_dict_stripped) as mock_get_files:
                    # Mock _query_and_extract to return appropriate responses
                    with patch.object(Agent, '_query_and_extract', side_effect=responses) as mock_query:
                        
                        # Call the method
                        result = agent_strip._select_by_individual(issue_text, files_text_with_line_nums)
                        
                        # Verify strip_line_num parameter was passed
                        mock_get_files.assert_called_once()
                        self.assertEqual(mock_get_files.call_args[0][1], True)
                
                # Test with strip_line_num=False
                agent_no_strip = AgentFileSelector(
                    model_client=self.model_client,
                    return_full_text=True,
                    strip_line_num=False
                )
                
                # Use return_value to properly mock _get_files
                with patch.object(Agent, '_get_files', return_value=files_dict_with_nums) as mock_get_files:
                    # Mock _query_and_extract to return appropriate responses
                    with patch.object(Agent, '_query_and_extract', side_effect=responses) as mock_query:
                        
                        # Call the method
                        result = agent_no_strip._select_by_individual(issue_text, files_text_with_line_nums)
                        
                        # Verify strip_line_num parameter was passed
                        mock_get_files.assert_called_once()
                        self.assertEqual(mock_get_files.call_args[0][1], False)
                        
                        # Verify at least some file has line numbers
                        found_line_numbers = False
                        for file_path, content in files_dict_with_nums.items():
                            content_lines = [line for line in content.splitlines() if line.strip()]
                            if any(re.match(r'^\d+ ', line) for line in content_lines):
                                found_line_numbers = True
                                break
                        
                        self.assertTrue(found_line_numbers, "No line numbers found in any file content")


if __name__ == '__main__':
    unittest.main()