import os
import unittest
from tempfile import TemporaryDirectory
from shutil import copyfile

from modules.ModuleGetTargetInfo import GetTargetPicOrTextInfo


class TestGetTargetPicOrTextInfo(unittest.TestCase):
    def setUp(self):
        # Set up a temporary directory
        self.test_dir = TemporaryDirectory()
        # Create some test text files
        self.test_file1 = os.path.join(self.test_dir.name, "test1.txt")
        with open(self.test_file1, 'w', encoding='utf-8') as f:
            f.write("Line 1\nLine 2\nLine 3")
        self.test_file2 = os.path.join(self.test_dir.name, "test2.txt")
        with open(self.test_file2, 'w', encoding='utf-8') as f:
            f.write("Another line 1\nAnother line 2")

    def test_reading_text_files(self):
        # Instantiate the class with a dummy module name and the path to the temp directory
        reader = GetTargetPicOrTextInfo("自定义", self.test_dir.name)
        # Simulate the method to get target info (specifically text data)
        _, _, _, _, _, text_data = reader.get_target_info
        # Check the contents of the text files
        expected_contents = {
            "test1.txt": ["Line 1", "Line 2", "Line 3"],
            "test2.txt": ["Another line 1", "Another line 2"]
        }
        self.assertEqual(text_data, expected_contents)

    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

if __name__ == '__main__':
    unittest.main()
